import abc
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
from .superpoint import SuperPoint
from .lightglue import LightGlue
from .geometry_utils import *

class LightGluePosePolicy(nn.Module):
    def __init__(
        self,
        goal_sensor_uuid='imagegoal',
        hidden_size=256,
        num_keypoints=4096,
    ):
        super().__init__()
        self.net = LightGluePoseKGNet(
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                num_keypoints=num_keypoints
            )
        self.dim_actions = 1
        self.hidden_size = hidden_size

        self.mlp = torch.nn.Linear(int(self.hidden_size/2), 6)
        
    def infer(
        self,
        observations,
    ):

        features, matches, alphas, kpts0, kpts1, mscores, inlier_masks, mkp3d = self.net(observations)
        pred = self.mlp(features)

        pred = torch.cat([pred[..., 0:3], qexp_t(pred[..., 3:])], dim=-1) # x y z x y z w

        return pred, matches, alphas, kpts0, kpts1, mscores, inlier_masks, mkp3d

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight, gain=gain)
    bias_init(module.bias)
    return module


class LightGluePoseKGNet(nn.Module, metaclass=abc.ABCMeta):

    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, hidden_size, goal_sensor_uuid, num_keypoints=4096):
        super().__init__()

        self._hidden_size = hidden_size
        
        # SuperPoint+LightGlue
        self.extractor = SuperPoint(max_num_keypoints=num_keypoints).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher
        
        for p in self.extractor.parameters():
            p.requires_grad = False
        for p in self.matcher.parameters():
            p.requires_grad = False # True # False
        self.matcher.compile(mode='reduce-overhead')
        
        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.constant_(x,0))

        self.gat1 = GATv2Conv(4, int(self._hidden_size/4), heads=4, concat=True, add_self_loops=True, edge_dim=2) 
        self.gat2 = GATv2Conv(int(self._hidden_size), int(self._hidden_size/4), heads=4, concat=True, add_self_loops=True, edge_dim=2)
        self.gat3 = GATv2Conv(int(self._hidden_size), int(self._hidden_size/4), heads=4, concat=True, add_self_loops=True, edge_dim=2)
        self.gat4 = GATv2Conv(int(self._hidden_size), int(self._hidden_size/2), heads=4, concat=False, add_self_loops=True, edge_dim=2)

        self.GELU = nn.GELU()
        
        self.mlp_desc = nn.Linear(4, self._hidden_size)
        # self.mlp_ol_0 = nn.Linear(self._hidden_size, 1) # nn.Sequential(nn.Linear(self._hidden_size, 64), nn.GELU(), nn.Linear(64, 1))
        self.mlp_ol_1 = nn.Sequential(nn.Linear(self._hidden_size, 64), nn.GELU(), nn.Linear(64, 1)) # nn.Linear(self._hidden_size, 1)
        self.mlp_ol_2 = nn.Sequential(nn.Linear(self._hidden_size, 64), nn.GELU(), nn.Linear(64, 1)) # nn.Linear(self._hidden_size, 1)
        self.mlp_ol_3 = nn.Sequential(nn.Linear(self._hidden_size, 64), nn.GELU(), nn.Linear(64, 1)) # nn.Linear(self._hidden_size, 1)

        self.K = torch.tensor([[526.5249633789062, 0.0,  651.398681640625],
                               [0.0, 526.5249633789062, 360.3771057128906],
                               [0.0,               0.0,               1.0]])
        self.K_inv = torch.linalg.inv(self.K)
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 0 #self.state_encoder.num_recurrent_layers #+ self.state_encoder_glob.num_recurrent_layers

    def normalize_keypoints(self, kpts, image_shape):
        """ Normalize keypoints locations based on image image_shape"""
        _, _, height, width = image_shape
        one = kpts.new_tensor(1)
        size = torch.stack([one*width, one*height])[None]
        center = size / 2
        scaling = size.max(1, keepdim=True).values * 0.5 # 0.7 originally
        return (kpts - center[:, None, :]) / scaling[:, None, :]     

    def forward(self, observations, sampson=False, mlp=None, lightglue_mode=False): # sampson pooling
        
        image0 = observations['imagegoal'] #/ 255.0
        image1 = observations['rgb'] #/ 255.0
        depth1 = observations['depth']
        # assert (depth1 <= 1.0).all(), "depth nor normalized"
        n_batch = observations['rgb'].shape[0]
        
        self.extractor.eval()
        self.matcher.eval()
        with torch.no_grad():
            # extract local features
            feats0 = self.extractor.extract(image0) #.permute(0,3,1,2))  # auto-resize the image, disable with resize=None
            feats1 = self.extractor.extract(image1) #.permute(0,3,1,2))
            # match the features
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        if lightglue_mode is True:
            return matches01

        descs = []
        alphas = []
        n_match = []
        inlier_masks = []
        # if sampson is True:
        sampson_err = torch.tensor(0.).to(observations['rgb'].device)
        outlier_err = torch.tensor(0.).to(observations['rgb'].device)
        n = 0.
        
        self.K_inv = self.K_inv.to(observations['rgb'].device)

        for i in range(n_batch): # for each batch element   

            matches = matches01['matches'][i]  # indices with shape (K,2)
            mscores = matches01['scores'][i]
            kpts0 = feats0['keypoints'][i][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            kpts1 = feats1['keypoints'][i][matches[..., 1]]  # coordinates in image #1, shape (K,2)
            desc1 = feats1['descriptors'][i][matches[..., 1]]   # shape (K,256) 
            # score1 = feats1['keypoint_scores'][i][matches[..., 1]]   # shape (K)
            mask_total = torch.zeros(desc1.shape[0]).bool().to(observations['rgb'].device) # shape (K,)

            mkp3d = None
            if observations['enable_depth'] is True:
                # mkp3d = observations['interp'](kpts1[:,0].detach().cpu().numpy(), kpts1[:,1].detach().cpu().numpy()) # shape (K,3)
                # mkp3d = torch.from_numpy(mkp3d).to(observations['rgb'].device, dtype=torch.float32)
                # # pt_cam = (m_pose_c2w_q * pt_w + m_pose_c2w_t), pt_cam(2) 
                # T_w2db = observations['db_pose'].to(observations['rgb'].device, dtype=torch.float32)
                # T_db2w = torch.linalg.inv(T_w2db)
                # mkp3d_cam = torch.matmul(mkp3d, T_db2w[:3,:3].T) + T_db2w[:3,3][None] # shape (K,3)   
                # depths1 = mkp3d_cam[:,2][:,None] # shape (K,1)
                # depths1 = depths1.clamp(0, 20.)/20. # normalize depth
                # valid = ~torch.any(torch.isnan(mkp3d), dim=-1)
                # depths1 = depths1[valid, ...]   # shape (K,1)
                # kpts0 = kpts0[valid, ...]   # shape (K,2)
                # kpts1 = kpts1[valid, ...]   # shape (K,2)
                # desc1 = desc1[valid, ...]   # shape (K,256)
                # mask_total = mask_total[valid]
                # mscores = mscores[valid]
                # matches = matches[valid, ...]

                depths1 = depth1[i, kpts1[:,1].long(), kpts1[:,0].long(), :]  # shape (K,1)
                valid = torch.all(torch.isfinite(depths1), dim=-1)
                depths1 = depths1[valid, ...]   # shape (K,1)
                assert (depths1 <= 1.0).all(), "depth nor normalized"
                kpts0 = kpts0[valid, ...]   # shape (K,2)
                kpts1 = kpts1[valid, ...]   # shape (K,2)
                desc1 = desc1[valid, ...]   # shape (K,256)
                mask_total = mask_total[valid]
                mscores = mscores[valid]
                matches = matches[valid, ...]

            if len(matches) < 10:
                descs.append(torch.zeros(self._hidden_size//2).to(observations['rgb'].device))
                alphas.append((torch.zeros(2,0).cpu().numpy(), torch.zeros(0,4).cpu().numpy()))
                num_inlier = 0
            else:
                # Ver. 5: normalized coord. + depth 
                kpts0_emb = torch.matmul(torch.nn.functional.pad(kpts0, [0, 1], value=1), self.K_inv.T)[:,0:2]
                kpts1_emb = torch.matmul(torch.nn.functional.pad(kpts1, [0, 1], value=1), self.K_inv.T)[:,0:2]
                # depths1 = depth1[i, kpts1[:,1].long(), kpts1[:,0].long(), :] # shape (K,1)
                if observations['enable_depth'] is True:
                    desc = torch.cat([kpts1_emb, depths1, kpts0_emb], dim=-1) 
                    # kpts1_coord = torch.cat([kpts1_emb*depths1, depths1], dim=-1)
                else:
                    desc = torch.cat([kpts1_emb, kpts0_emb], dim=-1)

                dist = torch.cdist(desc1, desc1, p=2.0)
                # dist = torch.cdist(kpts1_coord, kpts1_coord, p=2.0) #  for normalized coord. + depth
                _, index = dist.topk(k=min(4, len(matches)), dim=-1, largest=False) # False=knn, True=kfn
                dense_ind =  torch.zeros(len(matches),len(matches)).to(observations['rgb'].device).scatter_(-1, index, 1.)
                edge_index, _ = dense_to_sparse(dense_ind)
                edge_emb = kpts1_emb[edge_index[0,:]] - kpts1_emb[edge_index[1,:]]
                edge_emb = edge_emb.detach()

                # gat1
                desc_gat, alpha = self.gat1(desc, edge_index, edge_emb, return_attention_weights=True)
                desc_gat = self.GELU(desc_gat)
                desc = desc_gat + self.mlp_desc(desc) #self.GELU(self.mlp_desc(desc))
                ol1 = self.mlp_ol_1(desc_gat)
                # outlier_err += self.criterion(ol1, sampson_filter)
                # generate pooling mask
                m = torch.distributions.bernoulli.Bernoulli(logits=ol1)
                mask1 = m.sample().bool().squeeze()
                # print('mask1: ', mask1.float().sum())
                if mask1.float().sum() >= 10: # exception handling - # of inliers: too much pooling -> leave top-k score match or just not pooling
                    # define masked desc and sampson_filter
                    desc = desc[mask1, ...]
                    # sampson_filter = sampson_filter[mask1, ...]
                    # define dense_ind, edge_index, kpts1_emb, edge_emb
                    dense_ind = dense_ind[mask1,:][:,mask1]
                    edge_index, _ = dense_to_sparse(dense_ind)
                    kpts1_emb = kpts1_emb[mask1, ...]
                    edge_emb = kpts1_emb[edge_index[0,:]] - kpts1_emb[edge_index[1,:]]
                    edge_emb = edge_emb.detach()      
                    mask_total[mask1] = True 
                else:
                    mask1 = torch.ones_like(mask1).bool().to(observations['rgb'].device)

                # gat2
                desc_gat = self.GELU(self.gat2(desc, edge_index, edge_emb))
                desc = desc_gat + desc
                ol2 = self.mlp_ol_2(desc_gat)
                # outlier_err += self.criterion(ol2, sampson_filter)
                # generate pooling mask
                m = torch.distributions.bernoulli.Bernoulli(logits=ol2)
                mask2 = m.sample().bool().squeeze()
                # print('mask2: ', mask2.float().sum())
                if mask2.float().sum() >= 10:
                    # define desc, sampson_filter
                    desc = desc[mask2, ...]
                    # sampson_filter = sampson_filter[mask2, ...]
                    # define dense_ind, edge_index, kpts1_emb, edge_emb
                    dense_ind = dense_ind[mask2,:][:,mask2]
                    edge_index, _ = dense_to_sparse(dense_ind)
                    kpts1_emb = kpts1_emb[mask2, ...]
                    edge_emb = kpts1_emb[edge_index[0,:]] - kpts1_emb[edge_index[1,:]]
                    edge_emb = edge_emb.detach()
                    mask_total[mask1][mask2] = True
                else:
                    mask2 = torch.ones_like(mask2).bool().to(observations['rgb'].device)

                # gat3
                desc_gat = self.GELU(self.gat3(desc, edge_index, edge_emb))
                desc = desc_gat + desc
                ol3 = self.mlp_ol_3(desc_gat)
                # outlier_err += self.criterion(ol3, sampson_filter)
                # generate pooling mask
                m = torch.distributions.bernoulli.Bernoulli(logits=ol3)
                mask3 = m.sample().bool().squeeze()
                # print('mask3: ', mask3.float().sum())
                if mask3.float().sum() >= 10:
                    # define desc, sampson_filter
                    desc = desc[mask3, ...]
                    # sampson_filter = sampson_filter[mask3, ...]
                    # define dense_ind, edge_index, kpts1_emb, edge_emb
                    dense_ind = dense_ind[mask3,:][:,mask3]
                    edge_index, _ = dense_to_sparse(dense_ind)
                    kpts1_emb = kpts1_emb[mask3, ...]
                    edge_emb = kpts1_emb[edge_index[0,:]] - kpts1_emb[edge_index[1,:]]
                    edge_emb = edge_emb.detach()
                    mask_total[mask1][mask2][mask3] = True
                else:
                    mask3 = torch.ones_like(mask3).bool().to(observations['rgb'].device)

  
                # gat4
                desc_ = self.GELU(self.gat4(desc, edge_index, edge_emb))
                desc, _ = torch.max(desc_,dim=0)
                descs.append(desc)
                alphas.append((alpha[0].detach().cpu().numpy(), alpha[1].detach().cpu().numpy()))
              

            n_match.append(len(matches))
            inlier_masks.append(mask_total)

        x = torch.stack(descs, dim=0)
        inlier_masks = torch.concatenate(inlier_masks, dim=0)
        
        return x , n_match, alphas, kpts0, kpts1, mscores, inlier_masks, mkp3d
    
class PoseNetCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0,
                 saq=-3, sas=0, sao=0, learn_beta=True):
        super(PoseNetCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.sas = nn.Parameter(torch.Tensor([sas]), requires_grad=learn_beta)
        self.sao = nn.Parameter(torch.Tensor([sao]), requires_grad=learn_beta)

class PPO(nn.Module):
    def __init__(
        self,
        actor_critic
    ) -> None:

        super().__init__()

        self.actor_critic = actor_critic
        self.criterion = PoseNetCriterion(sax=0, saq=-3, sas=0, learn_beta=True)