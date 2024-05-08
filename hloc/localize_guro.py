import argparse
from pathlib import Path
import numpy as np
import h5py
import scipy
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
import quaternion
from scipy.io import loadmat
import torch
from tqdm import tqdm
import pickle
import cv2
import pycolmap

from . import logger
from .utils.parsers import parse_retrieval, names_to_pair

from .habitat.kgnet import PPO, LightGluePosePolicy
from .habitat.geometry_utils import *
from PIL import Image
import torchvision.transforms as transforms

def _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.

    """

    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = np.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        points = np.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[...,j] = item
    else:
        points = np.asanyarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points

class NearestNDInterpolatorQuery(NearestNDInterpolator):
    def __init__(self, x, y, rescale=False, tree_options=None):
        NearestNDInterpolator.__init__(self, x, y, rescale=rescale, tree_options=tree_options)

    def __call__(self, *args, **query_options):
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)
        
        # Flatten xi for the query
        xi_flat = xi.reshape(-1, xi.shape[-1])
        original_shape = xi.shape
        flattened_shape = xi_flat.shape

        dist, i = self.tree.query(xi_flat, **query_options)
        valid_mask = np.isfinite(dist)

        # create a holder interp_values array and fill with nans.
        if self.values.ndim > 1:
            interp_shape = flattened_shape[:-1] + self.values.shape[1:]
        else:
            interp_shape = flattened_shape[:-1]

        if np.issubdtype(self.values.dtype, np.complexfloating):
            interp_values = np.full(interp_shape, np.nan, dtype=self.values.dtype)
        else:
            interp_values = np.full(interp_shape, np.nan)

        interp_values[valid_mask] = self.values[i[valid_mask], ...]

        if self.values.ndim > 1:
            new_shape = original_shape[:-1] + self.values.shape[1:]
        else:
            new_shape = original_shape[:-1]
        interp_values = interp_values.reshape(new_shape)

        return interp_values

def eval_performance(results, image_dir, radius=[0.05, 0.10, 0.25, 0.5, 1]):
    with open(str(results) + "_logs.pkl", "rb") as f:
        logs = pickle.load(f)
    queries = list(logs["loc"].keys())
    
    pos_err, angle_err = [], []
    recall_within_radius = np.zeros([len(radius)])
    for qname in queries:
        loc = logs["loc"][qname]
        qvec, tvec = loc["PnP_ret"]["qvec"], loc["PnP_ret"]["tvec"]
        quat = quaternion.from_float_array(qvec)
        tvec = -quaternion.rotate_vectors(quat.inverse(), tvec)
        # rmat = quaternion.as_rotation_matrix(quat.inverse())

        with open(Path(image_dir, qname.split('.')[0] + '.txt'),'r') as f:
            line1 = f.readline()
        x, y, z, q_x, q_y, q_z, q_w = map(float, line1.split())
        quat_gt = quaternion.from_float_array([q_w, q_x, q_y, q_z])
        # rmat_gt = quaternion.as_rotation_matrix(quat)
        tvec_gt = np.array([x, y, z])

        pos_err.append(np.linalg.norm(tvec - tvec_gt))
        angle_err.append(angle_between_quaternions(quat.inverse(), quat_gt)*180/np.pi)
        recall_within_radius += np.array([np.any(np.linalg.norm(tvec - tvec_gt) < r) for r in radius])
        
    recall_within_radius /= len(queries)
    pos_err, angle_err = np.array(pos_err), np.array(angle_err)
    print("Top-5 largest error queries: ", [queries[i] for i in np.argsort(-pos_err)[:5]])
    print("pos mae: %.3f"% pos_err.mean(), "angle mae: %.3f"% angle_err.mean())
    print("pos rmse: %.3f"% ((pos_err**2).mean())**0.5, "angle rmse: %.3f"% ((angle_err**2).mean())**0.5)
    return pos_err, angle_err, recall_within_radius

def eval_retrieval_recall(retrieval, image_dir, radius=2.5, topks=[5]):
    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())
    topk_recalls = np.zeros([len(topks)])
    for qname in queries:
        retrieved = retrieval_dict[qname]
        tvecs = []        
        for r in retrieved:
            with open(Path(image_dir, r.split('.')[0] + '.txt'),'r') as f:
                line = f.readline()
            _, _, _, _, px, py, pz = map(float, line.split())
            tvecs.append([px, py, pz])
        tvecs = np.array(tvecs)

        with open(Path(image_dir, qname.split('.')[0] + '.txt'),'r') as f:
            line1 = f.readline()
        x, y, z, _, _, _, _ = map(float, line1.split())
        tvec_gt = np.array([[x, y, z]])

        for i in range(len(topks)):
            pos_err = np.linalg.norm(tvecs[:topks[i]] - tvec_gt, axis=-1)
            topk_recalls[i] += np.any(pos_err < radius)
    topk_recalls /= len(queries)

    return topk_recalls


def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file, topk=None,
                      skip=None, interp='linear'):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    cx = 651.398681640625 #325.6990051269531 #.5 * width
    cy = 360.3771057128906 #180.189453125 #.5 * height
    focal_length = 526.5249633789062 #263.27886962890625 #4032. * 28. / 36.

    all_mkpq = [] # matched keypoint coordinates in query images
    all_mkpr = [] # matched keypoint coordinates in retrieved images
    all_mkp3d = [] # matched keypoint 3d coordinate
    all_indices = []
    kpq = feature_file[q]['keypoints'].__array__()
    num_matches = 0

    if topk is None:
        topk = len(retrieved)

    # re-ranking using the number of the local descriptor matches
    num_match_arr = []
    for i, r in enumerate(retrieved):
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]['matches0'].__array__()
        v = (m > -1)

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_match_arr.append(len(mkpq))
    
    num_match_arr = np.array(num_match_arr)
    retrieved = [retrieved[i] for i in np.argsort(-num_match_arr)]

    for i, r in enumerate(retrieved):
        if i == topk:
            break
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]['matches0'].__array__()
        v = (m > -1)

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        # with open(Path(dataset_dir, r.split('_')[0] + '.txt'),'r') as f:
        with open(Path(dataset_dir, r.split('.')[0] + '.txt'),'r') as f:
            lines = f.readlines()

        # [Option2] depth image to camera coordinate and then to world coordinate
        depth_img = cv2.imread(str(dataset_dir / r.split('.')[0]) + '.tiff', cv2.IMREAD_ANYDEPTH)
        h, w = depth_img.shape
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        z = depth_img.flatten()
        # camera coordinate
        x = (u - cx) * z / focal_length
        y = (v - cy) * z / focal_length
        all_rp3d_cam = np.stack([x, y, z], axis=-1) 
        # world coordinate
        qx, qy, qz, qw, px, py, pz = map(float, lines[0].split())
        q_w2c = quaternion_from_coeff([qx, qy, qz, qw])
        t_w2c = np.array([px, py, pz])
        rmat = quaternion.as_rotation_matrix(q_w2c)
        all_rp3d = np.matmul(all_rp3d_cam, rmat.T) + t_w2c
        all_rp3d = all_rp3d.reshape([h, w, 3])
        # all_rpr = np.stack([u, v], axis=-1)
        # filter out mkpr which are not included in all_rpr
        mkp3d = all_rp3d[mkpr[:,1].astype(int), mkpr[:,0].astype(int)]
        valid = np.all(np.isfinite(mkp3d), axis=-1)
        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

        # [Option1] depth from r3live rendered points
        # all_uvxyz = lines[1:]
        # all_rpr = [] # rendered points retrieval
        # all_rp3d = [] # rendered points 3d coordinate
        # for uvxyz in all_uvxyz:
        #     u, v, x, y, z, d = map(float, uvxyz.split(' '))
        #     rpr, rp3d = np.array([u, v]), np.array([x, y, z])
        #     all_rpr.append(rpr)
        #     all_rp3d.append(rp3d)
        # all_rpr = np.array(all_rpr) # Nx2
        # all_rp3d = np.array(all_rp3d) # Nx3
        # try:
        #     if interp == 'linear': # piecewise linear interpolation
        #         lint = LinearNDInterpolator(all_rpr, all_rp3d, fill_value=np.nan)
        #         mkp3d = lint(mkpr[:,0], mkpr[:,1])
        #         valid = ~np.any(np.isnan(mkp3d), axis=-1)
        #     elif interp == 'cubic': # piecewise cubic interpolation
        #         cint = CloughTocher2DInterpolator(all_rpr, all_rp3d, fill_value=np.nan)
        #         mkp3d = cint(mkpr[:,0], mkpr[:,1])
        #         valid = ~np.any(np.isnan(mkp3d), axis=-1)
        #     else: # nearest neighbor
        #         knn = NearestNDInterpolatorQuery(all_rpr, all_rp3d)
        #         mkp3d = knn(mkpr[:,0], mkpr[:,1], distance_upper_bound=1.0)
        #         valid = ~np.any(np.isnan(mkp3d), axis=-1)
        # except:
        #     import pdb
        #     pdb.set_trace()
            
        # all_mkpq.append(mkpq[valid])
        # all_mkpr.append(mkpr[valid])
        # all_mkp3d.append(mkp3d[valid])
        # all_indices.append(np.full(np.count_nonzero(valid), i))

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    cfg = {
        'model': 'SIMPLE_PINHOLE',
        'width': width,
        'height': height,
        'params': [focal_length, cx, cy]
    }
    ret = pycolmap.absolute_pose_estimation(
        all_mkpq, all_mkp3d, cfg, 48.00)
    ret['cfg'] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches

def main(dataset_dir, retrieval, features, matches, results, topk=None,
         skip_matches=None, interp='linear'):

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    feature_file = h5py.File(features, 'r', libver='latest')
    match_file = h5py.File(matches, 'r', libver='latest')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')
    for q in tqdm(queries):
        db = retrieval_dict[q]
        # import time
        # a = time.time()
        ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
            dataset_dir, q, db, feature_file, match_file, topk, skip_matches, interp)
        # print(time.time()-a)
        

        poses[q] = (ret['qvec'], ret['tvec'])
        logs['loc'][q] = {
            'db': db,
            'PnP_ret': ret,
            'keypoints_query': mkpq,
            'keypoints_db': mkpr,
            '3d_points': mkp3d,
            'indices_db': indices,
            'num_matches': num_matches,
        }

    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in queries:
            qvec, tvec = poses[q]
            quat = quaternion.from_float_array(qvec)
            tvec = -quaternion.rotate_vectors(quat.inverse(), tvec)
            quat_inv = quat.inverse()
            qvec = quat_inv.imag.tolist() + [quat_inv.real]
            
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split("/")[-1]
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')

def eval_performance_kgnet(results, image_dir):
    with open(str(results) + "_logs.pkl", "rb") as f:
        logs = pickle.load(f)
    queries = list(logs["loc"].keys())
    
    pos_err, angle_err = [], []
    for qname in queries:
        loc = logs["loc"][qname]
        qvec, tvec = loc["KGNet_ret"]["qvec"], loc["KGNet_ret"]["tvec"]
        quat = quaternion_from_coeff(qvec)
        # tvec = -quaternion.rotate_vectors(quat.inverse(), tvec)
        # rmat = quaternion.as_rotation_matrix(quat)

        with open(Path(image_dir, qname.split('.')[0] + '.txt'),'r') as f:
            line1 = f.readline()
        x, y, z, q_x, q_y, q_z, q_w = map(float, line1.split())
        quat_gt = quaternion.from_float_array([q_w, q_x, q_y, q_z])
        # rmat_gt = quaternion.as_rotation_matrix(quat)
        tvec_gt = np.array([x, y, z])

        pos_err.append(np.linalg.norm(tvec - tvec_gt))
        angle_err.append(angle_between_quaternions(quat, quat_gt)*180/np.pi)

    pos_err, angle_err = np.array(pos_err), np.array(angle_err)
    print("pos mae: %.3f"% pos_err.mean(), "angle mae: %.3f"% angle_err.mean())
    print("pos rmse: %.3f"% ((pos_err**2).mean())**0.5, "angle rmse: %.3f"% ((angle_err**2).mean())**0.5)
    return pos_err, angle_err

def pose_from_kgnet(ckpt_dir, kgnet, dataset_dir, q, retrieved, interp='linear', device='cuda:0'):
    all_mkpq = [] # matched keypoint coordinates in query images
    all_mkpr = [] # matched keypoint coordinates in retrieved images
    all_mkp3d = [] # matched keypoint 3d coordinate
    all_indices = []
    num_matches = 0
    
    enable_depth = 'depth' in ckpt_dir
    # query image
    color_raw_q = Image.open(str(dataset_dir / q))
    input_transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
    img_q = input_transform(color_raw_q)[:3,...][None].to(device)

    for i, r in enumerate(retrieved):
        # db image
        color_raw_db = Image.open(str(dataset_dir / r))
        img_db = input_transform(color_raw_db)[:3,...][None].to(device)
        # db pose
        # with open(Path(dataset_dir, r.split('_')[0] + '.txt'),'r') as f:
        with open(Path(dataset_dir, r.split('.')[0] + '.txt'),'r') as f:
            lines = f.readlines()
        q_x, q_y, q_z, q_w, p_x, p_y, p_z = map(float, lines[0].split())
        q_w2db = quaternion.from_float_array([q_w, q_x, q_y, q_z])
        t_w2db = np.array([p_x, p_y, p_z])
        T_w2db = torch.eye(4)
        T_w2db[:3,:3] = torch.tensor(quaternion.as_rotation_matrix(q_w2db))
        T_w2db[:3,3] = torch.tensor(t_w2db)
        
        if enable_depth:
            # all_uvxyz = lines[1:]
            # all_rpr = [] # rendered points retrieval
            # all_rp3d = [] # rendered points 3d coordinate
            # for uvxyz in all_uvxyz:
            #     u, v, x, y, z, d = map(float, uvxyz.split(' '))
            #     rpr, rp3d = np.array([u, v]), np.array([x, y, z])
            #     all_rpr.append(rpr)
            #     all_rp3d.append(rp3d)
            # all_rpr = np.array(all_rpr) # Nx2
            # all_rp3d = np.array(all_rp3d) # Nx3
            
            # if interp == 'linear': # piecewise linear interpolation
            #     itpt = LinearNDInterpolator(all_rpr, all_rp3d, fill_value=np.nan)
            #     # mkp3d = lint(mkpr[:,0], mkpr[:,1])
            #     # valid = ~np.any(np.isnan(mkp3d), axis=-1)
            # elif interp == 'cubic': # piecewise cubic interpolation
            #     itpt = CloughTocher2DInterpolator(all_rpr, all_rp3d, fill_value=np.nan)
            #     # mkp3d = cint(mkpr[:,0], mkpr[:,1])
            #     # valid = ~np.any(np.isnan(mkp3d), axis=-1)
            # else: # nearest neighbor
            #     itpt = NearestNDInterpolatorQuery(all_rpr, all_rp3d)
            #     # mkp3d = knn(mkpr[:,0], mkpr[:,1], distance_upper_bound=1.0)
            #     # valid = ~np.any(np.isnan(mkp3d), axis=-1)
            depth_img = cv2.imread(str(dataset_dir / r.split('.')[0]) + '.tiff', cv2.IMREAD_ANYDEPTH)
            depth_img = torch.tensor(depth_img).to(device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            depth_img = depth_img.clamp(0., 20.) / 20.
            itpt = None
            T_w2db = None
        else:
            itpt = None
            T_w2db = None
            depth_img = None

        # kgnet inference
        pred, matches, alphas, mkpq, mkpr, mscore, inlier_masks, mkp3d = kgnet.infer(
                                                                {'rgb': img_db,
                                                                 'imagegoal': img_q,
                                                                 'enable_depth': enable_depth,
                                                                 'interp': itpt,
                                                                 'db_pose': T_w2db,
                                                                 'depth': depth_img,
                                                                })
        # pred is relative pose, let's convert it to absolute pose
        q_pred = quaternion_from_coeff(pred[0, 3:7].detach().cpu().numpy())  # xyzw
        t_pred = pred[0, 0:3].detach().cpu().numpy() # xyz

        z_rot = np.array([[0., 1., 0.],
                        [-1., 0., 0.],
                        [0., 0., 1.]])
        tr_coord = np.array([[1., 0., 0.],
                            [0., 0., -1.],   # -1 for habitat, 1 for opencv
                            [0., 1., 0.]])   # opposite of the right above
        tr_coord = np.matmul(z_rot, tr_coord) #T^u_h
        q_coord = quaternion.from_rotation_matrix(tr_coord)
        
        q_db2q = q_coord * q_pred * q_coord.inverse()
        # q_db2q.imag = quaternion_rotate_vector(q_coord, q_pred.imag)
        t_db2q = quaternion_rotate_vector(q_coord, t_pred)
        q_w2q = q_w2db * q_db2q
        t_w2q = t_w2db + quaternion_rotate_vector(q_w2db, t_db2q)

        num_matches += len(mkpq)
     
        all_mkpq.append(mkpq.detach().cpu().numpy())
        all_mkpr.append(mkpr.detach().cpu().numpy())
        all_indices.append(np.full(mkpq.shape[0], i))
        # if enable_depth:
        #     all_mkp3d.append(mkp3d.detach().cpu().numpy())
            
        break

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_indices = np.concatenate(all_indices, 0)
    # if enable_depth:
    #     all_mkp3d = np.concatenate(all_mkp3d, 0)

    ret = {}
    ret['qvec'] = quaternion_to_list(q_w2q) # xyzw
    ret['tvec'] = t_w2q
    ret['inliers'] = inlier_masks.detach().cpu().numpy()

    # cfg = {
    #     'model': 'SIMPLE_PINHOLE',
    #     'width': width,
    #     'height': height,
    #     'params': [focal_length, cx, cy]
    # }
    # ret = pycolmap.absolute_pose_estimation(
    #     all_mkpq, all_mkp3d, cfg, 48.00)
    # ret['cfg'] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches

def main_kgnet(ckpt_dir, dataset_dir, retrieval, results, interp='linear'):

    assert retrieval.exists(), retrieval

    ckpt_dict = torch.load(ckpt_dir, map_location="cpu")
    agent = PPO(LightGluePosePolicy())
    agent.load_state_dict(ckpt_dict["state_dict"])
    device = torch.device("cuda:0")
    policy = agent.actor_critic
    policy.to(device)
    policy.eval()

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    poses = {}
    logs = {
        # 'features': features,
        # 'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')
    for q in tqdm(queries):
        db = retrieval_dict[q]
        ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_kgnet(
            ckpt_dir, policy, dataset_dir, q, db, interp, device)

        poses[q] = (ret['qvec'], ret['tvec'])
        logs['loc'][q] = {
            'db': db,
            'KGNet_ret': ret,
            'keypoints_query': mkpq,
            'keypoints_db': mkpr,
            '3d_points': mkp3d,
            'indices_db': indices,
            'num_matches': num_matches,
        }

    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in queries:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split("/")[-1]
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--skip_matches', type=int)
    args = parser.parse_args()
    main(**args.__dict__)
