import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary

# import faiss
from PIL import Image
import torchvision.transforms as transforms
from .habitat.kgnet import LightGluePosePolicy
from tqdm import tqdm

def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(
    scores: torch.Tensor,
    invalid: np.array,
    num_select: int,
    min_score: Optional[float] = None,
):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    if isinstance(invalid, np.ndarray):
        invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs

def main_single(
    query_desc,
    db_desc,
    num_matched,
    db_names,
):
    # # logger.info("Extracting image pairs from a retrieval database.")

    # # We handle multiple reference feature files.
    # # We only assume that names are unique among them and map names to files.
    # if db_descriptors is None:
    #     db_descriptors = descriptors
    # if isinstance(db_descriptors, (Path, str)):
    #     db_descriptors = [db_descriptors]
    # name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
    # db_names_h5 = list(name2db.keys())

    # db_names = parse_names(db_prefix, db_list, db_names_h5)
    # if len(db_names) == 0:
    #     raise ValueError("Could not find any database image.")

    # db_desc = get_descriptors(db_names, db_descriptors, name2db)
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    sim = torch.einsum("id,jd->ij", query_desc.to(device).unsqueeze(0), db_desc.to(device).type(query_desc.dtype))

    # Avoid self-matching
    # self = np.array(query_names)[:, None] == np.array(db_names)[None]
    self = torch.zeros_like(sim, dtype=bool)
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    pairs = [db_names[j] for i, j in pairs]
    return pairs

def main(
    descriptors,
    output,
    num_matched,
    query_prefix=None,
    query_list=None,
    db_prefix=None,
    db_list=None,
    db_model=None,
    db_descriptors=None,
):
    logger.info("Extracting image pairs from a retrieval database.")

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = read_images_binary(db_model / "images.bin")
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError("Could not find any database image.")
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))

    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))

# def main_tracking(dataset_dir, descriptors, output, num_matched, n_neighbor=156, min_matches=200,
#          query_prefix=None, query_list=None,
#          db_prefix=None, db_list=None, db_model=None, db_descriptors=None, num_keypoints=4096):
#     logger.info('Extracting image pairs from a retrieval database.')

#     # We handle multiple reference feature files.
#     # We only assume that names are unique among them and map names to files.
#     if db_descriptors is None:
#         db_descriptors = descriptors
#     if isinstance(db_descriptors, (Path, str)):
#         db_descriptors = [db_descriptors]
#     name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
#     db_names_h5 = list(name2db.keys())
#     query_names_h5 = list_h5_names(descriptors)

#     if db_model:
#         images = read_images_binary(db_model / "images.bin")
#         db_names = [i.name for i in images.values()]
#     else:
#         db_names = parse_names(db_prefix, db_list, db_names_h5)
#     if len(db_names) == 0:
#         raise ValueError("Could not find any database image.")
#     query_names = parse_names(query_prefix, query_list, query_names_h5)

#     db_names = sorted(db_names, key=lambda x: int(''.join(filter(str.isdigit, x))))
#     query_names = sorted(query_names, key=lambda x: int(''.join(filter(str.isdigit, x))))

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     db_desc = get_descriptors(db_names, db_descriptors, name2db)
#     query_desc = get_descriptors(query_names, descriptors)

#     db_pos = []
#     for i in range(len(db_names)):
#         with open( str(dataset_dir / (db_names[i].split('.')[0] + '.txt')), 'r') as f:
#             line = f.readline()
#             _,_,_,_,x,y,z = map(float, line.split())
#             db_pos.append([x,y,z])
#     db_pos = np.asarray(db_pos)
    
#     all_pairs = []
#     knn = faiss.IndexFlatL2(3)
#     knn.add(db_pos.astype('float32'))

#     lightglue = LightGluePosePolicy(num_keypoints=num_keypoints).eval()
#     lost = False
#     for i in tqdm(range(len(query_names))):

#         color_raw_q = Image.open(str(dataset_dir / query_names[i]))
#         input_transform = transforms.Compose([
#                             transforms.ToTensor(),
#                         ])
#         img_q = input_transform(color_raw_q)[:3,...][None].to(device)

#         # import time
#         # a= time.time()
        
#         if i == 0 or lost:
#             sim = torch.einsum('id,jd->ij', query_desc[i:i+1].to(device), db_desc.to(device))
#             # Avoid self-matching
#             self = np.array(query_names)[i:i+1][:, None] == np.array(db_names)[None]
#             pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
#             for j, k in pairs:
#                 all_pairs.append((query_names[i:i+1][j], db_names[k]))
#             # _, k_nbr = knn.search(db_pos[pairs[0][1]].astype('float32')[None], n_neighbor) #156
#             # k_nbr = k_nbr.squeeze() 
#             img_db = []
#             for j, k in pairs:
#                 color_raw_db = Image.open(str(dataset_dir / np.array(db_names)[k]))
#                 img_db.append( input_transform(color_raw_db)[:3,...][None].to(device) )
#             img_db = torch.concatenate(img_db, dim=0)

#             if num_matched == 20:
#                 matches = {'matches': []}
#                 for l in range(num_matched):
#                     matches0 = lightglue.net({'rgb': img_db[l:l+1],
#                                         'imagegoal': torch.tile(img_q, (1,1,1,1)),
#                                         'enable_detph': False,
#                                         'interp': None,
#                                         'db_pose': None,
#                                         'depth': None}, lightglue_mode=True)
#                     matches = {'matches': matches['matches'] + matches0['matches']}
#                 num_matches = np.array([len(match) for match in matches['matches']])

#                 # matches0 = lightglue.net({'rgb': img_db[:10],
#                 #                         'imagegoal': torch.tile(img_q, (10,1,1,1)),
#                 #                         'enable_detph': False,
#                 #                         'interp': None,
#                 #                         'db_pose': None,
#                 #                         'depth': None}, lightglue_mode=True)
#                 # matches1 = lightglue.net({'rgb': img_db[10:],
#                 #                         'imagegoal': torch.tile(img_q, (10,1,1,1)),
#                 #                         'enable_detph': False,
#                 #                         'interp': None,
#                 #                         'db_pose': None,
#                 #                         'depth': None}, lightglue_mode=True)
#                 # matches = {'matches': matches0['matches'] + matches1['matches']}
#                 # num_matches = np.array([len(match) for match in matches['matches']])
#             else:
#                 matches = lightglue.net({'rgb': img_db,
#                                         'imagegoal': torch.tile(img_q, (num_matched,1,1,1)),
#                                         'enable_detph': False,
#                                         'interp': None,
#                                         'db_pose': None,
#                                         'depth': None}, lightglue_mode=True)
#                 num_matches = np.array([len(match) for match in matches['matches']])
#             if np.max(num_matches) < min_matches:
#                 lost = True
#                 continue
#             else:
#                 lost = False
#                 _, k_nbr = knn.search(db_pos[pairs[np.argmax(num_matches)][1]].astype('float32')[None], n_neighbor) #156
#                 k_nbr = k_nbr.squeeze()  
#         else:
#             sim = torch.einsum('id,jd->ij', query_desc[i:i+1].to(device), db_desc[k_nbr].to(device))
#             # Avoid self-matching
#             self = np.array(query_names)[i:i+1][:, None] == np.array(db_names)[k_nbr][None]
#             pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
#             for j, k in pairs:
#                 all_pairs.append((query_names[i:i+1][j], np.array(db_names)[k_nbr][k]))
#             # TODO: let this be the image with the largest number of keypoint matching, not the NetVLAD-nearest image.
#             # _, k_nbr = knn.search(db_pos[k_nbr][pairs[0][1]].astype('float32')[None], n_neighbor) #156
#             # k_nbr = k_nbr.squeeze()  
#             img_db = []
#             for j, k in pairs:
#                 color_raw_db = Image.open(str(dataset_dir / np.array(db_names)[k_nbr][k]))
#                 img_db.append( input_transform(color_raw_db)[:3,...][None].to(device) )
#             img_db = torch.concatenate(img_db, dim=0)

#             if num_matched == 20:
#                 matches = {'matches': []}
#                 for l in range(num_matched):
#                     matches0 = lightglue.net({'rgb': img_db[l:l+1],
#                                         'imagegoal': torch.tile(img_q, (1,1,1,1)),
#                                         'enable_detph': False,
#                                         'interp': None,
#                                         'db_pose': None,
#                                         'depth': None}, lightglue_mode=True)
#                     matches = {'matches': matches['matches'] + matches0['matches']}
#                 num_matches = np.array([len(match) for match in matches['matches']])

#                 # matches0 = lightglue.net({'rgb': img_db[:10],
#                 #                         'imagegoal': torch.tile(img_q, (10,1,1,1)),
#                 #                         'enable_detph': False,
#                 #                         'interp': None,
#                 #                         'db_pose': None,
#                 #                         'depth': None}, lightglue_mode=True)
#                 # matches1 = lightglue.net({'rgb': img_db[10:],
#                 #                         'imagegoal': torch.tile(img_q, (10,1,1,1)),
#                 #                         'enable_detph': False,
#                 #                         'interp': None,
#                 #                         'db_pose': None,
#                 #                         'depth': None}, lightglue_mode=True)
#                 # matches = {'matches': matches0['matches'] + matches1['matches']}
#                 # num_matches = np.array([len(match) for match in matches['matches']])
#             else:
#                 matches = lightglue.net({'rgb': img_db,
#                                         'imagegoal': torch.tile(img_q, (num_matched,1,1,1)),
#                                         'enable_detph': False,
#                                         'interp': None,
#                                         'db_pose': None,
#                                         'depth': None}, lightglue_mode=True)
#                 num_matches = np.array([len(match) for match in matches['matches']])
#             if np.max(num_matches) < min_matches:
#                 lost = True
#                 continue
#             else:
#                 lost = False
#                 _, k_nbr = knn.search(db_pos[k_nbr][pairs[np.argmax(num_matches)][1]].astype('float32')[None], n_neighbor) #156
#                 k_nbr = k_nbr.squeeze()  
                  
            
#             # nbr_all = np.array([], dtype=int)
#             # for l in range(num_matched):
#             #     _, nbr = knn.search(db_pos[k_nbr][pairs[l][1]].astype('float32')[None], num_matched) #156
#             #     nbr_all = np.unique(np.concatenate((nbr_all, nbr.squeeze())))
#             # k_nbr = nbr_all      
#         # print(time.time()-a)

#     logger.info(f'Found {len(all_pairs)} pairs.')
#     with open(output, 'w') as f:
#         f.write('\n'.join(' '.join([i, j]) for i, j in all_pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_matched", type=int, required=True)
    parser.add_argument("--query_prefix", type=str, nargs="+")
    parser.add_argument("--query_list", type=Path)
    parser.add_argument("--db_prefix", type=str, nargs="+")
    parser.add_argument("--db_list", type=Path)
    parser.add_argument("--db_model", type=Path)
    parser.add_argument("--db_descriptors", type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
