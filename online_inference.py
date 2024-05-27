import argparse
from pathlib import Path
from pprint import pformat

from types import SimpleNamespace
from typing import Dict, List, Optional, Union

from hloc import (
    extract_features,
    match_features,
    localize_guro,
    pairs_from_retrieval,
)

from hloc import extractors, matchers, logger
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names, read_image
from hloc.utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

from tqdm import tqdm
import numpy as np
import PIL.Image
import cv2
import torch
import quaternion
import h5py

def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class VisLoc():
    def __init__(self, dataset_dir, global_feature_dir, local_feature_dir, as_half=True, global_conf=None, local_conf=None, matcher_conf=None):
        self.global_conf = extract_features.confs["netvlad320"] if global_conf is None else extract_features.confs[global_conf]
        self.local_conf = extract_features.confs["superpoint_inloc"] if local_conf is None else extract_features.confs[local_conf]
        self.matcher_conf = match_features.confs["superpoint+lightglue"] if matcher_conf is None else match_features.confs[matcher_conf]
        self.dataset_dir = dataset_dir
        if global_feature_dir is None:
            self.global_feature_dir = "outputs/" + str(dataset_dir).split('/')[1] + "/" + self.global_conf["output"] + ".h5"
        else:
            self.global_feature_dir = global_feature_dir
        if local_feature_dir is None:
            self.local_feature_dir = "outputs/" + str(dataset_dir).split('/')[1] + "/" + self.local_conf["output"] + "-depth.h5"
        else:
            self.local_feature_dir = local_feature_dir
        self.as_half = as_half

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, self.global_conf["model"]["name"])
        self.model_global = Model(self.global_conf["model"]).eval().to(self.device)
        Model = dynamic_load(extractors, self.local_conf["model"]["name"])
        self.model_local = Model(self.local_conf["model"]).eval().to(self.device)
        Model = dynamic_load(matchers, self.matcher_conf["model"]["name"])
        self.model_matcher = Model(self.matcher_conf["model"]).eval().to(self.device)

        self.default_conf = {
            "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
            "grayscale": False,
            "resize_max": None,
            "resize_force": False,
            "interpolation": "cv2_area",  # pil_linear is more accurate but slower
        }
        
        db_descriptors = [self.global_feature_dir]
        name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
        db_names_h5 = list(name2db.keys())

        self.db_names = pairs_from_retrieval.parse_names('db', None, db_names_h5)
        if len(self.db_names) == 0:
            raise ValueError("Could not find any database image.")

        self.db_glob = pairs_from_retrieval.get_descriptors(self.db_names, db_descriptors, name2db)
        self.db_local = h5py.File(self.local_feature_dir, "r") 

        print("Visual localization inference ready.")

    def preprocess(self, image, conf):        
        conf = SimpleNamespace(**{**self.default_conf, **conf})
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if conf.resize_max and (
            conf.resize_force or max(size) > conf.resize_max
        ):
            scale = conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, conf.interpolation)

        if conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0
        return image[None], np.array(size)
    
    @torch.no_grad()
    def inference(self,
            image_rgb: np.array,
            image_dir: Path,
            top_k: int = 5,
    ):
        if image_rgb is None:
            image_rgb = read_image(image_dir, False)
            image_gray = read_image(image_dir, True)
        else:
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        image = {'rgb': image_rgb, 'gray': image_gray}
        
        # preprocess input image
        input_global, original_size_global = self.preprocess(image['rgb'], self.global_conf['preprocessing'])
        input_local, original_size_local = self.preprocess(image['rgb'], self.local_conf['preprocessing']) if self.local_conf['preprocessing']['grayscale'] is False else self.preprocess(image['gray'], self.local_conf['preprocessing'])

        # extract feature
        pred_glob = self.model_global({"image": torch.from_numpy(input_global).to(self.device, non_blocking=True)})
        pred_glob = {k: v[0] for k, v in pred_glob.items()}
        
        pred_local = self.model_local({"image": torch.from_numpy(input_local).to(self.device, non_blocking=True)})
        pred_local = {k: v[0] for k, v in pred_local.items()}

        pred_local["image_size"] = original_size_local
        if "keypoints" in pred_local:
            size = input_local.shape[-2:][::-1]
            scales = torch.from_numpy(original_size_local / size).type(torch.float32).to(self.device)
            pred_local["keypoints"] = (pred_local["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred_local:
                pred_local["scales"] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(self.model_local, "detection_noise", 1) * scales.mean()

        if self.as_half:
            for k in pred_glob:
                dt = pred_glob[k].dtype
                if (dt == torch.float32) and (dt != torch.float16):
                    pred_glob[k] = pred_glob[k].type(torch.float16)
            for k in pred_local:
                dt = pred_local[k].dtype
                if (dt == torch.float32) and (dt != torch.float16):
                    pred_local[k] = pred_local[k].type(torch.float16)
        
        # retrieve image pairs
        retrieved = pairs_from_retrieval.main_single(pred_glob["global_descriptor"], self.db_glob, num_matched=top_k, db_names=self.db_names)
        
        # match feature
        dataset = match_features.FeaturePairDataset(retrieved, pred_local, self.db_local) #self.local_feature_dir)
        loader = torch.utils.data.DataLoader(
            dataset, num_workers=0, batch_size=top_k, shuffle=False, pin_memory=True
        )
        pred_match = []
        # for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        for idx, data in enumerate(loader):
            data = {
                k: v if k.startswith("image") else v.to(self.device, non_blocking=True)
                for k, v in data.items()
            }
            pred = self.model_matcher(data)
            # pred_match.append(pred)
            pred_match = pred # match with the full batch size
            # import pdb
            # pdb.set_trace()

        # localize
        # localize_guro.main(images, loc_pairs, feature_path, match_path, results_linear_5_q1, topk=20, interp='linear')
        ret, mkpq, mkpr, mkp3d, indices, num_matches = localize_guro.pose_from_cluster_single(
                                                        self.dataset_dir, retrieved, pred_local, 
                                                        self.db_local, pred_match, top_k
                                                        )
        qvec, tvec = ret['qvec'], ret['tvec']
        quat = quaternion.from_float_array(qvec)
        tvec = -quaternion.rotate_vectors(quat.inverse(), tvec)
        quat_inv = quat.inverse()
        qvec = quat_inv.imag.tolist() + [quat_inv.real]
        pose = {'tvec': tvec.tolist(), 'qvec': qvec}

        return pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument('--global_feature_dir', type=Path, default=None)
    parser.add_argument('--local_feature_dir', type=Path, default=None)
    parser.add_argument(
        "--global_conf", type=str, default="netvlad320", choices=list(extract_features.confs.keys())
    )
    parser.add_argument(
        "--local_conf", type=str, default="superpoint_inloc800", choices=list(extract_features.confs.keys())
    )
    parser.add_argument(
        "--matcher_conf", type=str, default="superpoint+lightglue", choices=list(match_features.confs.keys()),
    )
    parser.add_argument("--as_half", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    # parser.add_argument("--feature_path", type=Path)
    args = parser.parse_args()

    visloc = VisLoc(
        args.dataset_dir, args.global_feature_dir, args.local_feature_dir, args.as_half, 
        args.global_conf, args.local_conf, args.matcher_conf
        )
    pose = visloc.inference(None, args.image_dir, args.top_k)
    print(pose)
    pose = visloc.inference(None, args.image_dir, args.top_k)
    print(pose)
    import time
    a= time.time()
    for i in tqdm(range(100)):
        pose = visloc.inference(None, "/".join(str(args.image_dir).split('/')[:-1]) + '/' + str(i) + '.png', args.top_k)
    print((time.time()-a)/100)
