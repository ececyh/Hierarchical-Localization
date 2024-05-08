import pickle
import random

import numpy as np
import pycolmap
from matplotlib import cm

from .utils.io import read_image
from .utils.viz import add_text, cm_RdGn, plot_images, plot_keypoints, plot_matches

from .utils.viz_3d import plot_points, plot_camera, init_figure
import quaternion
import open3d as o3d
from pathlib import Path

from .habitat.geometry_utils import *

def visualize_sfm_2d(
    reconstruction, image_dir, color_by="visibility", selected=[], n=1, seed=0, dpi=75
):
    assert image_dir.exists()
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)

    if not selected:
        image_ids = reconstruction.reg_image_ids()
        selected = random.Random(seed).sample(image_ids, min(n, len(image_ids)))

    for i in selected:
        image = reconstruction.images[i]
        keypoints = np.array([p.xy for p in image.points2D])
        visible = np.array([p.has_point3D() for p in image.points2D])

        if color_by == "visibility":
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f"visible: {np.count_nonzero(visible)}/{len(visible)}"
        elif color_by == "track_length":
            tl = np.array(
                [
                    reconstruction.points3D[p.point3D_id].track.length()
                    if p.has_point3D()
                    else 1
                    for p in image.points2D
                ]
            )
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f"max/median track length: {max_}/{med_}"
        elif color_by == "depth":
            p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
            z = np.array(
                [
                    (image.cam_from_world * reconstruction.points3D[j].xyz)[-1]
                    for j in p3ids
                ]
            )
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f"visible: {np.count_nonzero(visible)}/{len(visible)}"
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f"Coloring not implemented: {color_by}.")

        name = image.name
        plot_images([read_image(image_dir / name)], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        add_text(0, text)
        add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")


def visualize_loc(
    results,
    image_dir,
    reconstruction=None,
    db_image_dir=None,
    selected=[],
    n=1,
    seed=0,
    prefix=None,
    **kwargs,
):
    assert image_dir.exists()

    with open(str(results) + "_logs.pkl", "rb") as f:
        logs = pickle.load(f)

    if not selected:
        queries = list(logs["loc"].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        selected = random.Random(seed).sample(queries, min(n, len(queries)))

    if reconstruction is not None:
        if not isinstance(reconstruction, pycolmap.Reconstruction):
            reconstruction = pycolmap.Reconstruction(reconstruction)

    for qname in selected:
        loc = logs["loc"][qname]
        visualize_loc_from_log(
            image_dir, qname, loc, reconstruction, db_image_dir, **kwargs
        )


def visualize_loc_from_log(
    image_dir,
    query_name,
    loc,
    reconstruction=None,
    db_image_dir=None,
    top_k_db=2,
    dpi=75,
):
    q_image = read_image(image_dir / query_name)
    if loc.get("covisibility_clustering", False):
        # select the first, largest cluster if the localization failed
        loc = loc["log_clusters"][loc["best_cluster"] or 0]

    inliers = np.array(loc["PnP_ret"]["inliers"])
    mkp_q = loc["keypoints_query"]
    n = len(loc["db"])
    if reconstruction is not None:
        # for each pair of query keypoint and its matched 3D point,
        # we need to find its corresponding keypoint in each database image
        # that observes it. We also count the number of inliers in each.
        kp_idxs, kp_to_3D_to_db = loc["keypoint_index_to_db"]
        counts = np.zeros(n)
        dbs_kp_q_db = [[] for _ in range(n)]
        inliers_dbs = [[] for _ in range(n)]
        for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers, kp_to_3D_to_db)):
            track = reconstruction.points3D[p3D_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            for db_idx in db_idxs:
                counts[db_idx] += inl
                kp_db = track[loc["db"][db_idx]]
                dbs_kp_q_db[db_idx].append((i, kp_db))
                inliers_dbs[db_idx].append(inl)
    else:
        # for inloc the database keypoints are already in the logs
        assert "keypoints_db" in loc
        assert "indices_db" in loc
        counts = np.array([np.sum(loc["indices_db"][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        if reconstruction is not None:
            db = reconstruction.images[loc["db"][db_idx]]
            db_name = db.name
            db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
            kp_q = mkp_q[db_kp_q_db[:, 0]]
            kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
            inliers_db = inliers_dbs[db_idx]
        else:
            db_name = loc["db"][db_idx]
            kp_q = mkp_q[loc["indices_db"] == db_idx]
            kp_db = loc["keypoints_db"][loc["indices_db"] == db_idx]
            inliers_db = inliers[loc["indices_db"] == db_idx]

        db_image = read_image((db_image_dir or image_dir) / db_name)
        color = cm_RdGn(inliers_db).tolist()
        text = f"inliers: {sum(inliers_db)}/{len(inliers_db)}"

        plot_images([q_image, db_image], dpi=dpi)
        plot_matches(kp_q, kp_db, color, a=0.1)
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")
        add_text(0, query_name, **opts)
        add_text(1, db_name, **opts)

def visualize_loc_3d(
    results,
    pcd_dir,
    image_dir,
    selected=[],
    n=1,
    seed=0,
    prefix=None,
    K=None,
    ps=1,
    size=1,
    width=1,
    interval=10,
    remove_ceiling=True,
    ceiling_height=1.9,
    downsample=10,
    **kwargs,
):
    '''
    - n: # of query image to visualize
    - top_k_db: k of the top-k retrieved images to visualize its camera pose and image matching 
    - seed: random seed to sample queries. If seed=-1, sample from the query trajectory sequentially with interval.
    - K: camera matrix
    - ps: point size when visualizing point clouds
    - size: size of the camera frustum.
    - width: width of the lines in the camera frustum
    - remove_ceiling: boolean to decide whether the ceiling is visualized or not.
    '''
    pcd = o3d.io.read_point_cloud(str(pcd_dir))
    num_points = np.asarray(pcd.points).shape[0]
    ind = np.arange(num_points)
    np.random.shuffle(ind)
    ind = ind[:num_points//downsample]
    points = np.asarray(pcd.points)[ind,:]
    colors = np.asarray(pcd.colors)[ind,:]
    if remove_ceiling:
        colors = colors[points[:,2]<ceiling_height]
        points = points[points[:,2]<ceiling_height]
    fig = init_figure()
    plot_points(fig, points, colors, ps=ps, name=str(pcd_dir))

    with open(str(results) + "_logs.pkl", "rb") as f:
        logs = pickle.load(f)
        
    if not selected:
        queries = list(logs["loc"].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        if seed == -1:
            q_num = np.array([int(q.split('.')[0].split('/')[1]) for q in queries])
            selected = np.asarray(queries)[q_num.argsort()][0:min(interval*n, len(queries)):interval].tolist()
        else:
            selected = random.Random(seed).sample(queries, min(n, len(queries)))
        
    for qname in selected:
        loc = logs["loc"][qname]
        visualize_loc_3d_from_log(
            image_dir, qname, loc, fig, K, size, width, **kwargs
        )
    fig.show()
        
def visualize_loc_3d_from_log(
    image_dir,
    query_name,
    loc,
    fig,
    K = None,
    size = 1,
    width = 1,
    top_k_db=2,
    dpi=75,
):
    q_image = read_image(image_dir / query_name)
    if loc.get("covisibility_clustering", False):
        # select the first, largest cluster if the localization failed
        loc = loc["log_clusters"][loc["best_cluster"] or 0]

    inliers = np.array(loc["PnP_ret"]["inliers"])
    mkp_q = loc["keypoints_query"]
    n = len(loc["db"])

    # for inloc the database keypoints are already in the logs
    assert "keypoints_db" in loc
    assert "indices_db" in loc
    counts = np.array([np.sum(loc["indices_db"][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        db_name = loc["db"][db_idx]
        kp_q = mkp_q[loc["indices_db"] == db_idx]
        kp_db = loc["keypoints_db"][loc["indices_db"] == db_idx]
        inliers_db = inliers[loc["indices_db"] == db_idx]

        db_image = read_image((image_dir) / db_name)
        color = cm_RdGn(inliers_db).tolist()
        text = f"inliers: {sum(inliers_db)}/{len(inliers_db)}"

        plot_images([q_image, db_image], dpi=dpi)
        plot_matches(kp_q, kp_db, color, a=0.1)
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")
        add_text(0, query_name, **opts)
        add_text(1, db_name, **opts)
    
        # print(loc['3d_points'].shape, inliers_db.shape, inliers.shape )
        plot_points(fig, loc['3d_points'][loc["indices_db"] == db_idx][inliers_db], color='rgba(0,255,0,0.5)', ps=2, name='inlier')
        plot_points(fig, loc['3d_points'][loc["indices_db"] == db_idx][~inliers_db], color='rgba(255,0,0,0.5)', ps=2, name='outlier')
        
        with open(Path(image_dir, db_name.split('.')[0] + '.txt'),'r') as f:
            line1 = f.readline()
        q_x, q_y, q_z, q_w, x, y, z = map(float, line1.split(' '))
        quat = quaternion.from_float_array([q_w, q_x, q_y, q_z])
        rmat = quaternion.as_rotation_matrix(quat)
        tvec = np.array([x, y, z])
        plot_camera(fig, rmat, tvec, K, name=db_name, text=db_name, color='rgba(255,0,0,1)', size=size, width=width)
        
    qvec, tvec = loc["PnP_ret"]["qvec"], loc["PnP_ret"]["tvec"]
    quat = quaternion.from_float_array(qvec)
    tvec = -quaternion.rotate_vectors(quat.inverse(), tvec)
    rmat = quaternion.as_rotation_matrix(quat.inverse())
    plot_camera(fig, rmat, tvec, K, name=query_name, text=query_name, color='rgba(0,255,0,1)', size=size, width=width, fill=True)
    
    with open(Path(image_dir, query_name.split('.')[0] + '.txt'),'r') as f:
        line1 = f.readline()
    x, y, z, q_x, q_y, q_z, q_w = map(float, line1.split())
    quat = quaternion.from_float_array([q_w, q_x, q_y, q_z])
    rmat = quaternion.as_rotation_matrix(quat)
    tvec = np.array([x, y, z])
    plot_camera(fig, rmat, tvec, K, name='gt '+query_name, text='gt '+query_name, color='rgba(0,0,255,1)', size=size, width=width, fill=True)    


def visualize_loc_3d_kgnet(
    results,
    pcd_dir,
    image_dir,
    selected=[],
    n=1,
    seed=0,
    prefix=None,
    K=None,
    ps=1,
    size=1,
    width=1,
    interval=10,
    remove_ceiling=True,
    ceiling_height=1.9,
    downsample=10,
    **kwargs,
):
    '''
    - n: # of query image to visualize
    - top_k_db: k of the top-k retrieved images to visualize its camera pose and image matching 
    - seed: random seed to sample queries. If seed=-1, sample from the query trajectory sequentially with interval.
    - K: camera matrix
    - ps: point size when visualizing point clouds
    - size: size of the camera frustum.
    - width: width of the lines in the camera frustum
    - remove_ceiling: boolean to decide whether the ceiling is visualized or not.
    '''
    pcd = o3d.io.read_point_cloud(str(pcd_dir))
    num_points = np.asarray(pcd.points).shape[0]
    ind = np.arange(num_points)
    np.random.shuffle(ind)
    ind = ind[:num_points//downsample]
    points = np.asarray(pcd.points)[ind,:]
    colors = np.asarray(pcd.colors)[ind,:]
    if remove_ceiling:
        colors = colors[points[:,2]<ceiling_height]
        points = points[points[:,2]<ceiling_height]
    fig = init_figure()
    plot_points(fig, points, colors, ps=ps, name=str(pcd_dir))

    with open(str(results) + "_logs.pkl", "rb") as f:
        logs = pickle.load(f)
        
    if not selected:
        queries = list(logs["loc"].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        if seed == -1:
            q_num = np.array([int(q.split('.')[0].split('/')[1]) for q in queries])
            selected = np.asarray(queries)[q_num.argsort()][0:min(interval*n, len(queries)):interval].tolist()
        else:
            selected = random.Random(seed).sample(queries, min(n, len(queries)))
        
    for qname in selected:
        loc = logs["loc"][qname]
        visualize_loc_3d_from_log_kgnet(
            image_dir, qname, loc, fig, K, size, width, **kwargs
        )
    fig.show()
        
def visualize_loc_3d_from_log_kgnet(
    image_dir,
    query_name,
    loc,
    fig,
    K = None,
    size = 1,
    width = 1,
    top_k_db=2,
    dpi=75,
):
    q_image = read_image(image_dir / query_name)
    if loc.get("covisibility_clustering", False):
        # select the first, largest cluster if the localization failed
        loc = loc["log_clusters"][loc["best_cluster"] or 0]

    inliers = np.array(loc["KGNet_ret"]["inliers"])
    mkp_q = loc["keypoints_query"]
    n = len(loc["db"])

    # for inloc the database keypoints are already in the logs
    assert "keypoints_db" in loc
    assert "indices_db" in loc
    counts = np.array([np.sum(loc["indices_db"][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        db_name = loc["db"][db_idx]
        kp_q = mkp_q[loc["indices_db"] == db_idx]
        kp_db = loc["keypoints_db"][loc["indices_db"] == db_idx]
        inliers_db = inliers[loc["indices_db"] == db_idx]

        db_image = read_image((image_dir) / db_name)
        color = cm_RdGn(inliers_db).tolist()
        text = f"inliers: {sum(inliers_db)}/{len(inliers_db)}"

        plot_images([q_image, db_image], dpi=dpi)
        plot_matches(kp_q, kp_db, color, a=0.1)
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")
        add_text(0, query_name, **opts)
        add_text(1, db_name, **opts)
    
        # print(loc['3d_points'].shape, inliers_db.shape, inliers.shape )
        # plot_points(fig, loc['3d_points'][loc["indices_db"] == db_idx][inliers_db], color='rgba(0,255,0,0.5)', ps=2, name='inlier')
        # plot_points(fig, loc['3d_points'][loc["indices_db"] == db_idx][~inliers_db], color='rgba(255,0,0,0.5)', ps=2, name='outlier')
        
        with open(Path(image_dir, db_name.split('_')[0] + '.txt'),'r') as f:
            line1 = f.readline()
        q_x, q_y, q_z, q_w, x, y, z = map(float, line1.split(' '))
        quat = quaternion.from_float_array([q_w, q_x, q_y, q_z])
        rmat = quaternion.as_rotation_matrix(quat)
        tvec = np.array([x, y, z])
        plot_camera(fig, rmat, tvec, K, name=db_name, text=db_name, color='rgba(255,0,0,1)', size=size, width=width)
        
    qvec, tvec = loc["KGNet_ret"]["qvec"], loc["KGNet_ret"]["tvec"]
    quat = quaternion_from_coeff(qvec)
    # tvec = -quaternion.rotate_vectors(quat.inverse(), tvec)
    rmat = quaternion.as_rotation_matrix(quat)
    plot_camera(fig, rmat, tvec, K, name=query_name, text=query_name, color='rgba(0,255,0,1)', size=size, width=width, fill=True)
    
    with open(Path(image_dir, query_name.split('.')[0] + '.txt'),'r') as f:
        line1 = f.readline()
    x, y, z, q_x, q_y, q_z, q_w = map(float, line1.split())
    quat = quaternion.from_float_array([q_w, q_x, q_y, q_z])
    rmat = quaternion.as_rotation_matrix(quat)
    tvec = np.array([x, y, z])
    plot_camera(fig, rmat, tvec, K, name='gt '+query_name, text='gt '+query_name, color='rgba(0,0,255,1)', size=size, width=width, fill=True)    
