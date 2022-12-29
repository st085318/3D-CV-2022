#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import copy
import cv2
import itertools
import scipy

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,

    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)


def triangulate_points_on_two_frames(frame_1: int, frame_2: int, corner_storage: CornerStorage, view_mats: List[np.ndarray],
                                     intrinsic_mat: np.ndarray, params: TriangulationParameters) -> Tuple[np.ndarray, np.ndarray]:

    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
    points3d, ids, _ = triangulate_correspondences(correspondences, view_mats[frame_1], view_mats[frame_2],
                                                   intrinsic_mat, params)

    return ids, points3d


def calc_full_mat(frame: int, view_mats: List[np.ndarray], intrinsic_mat: np.ndarray):
    K = np.zeros((4, 4))
    K[: 3, : 3] = intrinsic_mat
    K[2][2] = 0
    K[2][3] = 1
    K[3][2] = 1
    cam_mat = np.vstack((view_mats[frame].astype(float), np.array([0, 0, 0, 1], dtype=float)))
    return K @ cam_mat


def triangulate_points_on_multiple_frames(frame_ids: np.ndarray, corner_storage: CornerStorage, view_mats: List[np.ndarray],
                                          intrinsic_mat: np.ndarray, params: TriangulationParameters,
                                          frames_cnt_threshold: int = 7) -> Tuple[np.ndarray, np.ndarray]:

    frames_per_point = {}
    for i, j in itertools.combinations(frame_ids, 2):
        p_ids, _ = triangulate_points_on_two_frames(i, j, corner_storage, view_mats, intrinsic_mat, params)
        for id in p_ids:
            if not(id in frames_per_point):
                frames_per_point[id] = set()
            frames_per_point[id].add(i)
            frames_per_point[id].add(j)

    full_mats = {}
    for frame in frame_ids:
        full_mats[frame] = calc_full_mat(frame, view_mats, intrinsic_mat)

    ids, points3d = [], []
    for point in frames_per_point:
        if len(frames_per_point[point]) < frames_cnt_threshold:
            continue


        sq_sys = []
        for frame in frames_per_point[point]:
            ind_of_point_in_cs = corner_storage[frame].ids.tolist().index(point)
            fm = full_mats.get(frame)
            sq_sys.append(fm[3] * corner_storage[frame].points[ind_of_point_in_cs][0] - fm[0])
            sq_sys.append(fm[3] * corner_storage[frame].points[ind_of_point_in_cs][1] - fm[1])

        sq_sys = np.array(sq_sys)
        coordinates = scipy.linalg.lstsq(sq_sys[:, : 3], -sq_sys[:, 3], lapack_driver="gelsy", check_finite = False)[0]

        ids.append(point)
        points3d.append(coordinates)

    return np.array(ids).astype(np.int64), np.array(points3d)


# reminder: think about weight
def get_best_points_for_frames(frame: int, frames: List[int], corner_storage: CornerStorage,
                              view_mats: List[np.ndarray], intrinsic_mat: np.ndarray, params: TriangulationParameters,
                               recalc=False, weight=5) \
        -> Tuple[np.ndarray, np.ndarray]:
    if recalc:
        print(f"        Retreangulation")
        best_ids, best_points = triangulate_points_on_multiple_frames(np.append(frame, frames), corner_storage, view_mats,
                                                                      intrinsic_mat, params)
        return best_ids, best_points

    best_ids, best_points = triangulate_points_on_two_frames(frame, frames[0], corner_storage, view_mats, intrinsic_mat, params)
    bi, bp = triangulate_points_on_two_frames(frame, frames[1], corner_storage, view_mats, intrinsic_mat, params)
    if len(bi) > len(best_ids):
        best_ids = bi
        best_points = bp
    for f in frames[2:]:
        ids, points = triangulate_points_on_two_frames(frame, f, corner_storage, view_mats, intrinsic_mat, params)
        if len(ids) > weight * len(best_ids):
            best_ids, best_points = ids, points
    return best_ids, best_points


def calc_init_frames(corner_storage: CornerStorage, intrinsic_mat, params):
    inliers_ratio = []
    cos_ratio = []
    frames_info = []
    step = 10 if len(corner_storage) > 50 else 2

    for frame1 in range(0, len(corner_storage), step):
        for frame2 in range(frame1 + step, len(corner_storage), step):

            corrs = build_correspondences(corner_storage[frame1], corner_storage[frame2])
            if len(corrs) < 50:
                continue
            E, inliers_essential = cv2.findEssentialMat(corrs.points_1, corrs.points_2,
                                                     intrinsic_mat, method=cv2.RANSAC)
            H, inliers_homography = cv2.findHomography(corrs.points_1, corrs.points_2, method=cv2.RANSAC)

            if inliers_homography.sum() >= inliers_essential.sum() * 0.7:
                continue
            num_inliers, R, t, _ = cv2.recoverPose(E, corrs.points_1, corrs.points_2, intrinsic_mat)

            if num_inliers < 500:
                continue

            view_mat_1 = pose_to_view_mat3x4(Pose(r_mat=np.eye(3), t_vec=np.zeros(3)))
            view_mat_2 = pose_to_view_mat3x4(Pose(r_mat=R.T, t_vec=-R.T @ t))

            _, ids, median_cos = triangulate_correspondences(
                corrs,
                view_mat_1,
                view_mat_2,
                intrinsic_mat,
                params
            )

            if len(ids) < 100:
                continue

            if median_cos > np.cos(5/180 * np.pi):
                continue

            inliers_ratio.append(num_inliers)
            cos_ratio.append(median_cos)
            frames_info.append({'frame1': frame1,
                                'frame2': frame2,
                                'num_inliers': num_inliers,
                                'cos': median_cos,
                                'R': R,
                                't': t}
                               )

    inliers_ratio = sorted(inliers_ratio, reverse=True)
    cos_ratio = sorted(cos_ratio)


    ranks_frames = {}
    ranks = []

    for info in frames_info:
        rank = inliers_ratio.index(info['num_inliers']) * cos_ratio.index(info['cos'])
        ranks.append(rank)
        if not(rank in ranks_frames):
            ranks_frames[rank] = []
        ranks_frames[rank].append((info['frame1'],
                                   info['frame2'],
                                   info['R'],
                                   info['t']))

    min_rank = sorted(ranks)[0]


    frame1, frame2, R, t = ranks_frames[min_rank][0]

    known_view_1 = (frame1, Pose(r_mat=np.eye(3, ), t_vec=np.zeros(3, )))
    known_view_2 = (frame2, Pose(R, -R @ t))

    return known_view_1, known_view_2


def choice_frame(open_, point_cloud_builder, corner_storage, intrinsic_mat, params, conf=0.999, hope=True):
    best = {'frame': None, 'inliers': [], 'rvec': None, 'tvec': None}
    for frame in open_:
        selected_points_ids = np.intersect1d(point_cloud_builder.ids, corner_storage[frame].ids)
        if len(selected_points_ids) < 10 and len(open_) > 1 and hope:
            continue
        selected_points_3d = [point_cloud_builder.points[np.where(point_cloud_builder.ids == id_)[0]][0] for id_ in
                              selected_points_ids]
        selected_points_3d = np.array(selected_points_3d)
        selected_corners = corner_storage[frame]
        selected_points_2d = [selected_corners.points[np.where(selected_corners.ids == id_)[0]][0] for id_ in
                              selected_points_ids]
        selected_points_2d = np.array(selected_points_2d)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(selected_points_3d.astype('float32'),
                                                          selected_points_2d.astype('float32'),
                                                          intrinsic_mat, None,
                                                          reprojectionError=params.max_reprojection_error,
                                                          confidence=conf)
        if not success or (inliers is None):
            continue

        if len(best['inliers']) < len(inliers):
            best['frame'] = frame
            best['inliers'] = inliers
            best['rvec'] = rvec
            best['tvec'] = tvec
    #print(f"BEST FRAME: {best}")
    if best['frame'] is None and hope:
        return choice_frame(open_, point_cloud_builder, corner_storage, intrinsic_mat, params, hope=False)
    return best['frame'], best['inliers'], best['rvec'], best['tvec']


def track_and_calc_colors(camera_parameters: CameraParameters, corner_storage: CornerStorage, frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None, known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    params = TriangulationParameters(4, 1, 0)

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = calc_init_frames(corner_storage, intrinsic_mat, params)

    frame1, frame2 = known_view_1[0], known_view_2[0]
    print(f"FRAME1: {frame1}")
    print(f"FRAME2: {frame2}")
    close = [frame1, frame2]
    open = np.delete(np.arange(len(corner_storage)), close)
    view_mats = np.full(len(corner_storage), None)
    view_known_1, view_known_2 = pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1])
    view_mats[close] = view_known_1, view_known_2



    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points_3d, correspondence_ids, med_cos = triangulate_correspondences(correspondences, view_known_1, view_known_2,
                                                                         intrinsic_mat, params)
    point_cloud_builder = PointCloudBuilder(correspondence_ids, points_3d)
    delta_rec = 0
    recalc = False
    while len(open) > 0:
        delta_rec += 1
        open_ = copy.deepcopy(open)

        selected_frame, inliers, rvec, tvec = choice_frame(open_, point_cloud_builder, corner_storage,
                                                           intrinsic_mat, params)
        if selected_frame is None:
            print('ERROR NO SELECTED FRAME')
            break
        selected_points_ids = np.intersect1d(point_cloud_builder.ids,
                                             corner_storage[selected_frame].ids)

        selected_points_3d = [point_cloud_builder.points[np.where(point_cloud_builder.ids == id_)[0]][0] for id_ in
                              selected_points_ids]

        print(f"selected frame: {selected_frame}")
        #print(point_cloud_builder.ids)

        outliers = np.setdiff1d(np.arange(0, len(selected_points_3d)), inliers.T,
                                assume_unique=True)
        print(f"    OUTLIERS: {outliers}")
        if len(outliers) > 0.2 * len(selected_points_3d) and delta_rec >= 0.2 * (len(open) + len(close)):
            recalc = True
            delta_rec = 0
        point_cloud_builder.remove_points(outliers)

        print(f"    success: {selected_frame}")
        is_enough = 0
        view_mats[selected_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        ids, points3d = get_best_points_for_frames(selected_frame, close, corner_storage,
                                                   view_mats, intrinsic_mat, params, recalc)
        recalc = False
        delta_rec += 1
        point_cloud_builder.add_points(ids, points3d)

        is_enough = 0
        print(f"{len(close)} / {len(open) + len(close)} frames done")
        open = np.delete(open, np.where(open == selected_frame))
        close.append(selected_frame)

    # TODO: implement
    frame_count = len(corner_storage)
    # view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    corners_0 = corner_storage[0]
    # point_cloud_builder = PointCloudBuilder(corners_0.ids[:1],
    # np.zeros((1, 3)))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

