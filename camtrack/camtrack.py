#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import copy
import cv2

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


# reminder: think about weight
def get_best_points_for_frames(frame: int, frames: List[int], corner_storage: CornerStorage,
                              view_mats: List[np.ndarray], intrinsic_mat: np.ndarray, params: TriangulationParameters, weight = 3) \
        -> Tuple[np.ndarray, np.ndarray]:
    best_ids, best_points = triangulate_points_on_two_frames(frame, frames[0], corner_storage, view_mats, intrinsic_mat, params)
    bi, bp = triangulate_points_on_two_frames(frame, frames[1], corner_storage, view_mats, intrinsic_mat, params)
    if len(bi) > len(best_ids):
        best_ids = bi
        best_points = bp
    for f in frames[2:]:
        ids, points = triangulate_points_on_two_frames(frame, f, corner_storage, view_mats, intrinsic_mat, params)
        if len(ids) > weight * len(best_ids):
            best_ids = ids
            best_points = points
    return best_ids, best_points


def track_and_calc_colors(camera_parameters: CameraParameters, corner_storage: CornerStorage, frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None, known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame1, frame2 = known_view_1[0], known_view_2[0]
    close = [frame1, frame2]
    open = np.delete(np.arange(len(corner_storage)), close)
    view_mats = np.full(len(corner_storage), None)
    view_known_1, view_known_2 = pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1])
    view_mats[close] = view_known_1, view_known_2
    params = TriangulationParameters(1, 1, 0.2)
    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points_3d, correspondence_ids, med_cos = triangulate_correspondences(correspondences, view_known_1, view_known_2,
                                                                         intrinsic_mat, params)
    point_cloud_builder = PointCloudBuilder(correspondence_ids, points_3d)

    '''
    def choice(open_, step=(frame2 - frame1) if (frame2 - frame1) > 7 else 7):
        l_ = None
        r_ = None
        prev = None
        for i in open_:
            if i == 0:
                prev = 0
                continue
            elif prev is None:
                prev = i
                l_ = i - 1
                continue
            if i != prev + 1:
                if l_ is None:
                    l_ = i - 1
                else:
                    r_ = i - 1
            prev = i
        if r_ is None:
            r_ = l_
        if r_ < len(open_) - step - 1:
            return r_ + step
        return np.random.choice(open_)
    '''

    is_enough = 0
    while len(open) > 0:
        open_ = copy.deepcopy(open)
        success = False
        while not success and len(open_) > 0:
            selected_frame = np.random.choice(open_) # choice(open_)
            print(f"selected frame: {selected_frame}")
            selected_points_ids = np.intersect1d(point_cloud_builder.ids, corner_storage[selected_frame].ids)

            selected_points_3d = [point_cloud_builder.points[np.where(point_cloud_builder.ids == id_)[0]][0] for id_ in
                                  selected_points_ids]
            selected_points_3d = np.array(selected_points_3d)
            selected_corners = corner_storage[selected_frame]
            selected_points_2d = [selected_corners.points[np.where(selected_corners.ids == id_)[0]][0] for id_ in
                                  selected_points_ids]
            selected_points_2d = np.array(selected_points_2d)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(selected_points_3d.astype('float32'),
                                                              selected_points_2d.astype('float32'),
                                                              intrinsic_mat, None,
                                                              reprojectionError=params.max_reprojection_error,
                                                              confidence=0.99)

            if not(inliers is None):
                outliers = np.setdiff1d(np.arange(0, len(selected_points_3d)), inliers.T, assume_unique=True)
                # I think it is nice but works so slow
                #if len(outliers) > 0.1 * len(selected_points_3d) and not(is_enough):
                #    success = False
                #if is_enough:
                # point_cloud_builder.remove_points(outliers)
                point_cloud_builder.remove_points(outliers)

            if success:
                print(f"    success: {selected_frame}")
                view_mats[selected_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                ids, points3d = get_best_points_for_frames(selected_frame, close, corner_storage, view_mats,
                                                          intrinsic_mat, params)
                point_cloud_builder.add_points(ids, points3d)

            else:
                print(f"    not success: {selected_frame}")
                print(f"    progress: {len(close)} / {len(open) + len(close)}")
                open_ = np.delete(open_, np.where(open_ == selected_frame))

        if len(open_) == 0 and not success:
            if not(is_enough):
                is_enough = 1
                continue
            break
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


