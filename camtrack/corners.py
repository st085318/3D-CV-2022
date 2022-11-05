#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli,
    filter_frame_corners,
    _to_int_tuple
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    max_quantity_corners = 1000
    quality_level = 0.1#0.05
    min_distance = 5 #10
    size_of_point = 12
    points = cv2.goodFeaturesToTrack(
        image=image_0,
        maxCorners=max_quantity_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        useHarrisDetector=False
    )

    corners = FrameCorners(
        np.arange(0, len(points)),
        points,
        np.array([size_of_point] * len(points))
    )
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        points, status, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255), corners.points, None)
        corners.set_points(points)

        if status is None:
            status = np.array([])

        status = (status == 1).reshape(len(status))
        corners = filter_frame_corners(corners, status.flatten())
        if len(corners.ids) < max_quantity_corners:
            mask = np.ones(image_1.shape, dtype=np.uint8)
            for point in points:
                mask = cv2.circle(mask, _to_int_tuple(point), min_distance, 0, -1)
            new_points = cv2.goodFeaturesToTrack(
                image=image_1,
                maxCorners=max_quantity_corners - len(points),
                qualityLevel=quality_level,
                minDistance=min_distance,
                mask=mask,
                useHarrisDetector=False
            )
            if not(new_points is None):
                corners.new_points(new_points, size_of_point)
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
