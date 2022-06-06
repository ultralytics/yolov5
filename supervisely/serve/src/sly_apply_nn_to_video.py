import os
import shutil
import time

import cv2

import supervisely as sly
from tqdm import tqdm


class InferenceVideoInterface:
    def __init__(self, api, start_frame_index, frames_count, frames_direction, video_info, imgs_dir):
        self.api = api

        self.video_info = video_info
        self.images_paths = []

        self.start_frame_index = start_frame_index
        self.frames_count = frames_count
        self.frames_direction = frames_direction

        self._video_fps = round(1 / self.video_info.frames_to_timecodes[1])

        self._geometries = []
        self._frames_indexes = []

        self._add_frames_indexes()

        self._frames_path = os.path.join(imgs_dir, f'video_inference_{video_info.id}_{time.time_ns()}', 'frames')

        os.makedirs(self._frames_path, exist_ok=True)

        sly.logger.info(f'{self.__class__.__name__} initialized')

    def _add_frames_indexes(self):
        total_frames = self.video_info.frames_count
        cur_index = self.start_frame_index

        if self.frames_direction == 'forward':
            end_point = cur_index + self.frames_count if cur_index + self.frames_count < total_frames else total_frames
            self._frames_indexes = [curr_frame_index for curr_frame_index in range(cur_index, end_point, 1)]
        else:
            end_point = cur_index - self.frames_count if cur_index - self.frames_count > -1 else -1
            self._frames_indexes = [curr_frame_index for curr_frame_index in range(cur_index, end_point, -1)]
            self._frames_indexes = []

    def download_frames(self):
        for index, frame_index in tqdm(enumerate(self._frames_indexes), desc='Downloading frames', total=len(self._frames_indexes)):
            frame_path = os.path.join(f"{self._frames_path}", f"frame{index:06d}.png")
            self.images_paths.append(frame_path)

            if os.path.isfile(frame_path):
                continue

            img_rgb = self.api.video.frame.download_np(self.video_info.id, frame_index)
            cv2.imwrite(os.path.join(f"{self._frames_path}", f"frame{index:06d}.png"), img_rgb)  # save frame as PNG file

    def __del__(self):
        if os.path.isdir(self._frames_path):
            shutil.rmtree(os.path.dirname(self._frames_path), ignore_errors=True)
