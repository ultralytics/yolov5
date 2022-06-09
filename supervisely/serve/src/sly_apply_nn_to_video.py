import os
import shutil
import time

import cv2
import ffmpeg

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
        self._imgs_dir = imgs_dir

        self._local_video_path = None

        os.makedirs(self._frames_path, exist_ok=True)

        # sly.logger.info(f'{self.__class__.__name__} initialized')

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

    def _download_video_by_frames(self):
        for index, frame_index in tqdm(enumerate(self._frames_indexes), desc='Downloading frames',
                                       total=len(self._frames_indexes)):
            frame_path = os.path.join(f"{self._frames_path}", f"frame{index:06d}.png")
            self.images_paths.append(frame_path)

            if os.path.isfile(frame_path):
                continue

            img_rgb = self.api.video.frame.download_np(self.video_info.id, frame_index)
            cv2.imwrite(os.path.join(f"{self._frames_path}", f"frame{index:06d}.png"),
                        img_rgb)  # save frame as PNG file

    def _download_entire_video(self):
        def videos_to_frames(video_path, frames_range=None):
            def check_rotation(path_video_file):
                # this returns meta-data of the video file in form of a dictionary
                meta_dict = ffmpeg.probe(path_video_file)

                # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
                # we are looking for
                rotate_code = None
                try:
                    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                        rotate_code = cv2.ROTATE_90_CLOCKWISE
                    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                        rotate_code = cv2.ROTATE_180
                    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                except Exception as ex:
                    pass

                return rotate_code

            def correct_rotation(frame, rotate_code):
                return cv2.rotate(frame, rotate_code)

            video_name = (video_path.split('/')[-1]).split('.mp4')[0]
            # output_path = os.path.join(, f'converted_{time.time_ns()}_{video_name}')

            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            count = 0
            rotateCode = check_rotation(video_path)

            while success:
                output_image_path = os.path.join(f"{self._frames_path}", f"frame{count:06d}.png")
                if frames_range:
                    if frames_range[0] <= count <= frames_range[1]:
                        if rotateCode is not None:
                            image = correct_rotation(image, rotateCode)
                        cv2.imwrite(output_image_path, image)  # save frame as PNG file
                        self.images_paths.append(output_image_path)
                else:
                    if rotateCode is not None:
                        image = correct_rotation(image, rotateCode)
                    cv2.imwrite(output_image_path, image)  # save frame as PNG file
                    self.images_paths.append(output_image_path)
                success, image = vidcap.read()
                count += 1

            fps = vidcap.get(cv2.CAP_PROP_FPS)

            return {'frames_path': self._frames_path, 'fps': fps, 'video_path': video_path}

        self._local_video_path = os.path.join(self._imgs_dir, f'{time.time_ns()}_{self.video_info.name}')
        self.api.video.download_path(self.video_info.id, self._local_video_path)
        return videos_to_frames(self._local_video_path, [self.start_frame_index, self.start_frame_index + self.frames_count - 1])

    def download_frames(self):
        if self.frames_count > (self.video_info.frames_count * 0.3):
            sly.logger.debug('Downloading entire video')
            self._download_entire_video()
        else:
            sly.logger.debug('Downloading video frame by frame')
            self._download_video_by_frames()

    def __del__(self):
        if os.path.isdir(self._frames_path):
            shutil.rmtree(os.path.dirname(self._frames_path), ignore_errors=True)

            if self._local_video_path is not None and os.path.isfile(self._local_video_path):
                os.remove(self._local_video_path)
