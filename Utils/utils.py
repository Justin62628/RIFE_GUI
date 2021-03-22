# coding: utf-8
from pprint import pprint
import argparse
import os
import re
import sys
import subprocess
import json
import datetime
import threading
import time
import logging
import traceback
import cv2
import numpy as np
from queue import Queue
from skvideo.io import ffprobe, check_output
import math
import shlex

class CommandResult:
    def __init__(self, command, output_path="output.txt"):
        # self.output_txt = os.path.join(output_txt)
        # ffmpeg = os.path.join(ffmpeg_folder, "ffmpeg.exe")
        # ffprobe = os.path.join(ffmpeg_folder, "ffprobe.exe")
        # ffplay = os.path.join(ffmpeg_folder, "ffplay.exe")
        # self.tool_list = {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "ffplay": ffplay}
        self.command = command
        self.output_path = output_path
        pass

    def execute(self,):
        os.system(f"{self.command} > {self.output_path} 2>&1")
        with open(self.output_path, "r") as tool_read:
            content = tool_read.read()
        # TODO: Warning ffmpeg failed
        return content


class Utils:
    def __init__(self):
        pass
    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'
    def get_logger(self, name, log_path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')

        logger_path = os.path.join(log_path,
                                   f"{datetime.datetime.now().date()}.txt")
        txt_handler = logging.FileHandler(logger_path)
        txt_handler.setLevel(level=logging.DEBUG)
        txt_handler.setFormatter(logger_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level=logging.INFO)
        console_handler.setFormatter(logger_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(txt_handler)
        return logger

    def make_dirs(self, dir_lists):
        for d in dir_lists:
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    def gen_next(self, gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None


    def generate_prebuild_map(self):
        """
        For Inference duplicate frames removal
        :return:
        """
        pos_map = dict()
        for gap in range(2, 8 + 1):
            exp = round(math.sqrt(gap - 1))
            interp_t = {i: (i + 1) / (2 ** exp) for i in range(2 ** exp - 1)}
            real_t = {i: (i + 1) / gap for i in range(gap - 1)}
            for rt in real_t:
                min_ans = (99999999, -1)
                for inp in interp_t:
                    tmp_ = abs(interp_t[inp] - real_t[rt])
                    if tmp_ <= min_ans[0]:
                        min_ans = (tmp_, inp)
                pos_map[(gap, rt)] = min_ans[1]
        return pos_map


class ImgSeqIO:
    def __init__(self, folder=None, is_read=True, thread=4):
        if folder is None or os.path.isfile(folder):
            print(f"[IMG.IO] Invalid ImgSeq Folder: {folder}")
            return
        self.seq_folder = folder
        self.frame_cnt = 0
        self.img_list = list()
        self.write_queue = Queue(maxsize=1000)
        self.thread = thread
        self.use_imdecode = False
        if is_read:
            tmp = os.listdir(folder)
            for p in tmp:
                if os.path.splitext(p)[-1] in [".jpg", ".png", ".jpeg"]:
                    self.img_list.append(os.path.join(self.seq_folder, p))
            print(f"[IMG.IO] Load {len(self.img_list)} frames from {self.seq_folder}")
        else:
            for t in range(self.thread):
                threading.Thread(target=self.write_buffer, name=f"[IMG.IO] Write Buffer No.{t + 1}").start()
            print(f"[IMG.IO] Set {self.seq_folder} As output Folder")

    def nextFrame(self):
        for p in self.img_list:
            if self.use_imdecode:
                p = cv2.imdecode(np.fromfile(p, dtype=np.uint8), 1)[:, :, ::-1].copy()
            else:
                try:
                    p = cv2.imread(p, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                except TypeError:
                    print("Change to use imdecode")
                    self.use_imdecode = True
                    p = cv2.imdecode(np.fromfile(p, dtype=np.uint8), 1)[:, :, ::-1].copy()

            yield p

    def write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                print(f"{threading.current_thread().name}: get None, break")
                break

            if self.use_imdecode:
                cv2.imencode('.png', cv2.cvtColor(img_data[1], cv2.COLOR_RGB2BGR))[1].tofile(img_data[0])
            else:
                try:
                    cv2.imwrite(img_data[0], cv2.cvtColor(img_data[1], cv2.COLOR_RGB2BGR))
                except Exception:
                    print("Change to use imdecode")
                    self.use_imdecode = True
                    cv2.imencode('.png', cv2.cvtColor(img_data[1], cv2.COLOR_RGB2BGR))[1].tofile(img_data[0])

    def writeFrame(self, img):
        img_path = os.path.join(self.seq_folder, f"{self.frame_cnt:0>8d}.png")
        if img is None:
            for t in range(self.thread):
                self.write_queue.put((img_path, None))
        self.write_queue.put((img_path, img))
        self.frame_cnt += 1
        return

    def close(self):
        return


class VideoInfo:
    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def __init__(self, input, HDR=False, ffmpeg=None, img_input=False, **kwargs):
        self.filepath = input
        self.img_input = img_input
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        if ffmpeg is not None:
            self.ffmpeg = os.path.join(ffmpeg, "ffmpeg.exe")
            self.ffprobe = os.path.join(ffmpeg, "ffprobe.exe")
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.ffprobe = "ffprobe"
        self.color_info = dict()
        if HDR:
            self.color_info.update({"-colorspace": "bt2020nc",
                                    "-color_trc": "smpte2084",
                                    "-color_primaries": "bt2020",
                                    "-color_range": "tv"})
        else:
            self.color_info.update({"-colorspace": "bt709",
                                    "-color_trc": "bt709",
                                    "-color_primaries": "bt709",
                                    "-color_range": "tv"})
        self.frames_cnt = 0
        self.frames_size = (0,0)
        self.fps = 0

    def update_frames_info_ffprobe(self):
        result = CommandResult(
            f'{self.ffprobe} -v error -show_streams -select_streams v:0 -v error '
            f'-show_entries stream=index,width,height,r_frame_rate,nb_frames,duration,'
            f'color_primaries,color_range,color_space,color_transfer -print_format json '
            f'{self.fillQuotation(self.filepath)}').execute()
        video_info = json.loads(result)["streams"][0]  # select first video stream as input
        pprint(video_info)
        # update color info
        if "color_range" in video_info:
            self.color_info["-color_range"] = video_info["color_range"]
        if "color_space" in video_info:
            self.color_info["-colorspace"] = video_info["color_space"]
        if "color_transfer" in video_info:
            self.color_info["-color_trc"] = video_info["color_transfer"]
        if "color_primaries" in video_info:
            self.color_info["-color_primaries"] = video_info["color_primaries"]

        # update frame size info
        if 'width' in video_info and 'height' in video_info:
            self.frames_size = (video_info['width'], video_info['height'])

        if "r_frame_rate" in video_info:
            fps_info = video_info["r_frame_rate"].split('/')
            self.fps = int(fps_info[0]) / int(fps_info[1])
            print(f"Auto Find FPS in r_frame_rate: {self.fps}")
        else:
            print("Auto Find FPS Failed")
            return False

        if "nb_frames" in video_info:
            self.frames_cnt = int(video_info["nb_frames"])
            print(f"Auto Find frames cnt in nb_frames: {self.frames_cnt}")
        elif "duration" in video_info:
            self.frames_cnt = round(float(video_info["duration"]) * self.fps)
            print(f"Auto Find Frames Cnt by duration deduction: {self.frames_cnt}")
        else:
            print("Not Find Frames Cnt")
            return False
        return True

    def update_frames_info_cv2(self):
        video_input = cv2.VideoCapture(self.filepath)
        if not self.fps:
            self.fps = video_input.get(cv2.CAP_PROP_FPS)
        if not self.frames_cnt:
            self.frames_cnt = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frames_size = (video_input.get(cv2.CAP_PROP_FRAME_WIDTH),video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update_info(self):
        if self.img_input:
            if os.path.isfile(self.filepath):
                self.filepath = os.path.dirname(self.filepath)
            self.frames_cnt = len(os.listdir(self.filepath))
            return
        self.update_frames_info_ffprobe()
        self.update_frames_info_cv2()

    def get_info(self):
        get_dict = {}
        get_dict.update(self.color_info)
        get_dict["fps"] = self.fps
        get_dict["size"] = self.frames_size
        get_dict["cnt"] = self.frames_cnt
        return get_dict


if __name__ == "__main__":
    check = VideoInfo("L:\Frozen\Remux\Frozen.Fever.2015.1080p.BluRay.REMUX.AVC.DTS-HD.MA.5.1-RARBG.mkv", False, img_input=True)
    check.update_info()
    pprint(check.get_info())

"""
D:\Program\Python\python.exe D:/60-fps-Project/arXiv2020-RIFE-main/RIFE_GUI/Utils/utils.py
[{'avg_frame_rate': '24000/1001',
  'closed_captions': 0,
  'codec_long_name': 'H.265 / HEVC (High Efficiency Video Coding)',
  'codec_name': 'hevc',
  'codec_tag': '0x0000',
  'codec_tag_string': '[0][0][0][0]',
  'codec_time_base': '1001/24000',
  'codec_type': 'video',
  'coded_height': 2160,
  'coded_width': 3840,
  'color_primaries': 'bt2020',
  'color_range': 'tv',
  'color_space': 'bt2020nc',
  'color_transfer': 'smpte2084',
  'display_aspect_ratio': '16:9',
  'disposition': {'attached_pic': 0,
                  'clean_effects': 0,
                  'comment': 0,
                  'default': 0,
                  'dub': 0,
                  'forced': 0,
                  'hearing_impaired': 0,
                  'karaoke': 0,
                  'lyrics': 0,
                  'original': 0,
                  'timed_thumbnails': 0,
                  'visual_impaired': 0},
  'has_b_frames': 3,
  'height': 2160,
  'index': 0,
  'level': 153,
  'pix_fmt': 'yuv420p10le',
  'profile': 'Main 10',
  'r_frame_rate': '24000/1001',
  'refs': 1,
  'sample_aspect_ratio': '1:1',
  'start_pts': 0,
  'start_time': '0.000000',
  'tags': {'BPS-eng': '56790683',
           'DURATION-eng': '01:42:12.877000000',
           'NUMBER_OF_BYTES-eng': '43536284650',
           'NUMBER_OF_FRAMES-eng': '147042',
           '_STATISTICS_TAGS-eng': 'BPS DURATION NUMBER_OF_FRAMES '
                                   'NUMBER_OF_BYTES',
           '_STATISTICS_WRITING_APP-eng': "mkvmerge v37.0.0 ('Leave It') "
                                          '64-bit',
           '_STATISTICS_WRITING_DATE_UTC-eng': '2019-09-23 21:31:26',
           'language': 'eng',
           'title': 'Frozen.2013.2160p.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-FGT'},
  'time_base': '1/1000',
  'width': 3840},
 {'avg_frame_rate': '0/0',
  'bits_per_raw_sample': '8',
  'chroma_location': 'center',
  'closed_captions': 0,
  'codec_long_name': 'Motion JPEG',
  'codec_name': 'mjpeg',
  'codec_tag': '0x0000',
  'codec_tag_string': '[0][0][0][0]',
  'codec_time_base': '0/1',
  'codec_type': 'video',
  'coded_height': 600,
  'coded_width': 1067,
  'color_range': 'pc',
  'color_space': 'bt470bg',
  'disposition': {'attached_pic': 1,
                  'clean_effects': 0,
                  'comment': 0,
                  'default': 0,
                  'dub': 0,
                  'forced': 0,
                  'hearing_impaired': 0,
                  'karaoke': 0,
                  'lyrics': 0,
                  'original': 0,
                  'timed_thumbnails': 0,
                  'visual_impaired': 0},
  'duration': '6132.896000',
  'duration_ts': 551960640,
  'has_b_frames': 0,
  'height': 600,
  'index': 15,
  'level': -99,
  'pix_fmt': 'yuvj444p',
  'profile': 'Baseline',
  'r_frame_rate': '90000/1',
  'refs': 1,
  'start_pts': 0,
  'start_time': '0.000000',
  'tags': {'filename': 'cover_land.jpg', 'mimetype': 'image/jpeg'},
  'time_base': '1/90000',
  'width': 1067},
 {'avg_frame_rate': '0/0',
  'bits_per_raw_sample': '8',
  'chroma_location': 'center',
  'closed_captions': 0,
  'codec_long_name': 'Motion JPEG',
  'codec_name': 'mjpeg',
  'codec_tag': '0x0000',
  'codec_tag_string': '[0][0][0][0]',
  'codec_time_base': '0/1',
  'codec_type': 'video',
  'coded_height': 176,
  'coded_width': 120,
  'color_range': 'pc',
  'color_space': 'bt470bg',
  'disposition': {'attached_pic': 1,
                  'clean_effects': 0,
                  'comment': 0,
                  'default': 0,
                  'dub': 0,
                  'forced': 0,
                  'hearing_impaired': 0,
                  'karaoke': 0,
                  'lyrics': 0,
                  'original': 0,
                  'timed_thumbnails': 0,
                  'visual_impaired': 0},
  'duration': '6132.896000',
  'duration_ts': 551960640,
  'has_b_frames': 0,
  'height': 176,
  'index': 16,
  'level': -99,
  'pix_fmt': 'yuvj444p',
  'profile': 'Baseline',
  'r_frame_rate': '90000/1',
  'refs': 1,
  'start_pts': 0,
  'start_time': '0.000000',
  'tags': {'filename': 'small_cover.jpg', 'mimetype': 'image/jpeg'},
  'time_base': '1/90000',
  'width': 120},
 {'avg_frame_rate': '0/0',
  'bits_per_raw_sample': '8',
  'chroma_location': 'center',
  'closed_captions': 0,
  'codec_long_name': 'Motion JPEG',
  'codec_name': 'mjpeg',
  'codec_tag': '0x0000',
  'codec_tag_string': '[0][0][0][0]',
  'codec_time_base': '0/1',
  'codec_type': 'video',
  'coded_height': 120,
  'coded_width': 213,
  'color_range': 'pc',
  'color_space': 'bt470bg',
  'disposition': {'attached_pic': 1,
                  'clean_effects': 0,
                  'comment': 0,
                  'default': 0,
                  'dub': 0,
                  'forced': 0,
                  'hearing_impaired': 0,
                  'karaoke': 0,
                  'lyrics': 0,
                  'original': 0,
                  'timed_thumbnails': 0,
                  'visual_impaired': 0},
  'duration': '6132.896000',
  'duration_ts': 551960640,
  'has_b_frames': 0,
  'height': 120,
  'index': 17,
  'level': -99,
  'pix_fmt': 'yuvj444p',
  'profile': 'Baseline',
  'r_frame_rate': '90000/1',
  'refs': 1,
  'start_pts': 0,
  'start_time': '0.000000',
  'tags': {'filename': 'small_cover_land.jpg', 'mimetype': 'image/jpeg'},
  'time_base': '1/90000',
  'width': 213},
 {'avg_frame_rate': '0/0',
  'bits_per_raw_sample': '8',
  'chroma_location': 'center',
  'closed_captions': 0,
  'codec_long_name': 'Motion JPEG',
  'codec_name': 'mjpeg',
  'codec_tag': '0x0000',
  'codec_tag_string': '[0][0][0][0]',
  'codec_time_base': '0/1',
  'codec_type': 'video',
  'coded_height': 882,
  'coded_width': 600,
  'color_range': 'pc',
  'color_space': 'bt470bg',
  'disposition': {'attached_pic': 1,
                  'clean_effects': 0,
                  'comment': 0,
                  'default': 0,
                  'dub': 0,
                  'forced': 0,
                  'hearing_impaired': 0,
                  'karaoke': 0,
                  'lyrics': 0,
                  'original': 0,
                  'timed_thumbnails': 0,
                  'visual_impaired': 0},
  'duration': '6132.896000',
  'duration_ts': 551960640,
  'has_b_frames': 0,
  'height': 882,
  'index': 18,
  'level': -99,
  'pix_fmt': 'yuvj444p',
  'profile': 'Baseline',
  'r_frame_rate': '90000/1',
  'refs': 1,
  'start_pts': 0,
  'start_time': '0.000000',
  'tags': {'filename': 'cover.jpg', 'mimetype': 'image/jpeg'},
  'time_base': '1/90000',
  'width': 600}]

Process finished with exit code 0

"""