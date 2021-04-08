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
import shutil
from configparser import ConfigParser, NoOptionError, NoSectionError


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

    def execute(self, ):
        os.system(f"{self.command} > {self.output_path} 2>&1")
        with open(self.output_path, "r") as tool_read:
            content = tool_read.read()
        # TODO: Warning ffmpeg failed
        return content


class DefaultConfigParser(ConfigParser):
    def get(self, section, option, fallback=None, raw=False):
        try:
            d = self._unify_values(section, None)
        except NoSectionError:
            if fallback is None:
                raise
            else:
                return fallback
        option = self.optionxform(option)
        try:
            value = d[option]
        except KeyError:
            if fallback is None:
                raise NoOptionError(option, section)
            else:
                return fallback

        if type(value) == str and not len(str(value)):
            return fallback

        if type(value) == str and value in ["false", "true"]:
            if value == "false":
                return False
            return True

        return value


class Utils:
    def __init__(self):
        pass

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def get_logger(self, name, log_path, debug=False):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')
        log_path = os.path.join(log_path, "log")  # private dir for logs
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger_path = os.path.join(log_path,
                                   f"{datetime.datetime.now().date()}.txt")
        txt_handler = logging.FileHandler(logger_path)

        txt_handler.setFormatter(logger_formatter)

        console_handler = logging.StreamHandler()
        if debug:
            txt_handler.setLevel(level=logging.DEBUG)
            console_handler.setLevel(level=logging.DEBUG)
        else:
            txt_handler.setLevel(level=logging.INFO)
            console_handler.setLevel(level=logging.INFO)
        console_handler.setFormatter(logger_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(txt_handler)
        return logger

    def make_dirs(self, dir_lists, rm=False):
        for d in dir_lists:
            if rm and os.path.exists(d):
                shutil.rmtree(d)
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    def gen_next(self, gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None

    def generate_prebuild_map(self, exp, req):
        """
        For Inference duplicate frames removal
        :return:
        """
        I_step = 1 / (2 ** exp)
        IL = [x * I_step for x in range(1, 2 ** exp)]
        N_step = 1 / (req + 1)
        NL = [x * N_step for x in range(1, req + 1)]
        KPL = []
        for x1 in NL:
            min = 1
            kpt = 0
            for x2 in IL:
                value = abs(x1 - x2)
                if value < min:
                    min = value
                    kpt = x2
            KPL.append(IL.index(kpt))
        return KPL

    def clean_parsed_config(self, args: dict) -> dict:
        for a in args:
            if args[a] in ["false", "true"]:
                if args[a] == "false":
                    args[a] = False
                else:
                    args[a] = True
                continue
            try:
                tmp = float(args[a])
                try:
                    if not tmp - int(args[a]):
                        tmp = int(args[a])
                except ValueError:
                    pass
                args[a] = tmp
                continue
            except ValueError:
                pass
            if not len(args[a]):
                print(f"Warning: Find Empty Args at '{a}'")
                args[a] = ""
        return args
        pass


class ImgSeqIO:
    def __init__(self, folder=None, is_read=True, thread=4, is_tool=False, start_frame=0):
        if folder is None or os.path.isfile(folder):
            print(f"ERROR - [IMG.IO] Invalid ImgSeq Folder: {folder}")
            return
        if start_frame in [-1, 0]:
            start_frame = 0
        self.seq_folder = folder
        self.frame_cnt = 0
        self.img_list = list()
        self.write_queue = Queue(maxsize=1000)
        self.thread = thread
        self.use_imdecode = False
        if is_tool:
            return
        if is_read:
            tmp = os.listdir(folder)
            for p in tmp:
                if os.path.splitext(p)[-1] in [".jpg", ".png", ".jpeg"]:
                    if self.frame_cnt < start_frame:
                        self.frame_cnt += 1
                        continue
                    self.img_list.append(os.path.join(self.seq_folder, p))
            print(f"INFO - [IMG.IO] Load {len(self.img_list)} frames from {self.seq_folder} at {start_frame}")
        else:
            png_re = re.compile("\d+\.png")
            write_png = sorted([i for i in os.listdir(self.seq_folder) if png_re.match(i)],
                               key=lambda x:int(x[:-4]), reverse=True)
            if len(write_png):
                self.frame_cnt = int(os.path.splitext(write_png[0])[0]) + 1
            for t in range(self.thread):
                threading.Thread(target=self.write_buffer, name=f"[IMG.IO] Write Buffer No.{t + 1}").start()
            print(f"INFO - [IMG.IO] Set {self.seq_folder} As output Folder")

    def read_frame(self, path):
        if self.use_imdecode:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
            return img
        else:
            read_flag = False
            retry = 0
            while not read_flag and retry < 10:
                try:
                    try:
                        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                        return img
                    except TypeError:
                        print("WARNING - [IMG.IO] Change to use imdecode")
                        self.use_imdecode = True
                        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
                        return img
                except Exception:
                    print("CRITICAL - [IMG.IO] Read Failed")
                    print(traceback.format_exc())
                    retry += 1
                    time.sleep(1)
            return None

    def write_frame(self, img, path):
        if self.use_imdecode:
            cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(path)
        else:
            try:
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except Exception:
                print("WARNING - [IMG.IO] Change to use imdecode")
                self.use_imdecode = True
                cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(path)

    def nextFrame(self):
        for p in self.img_list:
            img = self.read_frame(p)
            yield img

    def write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                print(f"INFO - [IMG.IO] {threading.current_thread().name}: get None, break")
                break
            self.write_frame(img_data[1], img_data[0])

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

class EncodePresetAssemply:
    preset = {
        "HEVC":{
            "x265":["slow", "ultrafast", "fast", "medium",  "veryslow"],
            "NVENC":["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
        },
        "H264":{
            "x264":["slow", "ultrafast",  "fast",  "medium",   "veryslow", "placebo",],
            "NVENC":["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
        },
        "ProRes":["hq", "4444", "4444xq"]
    }
    pixfmt = {
        "HEVC":{
            "x265":[ "yuv420p10le", "yuv420p", "yuv422p", "yuv444p", "yuv422p10le",  "yuv444p10le", "yuv420p12le",
                    "yuv422p12le","yuv444p12le"],
            "NVENC":["p010le", "yuv420p",  "yuv444p", "p016le",  "yuv444p16le"],
        },
        "H264":{
            "x264":["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le",  "yuv444p10le",],
            "NVENC":["yuv420p", "p010le", "yuv444p", "p016le",  "yuv444p16le"],
        },
        "ProRes":["yuv422p10le", "yuv444p10le"]
    }

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
        self.frames_size = (0, 0)
        self.fps = 0
        self.duration = 0

    def update_frames_info_ffprobe(self):
        result = CommandResult(
            f'{self.ffprobe} -v error -show_streams -select_streams v:0 -v error '
            f'-show_entries stream=index,width,height,r_frame_rate,nb_frames,duration,'
            f'color_primaries,color_range,color_space,color_transfer -print_format json '
            f'{self.fillQuotation(self.filepath)}').execute()
        video_info = json.loads(result)["streams"][0]  # select first video stream as input
        print("\nInput Video Info:")
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
            print(f"INFO - Auto Find FPS in r_frame_rate: {self.fps}")
        else:
            print("WARNING - Auto Find FPS Failed")
            return False

        if "nb_frames" in video_info:
            self.frames_cnt = int(video_info["nb_frames"])
            print(f"INFO - Auto Find frames cnt in nb_frames: {self.frames_cnt}")
        elif "duration" in video_info:
            self.duration = float(video_info["duration"])
            self.frames_cnt = round(float(self.duration * self.fps))
            print(f"INFO - Auto Find Frames Cnt by duration deduction: {self.frames_cnt}")
        else:
            print("WARNING - FFprobe Not Find Frames Cnt")
            return False
        return True

    def update_frames_info_cv2(self):
        video_input = cv2.VideoCapture(self.filepath)
        if not self.fps:
            self.fps = video_input.get(cv2.CAP_PROP_FPS)
        if not self.frames_cnt:
            self.frames_cnt = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self.duration:
            self.duration = self.frames_cnt / self.fps
        self.frames_size = (video_input.get(cv2.CAP_PROP_FRAME_WIDTH), video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        get_dict["duration"] = self.duration
        return get_dict


if __name__ == "__main__":
    u = Utils()
    cp = DefaultConfigParser(allow_no_value=True)
    cp.read(r"D:\60-fps-Project\arXiv2020-RIFE-main\release\SVFI.Ft.RIFE_GUI.release.v6.2.2.A\RIFE_GUI.ini",
            encoding='utf-8')
    print(cp.get("General", "UseCUDAButton=true", 6))
    print(u.clean_parsed_config(dict(cp.items("General"))))
    #
    # check = VideoInfo("L:\Frozen\Remux\Frozen.Fever.2015.1080p.BluRay.REMUX.AVC.DTS-HD.MA.5.1-RARBG.mkv", False, img_input=True)
    # check.update_info()
    # pprint(check.get_info())
