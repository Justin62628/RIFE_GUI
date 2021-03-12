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
import math
class FFmpegQuickReader:
    def __init__(self, ffmpeg_folder, output_txt):

        self.output_txt = os.path.join(output_txt)
        ffmpeg = os.path.join(ffmpeg_folder, "ffmpeg.exe")
        ffprobe = os.path.join( ffmpeg_folder, "ffprobe.exe")
        ffplay = os.path.join( ffmpeg_folder, "ffplay.exe")
        self.tool_list = {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "ffplay": ffplay}
        pass

    def get_tool(self, tool):
        if tool not in self.tool_list:
            print(f"Not Recognize tool: {tool}")
            return ""
        else:
            return self.tool_list[tool]

    def execute(self, tool, command):
        tool = self.get_tool(tool)
        os.system(f'{tool} {command} > {self.output_txt} 2>&1')
        with open(self.output_txt, "r", encoding="utf-8") as tool_read:
            content = tool_read.read()
        # TODO: Warning ffmpeg failed
        return content


class Utils:
    def __init__(self):
        pass
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
        pass
    def gen_next(self, gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None
    def parse_video_info(self, HDR:bool, video_info:dict):
        color_info = dict()
        if HDR:
            color_info.update({"-colorspace": "bt2020nc",
                                    "-color_trc": "smpte2084",
                                    "-color_primaries": "bt2020",
                                    "-color_range": "tv"})
        else:
            color_info.update({"-colorspace": "bt709",
                                    "-color_trc": "bt709",
                                    "-color_primaries": "bt709",
                                    "-color_range": "tv"})
        if "@color_range" in video_info:
            color_info["-color_range"] = video_info["@color_range"]
        if "@color_space" in video_info:
            color_info["-colorspace"] = video_info["@color_space"]
        if "@color_transfer" in video_info:
            color_info["-color_trc"] = video_info["@color_transfer"]
        if "@color_primaries" in video_info:
            color_info["-color_primaries"] = video_info["@color_primaries"]
        return color_info

    def generate_prebuild_map(self):
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
        if is_read:
            tmp = os.listdir(folder)
            for p in tmp:
                if os.path.splitext(p)[-1] in [".jpg", ".png", ".jpeg"]:
                    self.img_list.append(os.path.join(self.seq_folder, p))
            print(f"[IMG.IO] Load {len(self.img_list)} frames from {self.seq_folder}")
        else:
            for t in range(self.thread):
                threading.Thread(target=self.write_buffer, name=f"[IMG.IO] Write Buffer No.{t+1}").start()
            print(f"[IMG.IO] Set {self.seq_folder} As output Folder")

    def nextFrame(self):
        for p in self.img_list:
            p = cv2.imread(p, cv2.IMREAD_UNCHANGED)[:,:,::-1].copy()
            yield p

    def write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                print(f"{threading.current_thread().name}: get None, break")
                break
            cv2.imwrite(img_data[0], cv2.cvtColor(img_data[1], cv2.COLOR_RGB2BGR))

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
