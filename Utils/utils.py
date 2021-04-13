# coding: utf-8
import math
from pprint import pprint
import os
import re
import json
import datetime
import threading
import time
import logging
import traceback
import cv2
import numpy as np
from queue import Queue
from collections import deque
import shutil
from configparser import ConfigParser, NoOptionError, NoSectionError

from sklearn import linear_model


class CommandResult:
    def __init__(self, command, output_path="output.txt"):
        self.command = command
        self.output_path = output_path
        pass

    def execute(self, ):
        os.system(f"{self.command} > {Utils().fillQuotation(self.output_path)} 2>&1")
        with open(self.output_path, "r") as tool_read:
            content = tool_read.read()
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
        self.resize_param = (480, 270)
        self.crop_param = (0,0,0,0)
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

    def get_norm_img(self, img1):
        img1 = cv2.resize(img1, self.resize_param, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1 = cv2.equalizeHist(img1) #进行直方图均衡化
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img1

    def get_norm_img_diff(self, img1, img2) -> float:
        """
        Normalize Difference
        :param img1: cv2
        :param img2: cv2
        :return: float
        """
        img1 = self.get_norm_img(img1)
        img2 = self.get_norm_img(img2)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        diff = cv2.absdiff(img1, img2).mean()
        return diff

    def rm_edge(self, img):
        """
        return img info of edges
        :param img:
        :return:
        """
        gray = img
        x = gray.shape[1]
        y = gray.shape[0]

        if np.var(gray) == 0:
            """pure image, like white or black"""
            return 0, y, 0, x

        # if np.mean(self.crop_param) != 0:
        #     return self.crop_param

        edges_x = []
        edges_y = []
        edges_x_up = []
        edges_y_up = []
        edges_x_down = []
        edges_y_down = []
        edges_x_left = []
        edges_y_left = []
        edges_x_right = []
        edges_y_right = []

        for i in range(x):
            for j in range(y):
                if int(gray[j][i]) > 10:
                    edges_x_left.append(i)
                    edges_y_left.append(j)
            if len(edges_x_left) != 0 or len(edges_y_left) != 0:
                break

        for i in range(x):
            for j in range(y):
                if int(gray[j][x - i - 1]) > 10:
                    edges_x_right.append(i)
                    edges_y_right.append(j)
            if len(edges_x_right) != 0 or len(edges_y_right) != 0:
                break

        for j in range(y):
            for i in range(x):
                if int(gray[j][i]) > 10:
                    edges_x_up.append(i)
                    edges_y_up.append(j)
            if len(edges_x_up) != 0 or len(edges_y_up) != 0:
                break

        for j in range(y):
            for i in range(x):
                if int(gray[y - j - 1][i]) > 10:
                    edges_x_down.append(i)
                    edges_y_down.append(j)
            if len(edges_x_down) != 0 or len(edges_y_down) != 0:
                break

        edges_x.extend(edges_x_left)
        edges_x.extend(edges_x_right)
        edges_x.extend(edges_x_up)
        edges_x.extend(edges_x_down)
        edges_y.extend(edges_y_left)
        edges_y.extend(edges_y_right)
        edges_y.extend(edges_y_up)
        edges_y.extend(edges_y_down)

        left = min(edges_x) if len(edges_x) else 0  # 左边界
        right = max(edges_x) if len(edges_x) else x # 右边界
        bottom = min(edges_y) if len(edges_y) else 0 # 底部
        top = max(edges_y) if len(edges_y) else y # 顶部

        # image2 = img[bottom:top, left:right]
        self.crop_param = (bottom, top, left, right)
        return bottom, top, left, right


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
                               key=lambda x: int(x[:-4]), reverse=True)
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
                        img = cv2.imread(path)[:, :, ::-1].copy()
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
        "HEVC": {
            "x265": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "NVENC": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
        },
        "H264": {
            "x264": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "NVENC": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
        },
        "ProRes": ["hq", "4444", "4444xq"]
    }
    pixfmt = {
        "HEVC": {
            "x265": ["yuv420p10le", "yuv420p", "yuv422p", "yuv444p", "yuv422p10le", "yuv444p10le", "yuv420p12le",
                     "yuv422p12le", "yuv444p12le"],
            "NVENC": ["p010le", "yuv420p", "yuv444p", "p016le", "yuv444p16le"],
        },
        "H264": {
            "x264": ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", ],
            "NVENC": ["yuv420p", "p010le", "yuv444p", "p016le", "yuv444p16le"],
        },
        "ProRes": ["yuv422p10le", "yuv444p10le"]
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
        try:
            video_info = json.loads(result)["streams"][0]  # select first video stream as input
        except Exception as e:
            print(f"Error: Parse Video Info Failed: {result}")
            raise e
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

class TransitionDetection:
    def __init__(self, scene_stack_length, fixed_scdet=False, scdet_threshold=50, output="", **kwargs):
        self.scdet_threshold=scdet_threshold
        self.fixed_scdet = fixed_scdet
        self.scdet_cnt = 0
        self.scene_stack_len = scene_stack_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.utils = Utils()
        self.dead_thres = 80
        self.born_thres = 2
        self.img1 = None
        self.img2 = None
        self.scdet_cnt = 0
        self.scene_dir = os.path.join(os.path.dirname(output), "scene")
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir, )
        self.scene_stack = Queue(maxsize=scene_stack_length)

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.absdiff_queue))).reshape(-1, 1), np.array(self.absdiff_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue)
        sub_array = diff_array - coef_array
        return math.sqrt(sub_array.var())

    def __judge_mean(self, diff):
        var_before = self.__check_var()
        self.absdiff_queue.append(diff)
        var_after = self.__check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres:
            """Detect new scene"""
            self.see_result(
                f"diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, cnt: {self.scdet_cnt + 1}")
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            return True
        else:
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.see_result(f"diff: {diff:.3f}, False Alarm, cnt: {self.scdet_cnt + 1}")
                self.scdet_cnt += 1
                return True
            # see_result(f"compare: False, diff: {diff}, bm: {before_measure}")
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.see_result(title)


    def see_result(self, title):
        return
        comp_stack = np.hstack((self.img1, self.img2))
        cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
        cv2.moveWindow(title, 1000, 1000)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_scene(self, img1, img2, add_diff=False, no_diff=False) -> bool:
        """
        Check if current scene is scene
        :param img2:
        :param img1:
        :param add_diff:
        :param no_diff: check after "add_diff" mode
        :return: 是转场则返回帧
        """

        diff = self.utils.get_norm_img_diff(img1, img2)
        if self.fixed_scdet:
            if diff < self.scdet_threshold:
                return False
            else:
                self.scdet_cnt += 1
                return True
        self.img1 = img1
        self.img2 = img2
        # if diff == 0:
        #     """重复帧，不可能是转场，也不用添加到判断队列里"""
        #     return False

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            # if diff > dead_thres:
            #     if not add_diff:
            #         see_result(f"compare: True, diff: {diff:.3f}, Sparse Stack, cnt: {self.scdet_cnt + 1}")
            #     self.scene_stack.clear()
            #     return True
            return False

        """Duplicate Frames Special Judge"""
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()
            if not len(self.absdiff_queue):
                return False

        """Judge"""
        return self.__judge_mean(diff)


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
