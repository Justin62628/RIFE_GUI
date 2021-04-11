# coding: gbk
# 这里编码很恶心，要看情况改为utf-8
import argparse
import datetime
import math
import os
import re
import sys
import threading
import time
from collections import deque
from queue import Queue
import shlex
import cv2
import numpy as np
import tqdm
from skvideo.io import FFmpegWriter, FFmpegReader
# from sklearn import linear_model
import shutil
import traceback
import psutil
from Utils.utils import Utils, ImgSeqIO, VideoInfo, DefaultConfigParser

# 6.2.8 2021/4/11
Utils = Utils()
"""Set Path Environment"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing working dir to {0}".format(dname))
os.chdir(os.path.dirname(dname))
print("Added {0} to temporary PATH".format(dname))
sys.path.append(dname)

parser = argparse.ArgumentParser(prog="#### RIFE CLI tool/补帧分步设置命令行工具 by Jeanna ####",
                                 description='Interpolation for sequences of images')
basic_parser = parser.add_argument_group(title="Basic Settings, Necessary")
basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                          help="原视频/图片序列文件夹路径")
basic_parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                          help="成品输出的路径，注意默认在项目文件夹")
basic_parser.add_argument("-c", '--config', dest='config', type=str, required=True, help="配置文件路径")
basic_parser.add_argument('--concat-only', dest='concat_only', action='store_true', help='只执行合并已有区块操作')

args_read = parser.parse_args()
cp = DefaultConfigParser(allow_no_value=True)
cp.read(args_read.config, encoding='utf-8')
cp_items = dict(cp.items("General"))
args = Utils.clean_parsed_config(cp_items)
args.update(vars(args_read))  # update -i -o -c，将命令行参数更新到config生成的字典

# 设置可见的gpu
if int(args["use_specific_gpu"]) != -1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['use_specific_gpu']}"

try:
    import inference  # 导入补帧模块
except Exception:
    import inference_A as inference

    print("Error: Import Torch Failed, use NCNN instead")
    traceback.print_exc()
    args.update({"ncnn": True})


class InterpWorkFlow:
    def __init__(self, __args, **kwargs):
        self.args = __args

        if os.path.isfile(self.args["output"]):
            self.project_dir = os.path.dirname(self.args["output"])
        else:
            self.project_dir = self.args["output"]

        """Set Logger"""
        sys.path.append(self.project_dir)
        self.logger = Utils.get_logger("[ARGS]", self.project_dir)
        self.logger.info(f"Initial New Interpolation Project: project_dir: %s, INPUT_FILEPATH: %s", self.project_dir,
                         self.args["input"])

        """Set FFmpeg"""
        self.ffmpeg = os.path.join(self.args["ffmpeg"], "ffmpeg.exe")
        self.ffplay = os.path.join(self.args["ffmpeg"], "ffplay.exe")
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.logger.warning("Not find selected ffmpeg, use default")

        """Set input output and initiate environment"""
        self.input = self.args["input"]
        self.output = self.args["output"]
        self.input_dir = os.path.join(self.project_dir, 'frames')
        self.interp_dir = os.path.join(self.project_dir, 'interp')
        self.scene_dir = os.path.join(self.project_dir, 'scenes')
        env = [self.input_dir, self.interp_dir, self.scene_dir]
        Utils.make_dirs(env, rm=True)

        """Load Interpolation Exp"""
        self.exp = self.args["exp"]

        """Get input's info"""
        self.video_info_instance = VideoInfo(**self.args)
        self.video_info_instance.update_info()
        self.video_info = self.video_info_instance.get_info()
        if self.args["batch"] and not self.args["img_input"]:  # 检测到批处理，且输入不是文件夹，使用检测到的帧率
            self.fps = self.video_info["fps"]
        elif self.args["fps"]:
            self.fps = self.args["fps"]
        else:  # 用户有毒，未发现有效的输入帧率，用检测到的帧率
            if self.video_info["fps"] is None or not self.video_info["fps"]:
                raise OSError("Input File not valid")
            self.fps = self.video_info["fps"]

        if self.args["target_fps"]:
            self.target_fps = self.args["target_fps"]
        else:
            self.target_fps = (2 ** self.exp) * self.fps
        self.all_frames_cnt = int(self.video_info["cnt"])  # 视频总帧数
        if self.args["any_fps"]:  # 使用任意帧率，要相应更新总帧数
            self.all_frames_cnt = int(self.video_info["duration"] * self.target_fps)

        self.crop_param = [0, 0]  # crop parameter, 裁切参数
        crop_param = self.args["crop"]
        if crop_param not in ["", "0", None]:
            width_black, height_black = crop_param.split(":")
            width_black = int(width_black)
            height_black = int(height_black)
            self.crop_param = [width_black, height_black]
            self.logger.info(f"Update Crop Parameters to {self.crop_param}")

        """initiation almost ready"""
        self.logger.info(
            f"Check Interpolation Source, FPS: {self.fps}, TARGET FPS: {self.target_fps}, "
            f"FRAMES_CNT: {self.all_frames_cnt}, EXP: {self.exp}")

        self.rife_core = None  # 用于补帧的模块

        self.render_gap = self.args["render_gap"]  # 每个chunk的帧数
        self.frame_reader = None  # 读帧的迭代器／帧生成器
        self.render_thread = None  # 帧渲染器

        mem = psutil.virtual_memory()
        free_mem = round(mem.free / 1024 / 1024)
        self.frames_output_size = round(free_mem / (sys.getsizeof(
            np.random.rand(3, round(self.video_info["size"][0]), round(self.video_info["size"][1])))/1024/1024) * 0.8)
        if self.frames_output_size < 100:
            self.frames_output_size = 100
        self.frames_output = Queue(maxsize=self.frames_output_size)  # 补出来的帧序列队列（消费者）

        self.render_info_pipe = {"rendering": (0, 0, 0, 0)}  # 有关渲染的实时信息，目前只有一个key
        self.scene_stack_len = int(0.3 * self.fps)  # 用于判断转场的帧的absdiff的值的固定长队列的长度
        self.scene_stack = deque(maxlen=self.scene_stack_len)  # absdiff队列

        self.dup_skip_limit = int(0.5 * self.fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环

        self.main_event = threading.Event()
        self.render_lock = threading.Event()  # 渲染锁，没有用
        self.main_event.set()

        """Set output's color info"""
        self.color_info = {}
        for k in self.video_info:
            if k.startswith("-"):
                self.color_info[k] = self.video_info[k]

        """Set output extension"""
        self.output_ext = "." + self.args["output_ext"]
        if self.args["encoder"] == "ProRes":
            self.output_ext = ".mov"
            # TODO optimize this
        pass

    def generate_frame_reader(self, start_frame=-1):
        """
        输入帧迭代器
        :param start_frame:
        :return:
        """
        """If input is sequence of frames"""
        if self.args["img_input"]:
            return ImgSeqIO(folder=self.input, is_read=True, start_frame=start_frame)

        """If input is a video"""
        input_dict = {"-vsync": "0"}
        output_dict = {
            "-vframes": str(int(abs(self.all_frames_cnt * 100)))}  # use read frames cnt to avoid ffprobe, fuck

        output_dict.update(self.color_info)

        vf_args = "copy"

        """任意帧率输出基础支持"""
        if self.args["any_fps"]:
            # input_dict.update({"-vsync": "cfr"})
            vf_args += f",minterpolate=fps={self.target_fps}:mi_mode=dup"
            # output_dict = {"-r": f"{self.target_fps}"}  # no need of -vframes

        if len(self.args["resize"]):
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": self.args["resize"].replace(":", "x").replace("*", "x")})
            # output_dict.update({"-sws_flags": "lanczos+full_chroma_inp"})
            # vf_args += f",scale={self.args['resize']}"
        if start_frame not in [-1, 0]:
            vf_args += f",trim=start_frame={start_frame}"

        """Quick Extraction"""
        if not self.args["quick_extract"]:
            vf_args += f",format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24"

        """Update video filters"""
        output_dict["-vf"] = vf_args
        self.logger.debug(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=self.input, inputdict=input_dict, outputdict=output_dict)

    def generate_frame_renderer(self, output_path):
        """
        渲染帧
        :param output_path:
        :return:
        """
        """If output is sequence of frames"""
        if self.args["img_output"]:
            return ImgSeqIO(folder=self.output, is_read=False)

        """Output Video"""
        input_dict = {"-vsync": "cfr", "-r": f"{self.fps * 2 ** self.exp}"}

        output_dict = {"-r": f"{self.target_fps}", "-preset": self.args["preset"], "-pix_fmt": self.args["pix_fmt"]}
        if self.args["any_fps"]:
            input_dict.update({"-r": f"{self.target_fps}"})
            output_dict.update({"-r": f"{self.target_fps}"})
        output_dict.update(self.color_info)

        """Slow motion design"""
        if self.args["slow_motion"]:
            input_dict.update({"-r": f"{self.target_fps}"})
            output_dict.pop("-r")
        vf_args = "copy"  # debug
        output_dict.update({"-vf": vf_args})

        """Assign Render Codec"""
        if self.args["encoder"] == "H264/AVC":
            if self.args["hwaccel"]:
                output_dict.update({"-c:v": "h264_nvenc", "-rc:v": "vbr_hq"})
            else:
                output_dict.update({"-c:v": "libx264", "-tune": "grain"})
        elif self.args["encoder"] == "H265/HEVC":
            if self.args["hwaccel"]:
                output_dict.update({"-c:v": "hevc_nvenc", "-rc:v": "vbr_hq"})
            else:
                output_dict.update(
                    {"-c:v": "libx265", "-tune": "grain"})
        elif self.args["encoder"] == "ProRes":
            output_dict.pop("-preset")  # leave space for ProRes Profile
            output_dict.update({"-c:v": "prores_ks", "-profile:v": self.args["preset"], "-quant_mat": "hq"})

        if self.args["encoder"] not in ["ProRes"]:
            if self.args["crf"] and self.args["UseCRF".lower()]:
                if self.args["hwaccel"]:
                    output_dict.update({"-cq:v": str(self.args["crf"])})
                else:
                    output_dict.update({"-crf": str(self.args["crf"])})
            if self.args["UseTargetBitrate".lower()] and type(self.args["bitrate"]) == str and len(
                    self.args["bitrate"]):
                output_dict.update({"-b:v": str(self.args["bitrate"])})

        self.logger.debug(f"writer: {output_dict}, {input_dict}")

        """Customize FFmpeg Render Command"""
        ffmpeg_customized_command = {}
        if len(self.args["ffmpeg_customized"]):
            shlex_out = shlex.split(self.args["ffmpeg_customized"])
            if len(shlex_out) % 2 != 0:
                self.logger.warning(f"Customized FFmpeg is invalid: {self.args['ffmpeg_customized_command']}")
            else:
                for i in range(int(len(shlex_out) / 2)):
                    ffmpeg_customized_command.update({shlex_out[i * 2]: shlex_out[i * 2 + 1]})
        self.logger.debug(ffmpeg_customized_command)
        output_dict.update(ffmpeg_customized_command)

        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)

    def check_chunk(self, del_chunk=False):
        """
        Get Chunk Start
        :param: del_chunk: delete all chunks existed
        :return:
        """
        chunk_list = list()
        chunk_regex = rf"chunk-[\d+].*?\{self.output_ext}"
        for f in os.listdir(self.project_dir):
            if re.match(chunk_regex, f):
                if del_chunk:
                    os.remove(os.path.join(self.project_dir, f))
                else:
                    chunk_list.append(f)
        """If remove only"""
        if del_chunk:
            return 1, 0

        """Manually Prioritized"""
        if self.args["interp_start"] != 0 or self.args["chunk"] != 0:
            return int(self.args["chunk"]), int(self.args["interp_start"])

        """Not find previous chunk"""
        if not len(chunk_list):
            return 1, 0

        """Remove last chunk(high possibility of dilapidation)"""
        chunk_list.sort(key=lambda x: int(x.split('-')[2]))
        os.remove(os.path.join(self.project_dir, chunk_list[-1]))
        chunk_list.pop(-1)
        if not len(chunk_list):
            return 1, 0

        self.logger.info("Found Previous Chunks")
        last_chunk = chunk_list[-1]  # select last chunk to assign start frames
        chunk_regex = rf"chunk-(\d+)-(\d+)-(\d+)\{self.output_ext}"
        match_result = re.findall(chunk_regex, last_chunk)[0]

        chunk = int(match_result[0])
        last_frame = int(match_result[2])
        return chunk + 1, last_frame + 1

    def render(self, chunk_cnt, start_frame):
        """
        Render thread
        :param chunk_cnt:
        :param start_frame:
        :return:
        """
        end_frame = int(start_frame + self.render_gap / (2 ** self.exp) - 1)

        chunk_regex = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, end_frame, self.output_ext)
        output_path = chunk_regex
        output_path = os.path.join(self.project_dir, output_path)

        chunk_frame_cnt = 1  # number of frames of current output chunk
        frame_writer = self.generate_frame_renderer(output_path)  # get frame renderer

        if not self.args["img_output"]:
            self.logger.info(f"First Chunk Render Path: {output_path}")
        while True:
            if not self.main_event.is_set():
                self.logger.warning("Main interpolation thread Dead, break")  # 主线程已结束，这里的锁其实没用，调试用的
                frame_writer.close()
                break

            frame_data = self.frames_output.get()
            if frame_data is None:
                frame_writer.close()
                break

            frame = frame_data[1]
            frame_cnt = frame_data[0]
            frame_writer.writeFrame(frame)

            chunk_frame_cnt += 1
            self.render_info_pipe["rendering"] = (chunk_cnt, start_frame, end_frame, frame_cnt)  # update render info

            if not chunk_frame_cnt % self.render_gap:
                chunk_cnt += 1
                start_frame = end_frame + 1
                end_frame = int(start_frame + self.render_gap / (2 ** self.exp) - 1)
                output_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, end_frame,
                                                                       self.output_ext)
                output_path = os.path.join(self.project_dir, output_path)
                frame_writer.close()
                frame_writer = self.generate_frame_renderer(output_path)

        return

    def feed_to_render(self, frames_list: list, is_scene=False, is_end=False):
        """
        维护输出帧数组的输入（往输出渲染线程喂帧
        :param frames_list:
        :param is_scene: 是否是转场
        :param is_end: 是否是视频结尾
        :return:
        """
        frames_list_len = len(frames_list)
        if not len(frames_list) and is_end:
            self.frames_output.put(None)
            self.logger.info("Put None to write_buffer")

        for frame_i in range(frames_list_len):
            self.frames_output.put(frames_list[frame_i])  # 往输出队列（消费者）喂正常的帧
            if frame_i == frames_list_len - 1:
                if is_scene:
                    for put_i in range(2 ** self.exp - 1):
                        self.frames_output.put(frames_list[frame_i])  # 喂转场导致的重复帧，这个重复帧是framelist的最后一个元素
                    return
                if is_end:
                    self.frames_output.put(None)
                    self.logger.info("Put None to write_buffer")
                    return
        pass

    def check_scene(self, diff, add_diff=False, no_diff=False) -> bool:
        """
        Check if current scene is scene
        :param diff:
        :param add_diff:
        :return: 是转场则返回帧
        """

        # def check_coef():
        #     reg = linear_model.LinearRegression()
        #     reg.fit(np.array(range(len(self.scene_stack))).reshape(-1, 1), np.array(self.scene_stack).reshape(-1, 1))
        #     return reg.coef_

        def judge_mean():
            before_measure = np.mean(self.scene_stack) * 2.5  # 判断当前帧放入队列前的帧序列absdiff的方差
            if before_measure < diff and np.max(self.scene_stack) * 0.9 < diff and diff > 2:
                """Detect new scene"""
                self.scene_stack.clear()
                return True
            else:
                self.scene_stack.append(diff)
                return False

        def judge_var():
            pass

        if self.args.get("fixed_scdet", False):
            if diff < self.args["scdet_threshold"]:
                return False
            else:
                return True
        if diff == 0:
            """重复帧，不可能是转场，也不用添加到判断队列里"""
            return False

        if len(self.scene_stack) < self.scene_stack_len or add_diff:
            if diff not in self.scene_stack:
                self.scene_stack.append(diff)
            return False

        """Duplicate Frames Special Judge"""
        if no_diff:
            self.scene_stack.pop()
            if not len(self.scene_stack):
                return False

        """Judge"""
        return judge_mean()

    def check_scene_ST(self, diff, add_diff=False, no_diff=False) -> bool:
        """
        Check if current scene is scene
        :param diff:
        :param add_diff:
        :return: 是转场则返回帧
        """

        if self.args.get("fixed_scdet", False):
            if diff < self.args["scdet_threshold"]:
                return False
            else:
                return True
        if diff == 0:
            """重复帧，不可能是转场，也不用添加到判断队列里"""
            return False

        if len(self.scene_stack) < self.scene_stack_len or add_diff:
            if diff not in self.scene_stack:
                self.scene_stack.append(diff)
            return False

        """Duplicate Frames Special Judge"""
        if no_diff:
            self.scene_stack.pop()
            if not len(self.scene_stack):
                return False

        """Judge"""
        before_measure = np.var(self.scene_stack)  # 判断当前帧放入队列前的帧序列absdiff的方差
        self.scene_stack.append(diff)
        after_measure = np.var(self.scene_stack)  # 判断放入后的
        if abs(after_measure - before_measure) > self.args["scdet_threshold"]:
            """Detect new scene"""
            self.scene_stack.clear()
            return True
        else:
            return False

    def crop_read_img(self, img):
        """
        Crop using self.crop parameters
        :param img:
        :return:
        """
        if img is None:
            return img

        h, w, _ = img.shape
        if self.crop_param[0] > w or self.crop_param[1] > h:
            return img
        return img[self.crop_param[1]:h - self.crop_param[1], self.crop_param[0]:w - self.crop_param[0]]

    def nvidia_run(self):
        """
        Go through all procedures to produce interpolation result
        :return:
        """
        """Update RIFE Core, NVIDIA Only"""
        self.rife_core = inference.RifeInterpolation(self.args)
        self.rife_core.initiate_rife(args)

        """Get Start Info"""
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.frame_reader = self.generate_frame_reader(start_frame)

        """Get Renderer"""
        self.render_thread = threading.Thread(target=self.render, name="[ARGS] RenderThread",
                                              args=(chunk_cnt, start_frame,))
        self.render_thread.setDaemon(True)
        self.render_thread.start()

        """Get Frames to interpolate"""
        videogen = self.frame_reader.nextFrame()
        img1 = self.crop_read_img(Utils.gen_next(videogen))
        now_frame = start_frame
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}")

        is_end = False
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit=" frames")
        pbar.update(n=start_frame)
        pbar.unpause()
        recent_scene = 0  # 最近的转场
        previous_cnt = now_frame  # 上一帧的计数
        scedet_info = {"0": 0, "1": 0, "1+": 0}  # 帧种类，0为转场，1为正常帧，1+为重复帧，即两帧之间的计数关系

        extract_cnt = 0

        """Update Mode Info"""
        if self.args["any_fps"]:
            self.args["dup_threshold"] = self.args["dup_threshold"] if self.args["dup_threshold"] > 0.5 else 0.5

        while True:
            if is_end:
                break
            img0 = img1
            frames_list = []

            if self.args["any_fps"]:
                """任意帧率模式"""
                frames_list.append([now_frame, img0])
                img1 = self.crop_read_img(Utils.gen_next(videogen))
                now_frame += 1
                extract_cnt += 1
                if img1 is None:
                    self.feed_to_render(frames_list, is_end=True)
                    break

                diff = cv2.absdiff(img0, img1).mean()

                skip = 0  # 用于记录跳过的帧数

                """Find Scene"""
                if not self.args["no_scdet"] and self.check_scene(diff):
                    self.feed_to_render(frames_list)  # no need to update scene flag
                    recent_scene = now_frame
                    # now_frame += 1  # to next frame img0 = img1
                    scedet_info["0"] += 1
                    continue
                else:
                    if diff < self.args["dup_threshold"]:
                        before_img = img1
                        last_diff = diff
                        while diff < self.args["dup_threshold"]:
                            skip += 1
                            scedet_info["1+"] += 1
                            img1 = self.crop_read_img(Utils.gen_next(videogen))
                            extract_cnt += 1

                            if img1 is None:
                                img1 = before_img
                                is_end = True
                                break

                            diff = cv2.absdiff(img0, img1).mean()
                            if diff != last_diff:  # detect duplicated frames
                                self.check_scene(diff, add_diff=True)  # update scene stack
                                last_diff = diff
                            if skip == int(self.dup_skip_limit * self.target_fps / self.fps):
                                """超过重复帧计数限额，直接跳出"""
                                break

                        # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                        if not self.args["no_scdet"] and self.check_scene(diff, no_diff=True):
                            skip -= 1  # 两帧间隔计数器-1
                            if skip:
                                # 将转场前一帧作为img1，补足
                                exp = int(math.log(skip, 2)) + 1
                                interp_output = self.rife_core.generate_interp(img0, before_img, exp,
                                                                               self.args["scale"])
                                kpl = Utils.generate_prebuild_map(exp, skip)
                                for x in kpl:
                                    frames_list.append([now_frame, interp_output[x]])
                            frames_list.append([now_frame, before_img])
                            recent_scene = now_frame
                            scedet_info["0"] += 1
                            now_frame += skip + 1

                        elif skip != 0:
                            exp = int(math.log(skip, 2)) + 1
                            interp_output = self.rife_core.generate_interp(img0, img1, exp, self.args["scale"])
                            kpl = Utils.generate_prebuild_map(exp, skip)
                            for x in kpl:
                                frames_list.append([now_frame, interp_output[x]])
                            now_frame += skip
                    else:
                        scedet_info["1"] += 1

                self.feed_to_render(frames_list, is_end=is_end)
                pass
            elif self.args["remove_dup"]:
                """Remove duplicated Frames"""
                img1 = self.crop_read_img(Utils.gen_next(videogen))

                if img1 is None:
                    frames_list.append([now_frame, img0])
                    self.feed_to_render(frames_list, is_end=True)
                    break

                gap_frames_list = list()  # 用于去除重复帧的临时帧列表
                gap_frames_list.append(img0)  # 放入当前判断区间头一帧

                diff = cv2.absdiff(img0, img1).mean()

                skip = 0  # 用于记录跳过的帧数
                is_scene = False

                """Find Scene"""
                if not self.args["no_scdet"] and self.check_scene(diff):
                    frames_list.append([now_frame, img0])
                    self.feed_to_render(frames_list, is_scene=True)
                    recent_scene = now_frame
                    now_frame += 1  # to next frame img0 = img1
                    scedet_info["0"] += 1
                    continue
                else:
                    if diff < self.args["dup_threshold"]:
                        before_img = img1
                        while diff < self.args["dup_threshold"]:
                            skip += 1
                            scedet_info["1+"] += 1
                            img1 = self.crop_read_img(Utils.gen_next(videogen))

                            if img1 is None:
                                img1 = before_img
                                is_end = True
                                break

                            diff = cv2.absdiff(img0, img1).mean()
                            self.check_scene(diff, add_diff=True)  # update scene stack
                            if skip == self.dup_skip_limit:
                                """超过重复帧计数限额，直接跳出"""
                                break

                        # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                        if not self.args["no_scdet"] and self.check_scene(diff, no_diff=True):
                            skip -= 1  # 两帧间隔计数器-1
                            if not skip:
                                gap_frames_list.append(img0)
                            else:
                                # 将转场前一帧作为img1，补足
                                exp = int(math.log(skip, 2)) + 1
                                interp_output = self.rife_core.generate_interp(img0, before_img, exp,
                                                                               self.args["scale"])
                                kpl = Utils.generate_prebuild_map(exp, skip)
                                for x in kpl:
                                    gap_frames_list.append(interp_output[x])
                                gap_frames_list.append(before_img)

                            is_scene = True
                            scedet_info["0"] += 1
                        elif skip != 0:
                            exp = int(math.log(skip, 2)) + 1
                            interp_output = self.rife_core.generate_interp(img0, img1, exp, self.args["scale"])
                            kpl = Utils.generate_prebuild_map(exp, skip)
                            for x in kpl:
                                gap_frames_list.append(interp_output[x])
                    else:
                        scedet_info["1"] += 1

                # 进行到正常补帧序列（可修改）
                if not is_scene:
                    gap_frames_list.append(img1)

                for pos in range(len(gap_frames_list) - 1):
                    frames_list.append([now_frame, gap_frames_list[pos]])
                    output = self.rife_core.generate_interp(gap_frames_list[pos], gap_frames_list[pos + 1], self.exp,
                                                            self.args["scale"])
                    for mid_index in range(len(output)):
                        frames_list.append([now_frame, output[mid_index]])
                    now_frame += 1

                if is_scene:
                    frames_list.append([now_frame, gap_frames_list[-1]])
                    now_frame += 1
                self.feed_to_render(frames_list, is_scene, is_end)
            else:
                """No Duplicated Frames Removal, nor Any FPS mode"""
                img1 = self.crop_read_img(Utils.gen_next(videogen))

                if img1 is None:
                    frames_list.append((now_frame, img0))
                    self.feed_to_render(frames_list, is_end=True)
                    # is_end = True
                    break

                if not self.args["no_scdet"]:

                    diff = cv2.absdiff(img0, img1).mean()

                    if self.check_scene(diff):
                        """!!!scene"""
                        frames_list.append((now_frame, img0))
                        self.feed_to_render(frames_list, is_scene=True)
                        recent_scene = now_frame
                        now_frame += 1  # to next frame img0 = img1
                        scedet_info["0"] += 1
                        continue

                frames_list.append((now_frame, img0))

                """Generate Interpolated Result"""
                interp_output = self.rife_core.generate_interp(img0, img1, self.exp, self.args["scale"])

                for mid_index in range(len(interp_output)):
                    frames_list.append([now_frame, interp_output[mid_index]])
                self.feed_to_render(frames_list)
                now_frame += 1  # to next frame img0 = img1
                scedet_info["1"] += 1

            """Update Render Info"""
            rsq = self.render_info_pipe["rendering"]  # render status quo
            """(chunk_cnt, start_frame, end_frame, frame_cnt)"""

            pbar.set_description(
                f"Process at Chunk {rsq[0]:0>3d}")
            pbar.set_postfix({"Render": f"{rsq[3]}", "Current": f"{now_frame}", "Scene": f"{recent_scene}"})
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame

        """End of Interpolation Process"""
        self.logger.info(f"Scedet Status Quo: {scedet_info}")

        """Wait for Render Thread to finish"""
        now_frame = self.all_frames_cnt
        while self.render_thread.is_alive():
            """Update Render Info"""
            rsq = self.render_info_pipe["rendering"]  # render status quo
            """(chunk_cnt, start_frame, end_frame, frame_cnt)"""
            pbar.set_description(
                f"Process at Chunk {rsq[0]:0>3d}")
            pbar.set_postfix({"Render": f"{rsq[3]}", "Current": f"{now_frame}", "Scene": f"{recent_scene}"})
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame
            time.sleep(0.1)
        pbar.close()

        """Concat the chunks"""
        if not self.args["no_concat"] and not self.args["img_output"]:
            self.concat_all()
            return

    def amd_run(self):
        """
        Use AMD Card to Interpolate
        :return:
        """

        self.args["input_dir"] = self.input_dir
        self.rife_core = inference.NCNNinterpolator(self.args)
        self.rife_core.initiate_rife()

        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.frame_reader = self.generate_frame_reader(start_frame)
        self.render_thread = threading.Thread(target=self.render, name="[ARGS] RenderThread",
                                              args=(chunk_cnt, start_frame,))
        self.render_thread.setDaemon(True)
        self.render_thread.start()

        videogen = self.frame_reader.nextFrame()
        img1 = self.crop_read_img(Utils.gen_next(videogen))
        now_frame = start_frame
        if img1 is None:
            self.logger.critical(f"Input file not valid: {self.input}")
            self.main_event.clear()
            sys.exit(0)

        is_end = False
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit=" frames")
        pbar.update(n=start_frame)
        pbar.unpause()
        recent_scene = 0
        scedet_info = {"0": 0, "1": 0, "1+": 0}
        scenes_list = []
        previous_cnt = now_frame
        origianl_frame_cnt = 0  # to mark relation between original frame and interp frame
        frame_ioer = ImgSeqIO(self.input_dir, is_tool=True)

        while True:
            if is_end:
                break
            img0 = img1
            img1 = self.crop_read_img(Utils.gen_next(videogen))
            if img1 is None:
                is_end = True

            if not self.args["no_scdet"] and img1 is not None:
                diff = cv2.absdiff(img0, img1).mean()
                if self.check_scene(diff):
                    """!!!scene"""
                    recent_scene = now_frame
                    scenes_list.append(origianl_frame_cnt)
                    scedet_info["0"] += 1
                else:
                    scedet_info["1"] += 1

            now_frame += 1  # to next frame img0 = img1
            origianl_frame_cnt += 1
            frame_path = os.path.join(self.input_dir, f"{origianl_frame_cnt:0>8d}.png")
            frame_ioer.write_frame(img0, frame_path)

            rsq = self.render_info_pipe["rendering"]  # render status quo
            pbar.set_description(
                f"Process at Step 1, Extract Frames: Chunk {rsq[0]}")
            pbar.set_postfix({"Frame": f"{now_frame}", "Scene": f"{recent_scene}"})
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame

            if now_frame % (self.render_gap / (2 ** self.exp)) == 0 or img1 is None:
                """Time for interpolation"""
                self.rife_core = inference.NCNNinterpolator(self.args)
                self.rife_core.initiate_rife()
                self.rife_core.start()
                interp_scene_list = {}
                interp_scene_edge_list = {}

                now_dir = ""
                now_cnt = 0  # for interp and scene positioning
                while self.rife_core.is_alive():
                    if "now_dir" not in self.rife_core.supervise_data:
                        "NCNN initializing"
                        time.sleep(0.1)
                        continue

                    if now_dir != self.rife_core.supervise_data["now_dir"]:
                        pbar.close()
                        pbar = tqdm.tqdm(total=self.rife_core.supervise_data["input_cnt"], unit=" frames")
                        now_dir = self.rife_core.supervise_data["now_dir"]
                        now_cnt = 0

                    time.sleep(0.1)
                    pbar.update(self.rife_core.supervise_data["now_cnt"] - now_cnt)
                    now_cnt = self.rife_core.supervise_data["now_cnt"]
                    rsq = self.render_info_pipe["rendering"]  # render status quo
                    pbar.set_description(
                        f"Process at Step 2, Interpolation: Chunk {rsq[0]}")
                    pbar.set_postfix({"Round": f"{self.rife_core.supervise_data['now_dir']}",
                                      "Status": f"{now_frame / self.all_frames_cnt * 100:.2f}%"})

                # generate scene map info
                for scene in scenes_list:
                    interp_scene = scene * 2 ** self.exp + 1
                    for i in range(1, 2 ** self.exp):
                        interp_scene_list[str(interp_scene + i)] = True
                    interp_scene_edge_list[str(interp_scene)] = True

                """Final interpolation is done, time to render"""
                frames_list = []
                interp_list = sorted(os.listdir(self.interp_dir), key=lambda x: x[:-4])

                pbar.close()
                pbar = tqdm.tqdm(total=len(interp_list), unit=" frames")

                render_cnt = 0
                for interp_frame in interp_list:
                    interp_cnt = int(interp_frame[:-4])
                    interp_frame = os.path.join(self.interp_dir, interp_frame)  # realpath

                    if str(interp_cnt) in interp_scene_list:
                        pass
                    elif str(interp_cnt) in interp_scene_edge_list:
                        frames_list.append((interp_cnt, frame_ioer.read_frame(interp_frame)))
                        self.feed_to_render(frames_list, True, False)  # we should had added check point here
                        frames_list.clear()
                    else:
                        frames_list.append((interp_cnt, frame_ioer.read_frame(interp_frame)))
                        if len(frames_list) > 40:
                            self.feed_to_render(frames_list, False, False)
                            frames_list.clear()

                    rsq = self.render_info_pipe["rendering"]  # render status quo
                    """(chunk_cnt, start_frame, end_frame, frame_cnt)"""
                    pbar.set_description(
                        f"Process at Step 3, Render: Chunk {rsq[0]}")
                    pbar.set_postfix({"Frame": f"{interp_cnt}", "Render": f"{int(rsq[3])}",
                                      "Status": f"{now_frame / self.all_frames_cnt * 100:.2f}%"})
                    pbar.update(interp_cnt - render_cnt)
                    render_cnt = interp_cnt

                # the last frame is single, with no interpolated frames followed
                self.feed_to_render(frames_list, False, is_end)
                frames_list.clear()

                while not self.frames_output.empty():
                    """Wait for frames all rendered"""
                    rsq = self.render_info_pipe["rendering"]  # render status quo
                    """(chunk_cnt, start_frame, end_frame, frame_cnt)"""
                    pbar.set_description(
                        f"Process at Step 3, Render: Chunk {rsq[0]}")
                    pbar.set_postfix({"Frame": f"{render_cnt}", "Render": f"{int(rsq[3])}",
                                      "Status": f"{now_frame / self.all_frames_cnt * 100:.2f}%"})
                    time.sleep(0.1)

                """Clean out"""
                shutil.rmtree(self.input_dir)
                os.mkdir(self.input_dir)

                scenes_list.clear()
                origianl_frame_cnt = 0
                pbar.close()
                pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
                pbar.update(n=now_frame)
                pbar.unpause()

            if img1 is None:
                break

        pbar.close()
        self.logger.info(f"Scedet Status Quo: {scedet_info}")

        while self.render_thread.is_alive():
            # Assume main thread is always good
            time.sleep(0.1)

        if not self.args["no_concat"] and not self.args["img_output"]:
            self.concat_all()
            return

    def run(self):
        if self.args["concat_only"]:
            self.concat_all()
            self.logger.info(f"Program finished at {datetime.datetime.now()}")
            return
        if self.args["ncnn"]:
            self.amd_run()
        else:
            self.nvidia_run()
        self.logger.info(f"Program finished at {datetime.datetime.now()}")
        pass

    def concat_all(self):
        """
        Concat all the chunks
        :return:
        """

        os.chdir(self.project_dir)
        concat_path = os.path.join(self.project_dir, "concat.ini")
        self.logger.info("Final Round Finished, Start Concating")
        concat_list = list()

        for f in os.listdir(self.project_dir):
            chunk_re = rf"chunk-[\d+].*?\{self.output_ext}"
            if re.match(chunk_re, f):
                concat_list.append(os.path.join(self.project_dir, f))
            else:
                self.logger.debug(f"concat escape {f}")

        concat_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))  # sort as start-frame

        if os.path.exists(concat_path):
            os.remove(concat_path)

        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")

        """Check Input file ext"""
        output_ext = os.path.splitext(self.input)[-1]
        if output_ext not in [".mp4", ".mov", ".mkv"]:
            output_ext = self.output_ext
        if self.args["encoder"] == "ProRes":
            output_ext = ".mov"

        concat_filepath = f"{os.path.join(self.output, os.path.splitext(os.path.basename(self.input))[0])}_{2 ** self.exp}x" + output_ext
        if self.args["any_fps"]:
            concat_filepath = f"{os.path.join(self.output, os.path.splitext(os.path.basename(self.input))[0])}_{int(self.target_fps)}fps" + output_ext
        if self.args["save_audio"]:
            map_audio = f'-i "{self.input}" -map 0:v:0 -map 1:a? -c:a copy -shortest '
        else:
            map_audio = ""

        ffmpeg_command = f'{self.ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy {Utils.fillQuotation(concat_filepath)} -y'
        self.logger.debug(f"Concat command: {ffmpeg_command}")
        os.system(ffmpeg_command)
        if self.args["output_only"] and os.path.exists(concat_filepath):
            if not os.path.getsize(concat_filepath):
                self.logger.error(f"Concat Error, {output_ext}")
                raise FileExistsError("Concat Error, Check Output Extension!!!")
            self.check_chunk(del_chunk=True)


interpworkflow = InterpWorkFlow(args)
interpworkflow.run()
sys.exit(0)
