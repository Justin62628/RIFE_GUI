# coding: gbk
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
from pprint import pprint
import shutil
from Utils.utils import Utils, ImgSeqIO, VideoInfo, DefaultConfigParser
# 6.2.6
Utils = Utils()
"""Set Path Environment"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing working dir to {0}".format(dname))
os.chdir(os.path.dirname(dname))
print("Added {0} to temporary PATH".format(dname))
sys.path.append(dname)

parser = argparse.ArgumentParser(prog="#### RIFE Step by Step CLI tool/补帧分步设置命令行工具 from Jeanna ####",
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
# pprint(cp_items)
args = Utils.clean_parsed_config(cp_items)
args.update(vars(args_read))  # update -i -o -c
# pprint(args)
print("\n\n")

# torch env
if int(args["use_specific_gpu"]) != -1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['use_specific_gpu']}"

import inference


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

        self.ffmpeg = os.path.join(self.args["ffmpeg"], "ffmpeg.exe")
        self.ffplay = os.path.join(self.args["ffmpeg"], "ffplay.exe")
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.logger.warning("Not find selected ffmpeg, use default")

        self.input = self.args["input"]
        self.output = self.args["output"]
        self.input_dir = os.path.join(self.project_dir, 'frames')
        self.interp_dir = os.path.join(self.project_dir, 'interp')
        self.scene_dir = os.path.join(self.project_dir, 'scenes')
        env = [self.input_dir, self.interp_dir, self.scene_dir]
        Utils.make_dirs(env)

        self.exp = self.args["exp"]
        """Frame Count"""
        self.video_info_instance = VideoInfo(**self.args)
        self.video_info_instance.update_info()
        self.video_info = self.video_info_instance.get_info()
        if self.args["fps"]:
            self.fps = self.args["fps"]
        else:
            self.fps = self.video_info["fps"]
        if self.args["target_fps"]:
            self.target_fps = self.args["target_fps"]
        else:
            self.target_fps = (2 ** self.exp) * self.fps
        self.all_frames_cnt = int(self.video_info["cnt"])
        self.crop_par = [0, 0]
        crop_par = self.args["crop"]
        if crop_par not in ["", "0", None]:
            width_black, height_black = crop_par.split(":")
            width_black = int(width_black)
            height_black = int(height_black)
            self.crop_par = [width_black, height_black]
            self.logger.info(f"Update Crop Parameters to {self.crop_par}")

        self.logger.info(
            f"Check Interpolation Source, FPS: {self.fps}, TARGET FPS: {self.target_fps}, "
            f"FRAMES_CNT: {self.all_frames_cnt}, EXP: {self.exp}")

        self.rife_core = None

        self.render_gap = self.args["render_gap"]
        self.frame_reader = None
        self.render_thread = None
        self.frames_output = Queue(maxsize=int(self.render_gap * 3))

        self.render_info_pipe = {"rendering": (0, 0, 0, 0)}
        self.scene_stack_len = int(0.5 * self.fps)
        self.scene_stack = deque(maxlen=self.scene_stack_len)

        self.dup_skip_limit = int(0.5 * self.fps) + 1

        self.main_event = threading.Event()
        self.main_event.set()

        self.color_info = {}
        for k in self.video_info:
            if k.startswith("-"):
                self.color_info[k] = self.video_info[k]
        self.output_ext = "." + self.args["output_ext"]
        if self.args["encoder"] == "ProRes":
            self.output_ext = ".mov"
            # TODO Beautify this
        self.render_lock = threading.Event()
        pass

    def generate_frame_reader(self, start_frame=-1):
        if self.args["img_input"]:
            return ImgSeqIO(folder=self.input, is_read=True)
        # input_dict = {"-vsync": "0", "-to": "00:00:01"}
        input_dict = {"-vsync": "0"}
        output_dict = {
            "-vframes": str(int(abs(self.all_frames_cnt * 100)))}  # use read frames cnt to avoid ffprobe, fuck
        output_dict.update(self.color_info)
        # output_dict = {"-vframes": "0"}  # use read frames cnt to avoid ffprobe, assign to auto 0
        vf_args = "copy"

        if len(self.args["resize"]):
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": self.args["resize"].replace(":", "x").replace("*", "x")})
        if start_frame not in [-1, 0]:
            vf_args += f",trim=start_frame={start_frame}"
        if not self.args["quick_extract"]:
            vf_args += f",format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24"
        output_dict["-vf"] = vf_args
        self.logger.debug(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=self.input, inputdict=input_dict, outputdict=output_dict)

    def generate_frame_renderer(self, output_path):
        if self.args["img_output"]:
            return ImgSeqIO(folder=self.output, is_read=False)
        input_dict = {"-vsync": "cfr", "-r": f"{self.fps * 2 ** self.exp}"}

        output_dict = {"-r": f"{self.target_fps}", "-preset": self.args["preset"], "-pix_fmt": self.args["pix_fmt"]}
        output_dict.update(self.color_info)
        if self.args["slow_motion"]:
            input_dict.update({"-r": f"{self.target_fps}"})
            output_dict.pop("-r")
        vf_args = "copy"
        output_dict.update({"-vf": vf_args})

        if self.args["encoder"] == "H264":
            if self.args["hwaccel"]:
                output_dict.update({"-c:v": "h264_nvenc", "-rc:v": "vbr_hq"})
            else:
                output_dict.update({"-c:v": "libx264", "-tune": "grain"})

        elif self.args["encoder"] == "HEVC":
            if self.args["hwaccel"]:
                output_dict.update({"-c:v": "hevc_nvenc", "-rc:v": "vbr_hq"})
            else:
                # TODO check profile:v auto changed with pix_fmt
                output_dict.update(
                    {"-c:v": "libx265", "-tune": "grain"})

        elif self.args["encoder"] == "ProRes":
            output_dict.pop("-preset")  # leave space for ProRes Profile
            output_dict.update({"-c:v": "prores_ks", "-profile:v": self.args["preset"], "-quant_mat": "hq"})

        if self.args["encoder"] not in ["ProRes"]:
            if self.args["crf"] and self.args["UseCRF".lower()]:
                # TODO Check Warning
                if self.args["hwaccel"]:
                    output_dict.update({"-cq:v": str(self.args["crf"])})
                else:
                    output_dict.update({"-crf": str(self.args["crf"])})
            if self.args["UseTargetBitrate".lower()] and type(self.args["bitrate"]) == str and len(
                    self.args["bitrate"]):
                output_dict.update({"-b:v": str(self.args["bitrate"])})
        self.logger.debug(f"writer: {output_dict}, {input_dict}")
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
        chunk_re = rf"chunk-[\d+].*?\{self.output_ext}"
        for f in os.listdir(self.project_dir):
            if re.match(chunk_re, f):
                if del_chunk:
                    os.remove(os.path.join(self.project_dir, f))
                else:
                    chunk_list.append(f)
        if del_chunk:
            return 1, 0
        if self.args["interp_start"] != 0 or self.args["chunk"] != 0:
            """Manually Prioritized"""
            return int(self.args["chunk"]), int(self.args["interp_start"])
        if not len(chunk_list):
            return 1, 0
        """Remove last chunk(high possibility of dilapidation)"""
        chunk_list.sort(key=lambda x: int(x.split('-')[2]))
        os.remove(os.path.join(self.project_dir, chunk_list[-1]))
        chunk_list.pop(-1)
        if not len(chunk_list):
            return 1, 0
        self.logger.info("Found Previous Chunks")
        last_chunk = chunk_list[-1]
        chunk_re = rf"chunk-(\d+)-(\d+)-(\d+)\{self.output_ext}"
        match_result = re.findall(chunk_re, last_chunk)[0]

        chunk = int(match_result[0])
        last_frame = int(match_result[2])
        return chunk + 1, last_frame + 1

    def render(self, chunk_cnt, start_frame):
        end_frame = int(start_frame + self.render_gap / (2 ** self.exp) - 1)

        chunk_re = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, end_frame, self.output_ext)
        output_path = chunk_re
        output_path = os.path.join(self.project_dir, output_path)

        chunk_round = 1
        frame_writer = self.generate_frame_renderer(output_path)
        if not self.args["img_output"]:
            self.logger.info(f"First Chunk Render Path: {output_path}")
        while True:
            if not self.main_event.is_set():
                self.logger.warning("Main interpolation thread Dead, break")
                frame_writer.close()
                break
            frame_data = self.frames_output.get()
            if frame_data is None:
                frame_writer.close()
                break
            frame = frame_data[1]
            frame_cnt = frame_data[0]
            frame_writer.writeFrame(frame)
            chunk_round += 1
            self.render_info_pipe["rendering"] = (chunk_cnt, start_frame, end_frame, frame_cnt)
            if not chunk_round % self.render_gap:
                chunk_cnt += 1
                start_frame = end_frame + 1
                end_frame = int(start_frame + self.render_gap / (2 ** self.exp) - 1)
                output_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, end_frame,
                                                                       self.output_ext)
                output_path = os.path.join(self.project_dir, output_path)
                frame_writer.close()
                frame_writer = self.generate_frame_renderer(output_path)

            pass

    def feed_to_render(self, frames_list, is_scene=False, is_end=False):
        frames_list_len = len(frames_list)
        if not len(frames_list) and is_end:
            self.frames_output.put(None)
            self.logger.info("Put None to write_buffer")
        for frame_i in range(frames_list_len):
            self.frames_output.put(frames_list[frame_i])
            if frame_i == frames_list_len - 1:
                if is_scene:
                    for put_i in range(2 ** self.exp - 1):
                        self.frames_output.put(frames_list[frame_i])
                    return
                if is_end:
                    self.frames_output.put(None)
                    self.logger.info("Put None to write_buffer")
                    return
        pass

    def check_scene(self, diff, add_diff=False) -> bool:
        """
        Check if current scene is scene
        :param diff:
        :param add_diff:
        :return:
        """
        if add_diff:
            self.scene_stack.append(diff)
            return False
        if len(self.scene_stack) < self.scene_stack_len:
            self.scene_stack.append(diff)
            return False
        before_measure = np.var(self.scene_stack)
        self.scene_stack.append(diff)
        after_measure = np.var(self.scene_stack)
        if abs(after_measure) - abs(before_measure) > self.args["scdet_threshold"]:
            """Detect new scene"""
            self.scene_stack.clear()
            return True
        else:
            self.scene_stack.append(diff)
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
        if self.crop_par[0] > w or self.crop_par[1] > h:
            return img
        return img[self.crop_par[1]:h - self.crop_par[1], self.crop_par[0]:w - self.crop_par[0]]

    def nvidia_run(self):
        """
        Go through all procedures to produce interpolation result
        :return:
        """
        self.rife_core = inference.RifeInterpolation(self.args)
        self.rife_core.initiate_rife()

        _debug = False
        self.rife_core.initiate_rife(args)
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
            raise OSError(f"Input file not valid: {self.input}")

        is_end = False
        pbar = tqdm.tqdm(total=self.all_frames_cnt)
        pbar.update(n=start_frame)
        slot_img = None
        recent_scene = 0
        previous_cnt = now_frame
        scedet_info = {"0": 0, "1": 0, "1+": 0}

        while True:
            if is_end:
                break
            img0 = img1
            frames_list = []
            if self.args["remove_dup"]:

                img1 = self.crop_read_img(Utils.gen_next(videogen))

                if img1 is None:
                    frames_list.append([now_frame, img0])
                    self.feed_to_render(frames_list, is_end=True)
                    break

                gap_frames_list = []  # 用于去除重复帧的临时帧列表
                gap_frames_list.append(img0)  # 放入当前判断区间头一帧
                diff = cv2.absdiff(img0, img1).mean()
                skip = 0  # 用于记录跳过的帧数
                is_scene = False
                if self.check_scene(diff):
                    """!!!scene"""
                    # 入乡随俗
                    frames_list.append([now_frame, img0])
                    self.feed_to_render(frames_list, is_scene=True)
                    recent_scene = now_frame
                    now_frame += 1  # to next frame img0 = img1
                    scedet_info["0"] += 1
                    continue
                else:
                    # No scene
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
                            self.check_scene(diff, add_diff=True)  # update
                            if skip == self.dup_skip_limit:
                                break

                        # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                        if self.check_scene(diff):
                            # 入乡随俗
                            skip -= 1
                            if not skip:
                                gap_frames_list.append(img0)
                            else:
                                exp = int(math.log(skip, 2)) + 1
                                interp_output = self.rife_core.generate_interp(img0, before_img, exp, self.args["scale"])
                                kpl = Utils.generate_prebuild_map(exp, skip)
                                for x in kpl:
                                    gap_frames_list.append(interp_output[x])
                                gap_frames_list.append(before_img)
                            is_scene = True
                            scedet_info["0"] += 1
                        elif skip != 0:
                            # 推导 exp
                            exp = int(math.log(skip, 2)) + 1
                            interp_output = self.rife_core.generate_interp(img0, img1, exp, self.args["scale"])
                            kpl = Utils.generate_prebuild_map(exp, skip)
                            for x in kpl:
                                gap_frames_list.append(interp_output[x])
                    else:
                        scedet_info["1"] += 1

                # post进行到正常补帧序列（可修改）
                if not is_scene:
                    gap_frames_list.append(img1)
                # now_frame += 1
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
                interp_output = self.rife_core.generate_interp(img0, img1, self.exp, self.args["scale"])
                for mid_index in range(len(interp_output)):
                    frames_list.append([now_frame, interp_output[mid_index]])
                self.feed_to_render(frames_list)
                now_frame += 1  # to next frame img0 = img1
                scedet_info["1"] += 1

            rsq = self.render_info_pipe["rendering"]  # render status quo
            """(chunk_cnt, start_frame, end_frame, frame_cnt)"""
            pbar.set_description(
                f"Process at Chunk {rsq[0]:0>3d}, RenderedFrame {rsq[3]:0>6d}, CurrentFrame {now_frame:0>6d}, NearScene {recent_scene:0>6d}")
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame
        pbar.close()
        self.logger.info(f"Scedet Status Quo: {scedet_info}")
        while self.render_thread.is_alive():
            time.sleep(0.1)

        if not self.args["no_concat"] and not self.args["img_output"]:
            self.concat_all()
            return

    def amd_whole_run(self):
        """
        fuck
        whole shots
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
        pbar = tqdm.tqdm(total=self.all_frames_cnt)
        pbar.update(n=start_frame)
        recent_scene = 0
        scedet_info = {"0": 0, "1": 0, "1+": 0}
        scenes_list = []
        previous_cnt = now_frame
        origianl_frame_cnt = 0  # to mark relation between original frame and interp frame
        frame_ioer = ImgSeqIO(self.input_dir, is_tool=True)

        """Render on its own, fuck"""
        interp_video_path = f"{os.path.splitext(self.input)[0]}_{2 ** self.exp}x_video" + self.output_ext
        video_renderer = self.generate_frame_renderer(interp_video_path)
        shutil.rmtree(self.input_dir)
        os.mkdir(self.input_dir)
        while True:
            img0 = img1
            img1 = self.crop_read_img(Utils.gen_next(videogen))

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
            pbar.set_description(
                f"Process at Step 1: Extract Frames： Current {now_frame:0>6d}, Scene {recent_scene:0>6d}")
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame

            if img1 is None:
                """Time for interpolation, once enter, no out"""
                self.rife_core = inference.NCNNinterpolator(self.args)
                self.rife_core.initiate_rife()
                self.rife_core.start()
                interp_scene_list = {}
                interp_scene_edge_list = {}

                now_dir = ""  # current interpolation dir
                now_cnt = 0  # for interp and scene positioning
                while self.rife_core.is_alive():
                    if "now_dir" not in self.rife_core.supervise_data:
                        "NCNN initializing"
                        time.sleep(0.1)
                        continue
                    if now_dir != self.rife_core.supervise_data["now_dir"]:
                        pbar.close()
                        pbar = tqdm.tqdm(total=self.rife_core.supervise_data["input_cnt"])
                        now_dir = self.rife_core.supervise_data["now_dir"]
                        now_cnt = 0
                    time.sleep(0.1)
                    pbar.update(self.rife_core.supervise_data["now_cnt"] - now_cnt)
                    now_cnt = self.rife_core.supervise_data["now_cnt"]
                    pbar.set_description(
                        f"Process at Step 2: Interpolation No. {self.rife_core.supervise_data['now_dir']} Round")

                # generate scene map info
                for scene in scenes_list:
                    interp_scene = scene * 2 ** self.exp + 1
                    for i in range(1, 2 ** self.exp):
                        interp_scene_list[str(interp_scene + i)] = True
                    interp_scene_edge_list[str(interp_scene)] = True

                """Final interpolation is done, final render"""
                interp_list = sorted(os.listdir(self.interp_dir), key=lambda x: x[:-4])
                pbar.close()
                pbar = tqdm.tqdm(total=len(interp_list))
                for frame_path in interp_list:
                    interp_cnt = int(frame_path[:-4])
                    frame_path = os.path.join(self.interp_dir, frame_path)  # realpath
                    if str(interp_cnt) in interp_scene_list:
                        pass
                    elif str(interp_cnt) in interp_scene_edge_list:
                        for i in range(2 ** self.exp):
                            video_renderer.writeFrame(frame_ioer.read_frame(frame_path))
                    else:
                        video_renderer.writeFrame(frame_ioer.read_frame(frame_path))
                    pbar.set_description(
                        f"Process at Step 3: RenderFrame {interp_cnt:0>6d}")
                    pbar.update(1)

                video_renderer.close()
                if self.args["output_only"]:
                    shutil.rmtree(self.input_dir)
                    os.mkdir(self.input_dir)
                break

        pbar.close()
        self.logger.info(f"Scedet Status Quo: {scedet_info}")
        if not self.args["img_output"]:
            shutil.rmtree(self.interp_dir)
        if not self.args["no_concat"] and not self.args["img_output"]:
            if self.args["save_audio"]:
                map_audio = f'-i "{self.input}" -map 0:v:0 -map 1:a:0 -c:a copy -shortest '
            else:
                map_audio = ""
            output_video_path = f"{os.path.splitext(self.input)[0]}_{2 ** self.exp}x" + self.output_ext
            ffmpeg_command = f'{self.ffmpeg} -hide_banner -i "{interp_video_path}" {map_audio} -c:v copy {output_video_path} -y'
            self.logger.debug(f"Concat command: {ffmpeg_command}")
            os.system(ffmpeg_command)
            return

    def amd_run(self):
        """
        LMAO
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
        pbar = tqdm.tqdm(total=self.all_frames_cnt)
        pbar.update(n=start_frame)
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
                f"Process at Step 1: Chunk {rsq[0]}, Extract {now_frame:0>8d}, Scene {recent_scene:0>8d}")
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
                        pbar = tqdm.tqdm(total=self.rife_core.supervise_data["input_cnt"])
                        now_dir = self.rife_core.supervise_data["now_dir"]
                        now_cnt = 0
                    time.sleep(0.1)
                    pbar.update(self.rife_core.supervise_data["now_cnt"] - now_cnt)
                    now_cnt = self.rife_core.supervise_data["now_cnt"]
                    rsq = self.render_info_pipe["rendering"]  # render status quo
                    pbar.set_description(
                        f"Process at Step 2: Chunk {rsq[0]}, No. {self.rife_core.supervise_data['now_dir']} Round")

                # generate scene map info
                for scene in scenes_list:
                    interp_scene = scene * 2 ** self.exp + 1
                    for i in range(1, 2 ** self.exp):
                        interp_scene_list[str(interp_scene + i)] = True
                    interp_scene_edge_list[str(interp_scene)] = True

                """Final interpolation is done, final render"""
                frames_list = []
                interp_list = sorted(os.listdir(self.interp_dir), key=lambda x: x[:-4])
                pbar.close()
                pbar = tqdm.tqdm(total=len(interp_list))
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
                        f"Process at Step 3: Chunk {rsq[0]}, InputFrame {interp_cnt:0>6d}, RenderFrame {int(rsq[3]):0>6d}")
                    pbar.update(interp_cnt - render_cnt)
                    render_cnt = interp_cnt

                # the last frame is single, with no interpolated frames followed
                self.feed_to_render(frames_list, False, is_end)
                frames_list.clear()

                while not self.frames_output.empty():
                    """Wait for frames all rendered"""
                    time.sleep(0.1)
                shutil.rmtree(self.input_dir)
                os.mkdir(self.input_dir)

                scenes_list.clear()
                origianl_frame_cnt = 0
                pbar.close()
                pbar = tqdm.tqdm(total=self.all_frames_cnt)
                pbar.update(n=now_frame)
                pass
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

        # TODO Beautify ProRes Special Judge
        if output_ext not in [".mp4", ".mov", ".mkv"]:
            output_ext = self.output_ext
        if self.args["encoder"] == "ProRes":
            output_ext = ".mov"

        concat_filepath = f"{os.path.join(self.output, os.path.splitext(os.path.basename(self.input))[0])}_{2 ** self.exp}x" + output_ext
        if self.args["save_audio"]:
            map_audio = f'-i "{self.input}" -map 0:v:0 -map 1:a:0 -c:a copy -shortest '
        else:
            map_audio = ""
        ffmpeg_command = f'{self.ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy {Utils.fillQuotation(concat_filepath)} -y'
        self.logger.debug(f"Concat command: {ffmpeg_command}")
        os.system(ffmpeg_command)
        if self.args["output_only"] and os.path.exists(concat_filepath):
            if not os.path.getsize(concat_filepath):
                raise FileExistsError("Concat Error, Check Output Extension!!!")
            self.check_chunk(del_chunk=True)


interpworkflow = InterpWorkFlow(args)
interpworkflow.run()
sys.exit(0)
