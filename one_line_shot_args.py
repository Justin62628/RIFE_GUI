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
import shutil
import traceback
import psutil
from Utils.utils import Utils, ImgSeqIO, VideoInfo, DefaultConfigParser, TransitionDetection

# 6.2.11 2021/4/22
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
        self.env = [self.input_dir, self.interp_dir, self.scene_dir]
        Utils.make_dirs(self.env)

        """Load Interpolation Exp"""
        self.exp = self.args["exp"]

        """Get input's info"""
        self.video_info_instance = VideoInfo(**self.args)
        self.video_info = self.video_info_instance.get_info()
        if self.args["batch"] and not self.args["img_input"]:  # 检测到批处理，且输入不是文件夹，使用检测到的帧率
            self.fps = self.video_info["fps"]
        elif self.args["fps"]:
            self.fps = self.args["fps"]
        else:  # 用户有毒，未发现有效的输入帧率，用检测到的帧率
            if self.video_info["fps"] is None or not self.video_info["fps"]:
                raise OSError("Input File not valid")
            self.fps = self.video_info["fps"]

        if self.args["img_input"]:
            self.args["any_fps"] = False  # when img input is detected, any_fps mode is disabled
            self.target_fps = self.args["target_fps"]
            # but assigned output fps will be not touched
        else:
            if self.args["target_fps"]:
                self.target_fps = self.args["target_fps"]
                if abs(self.args["target_fps"] - 2 ** self.exp * self.fps) > 1e-3:
                    """Activate Any FPS Mode"""
                    self.logger.info(
                        f"Activate Any FPS Mode: fps: {self.fps:.2f} -> target_fps: {self.args['target_fps']:.2f}")
                    self.args["any_fps"] = True
            else:
                self.target_fps = (2 ** self.exp) * self.fps

        """Update All Frames Count"""
        self.all_frames_cnt = int(self.video_info["cnt"])  # 视频总帧数
        if self.args["any_fps"]:  # 使用任意帧率，要相应更新总帧数（图片序列输入是不可能的）
            self.all_frames_cnt = int(self.video_info["duration"] * self.target_fps)

        """Crop Video"""
        self.crop_param = [0, 0]  # crop parameter, 裁切参数
        crop_param = self.args["crop"].replace("：", ":")
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

        """RIFE Core"""
        self.rife_core = None  # 用于补帧的模块

        """Guess Memory and Render System"""
        if self.args["manual_buffer"]:
            free_mem = self.args["manual_buffer"] * 1024
        else:
            mem = psutil.virtual_memory()
            free_mem = round(mem.free / 1024 / 1024)
        self.frames_output_size = round(free_mem / (sys.getsizeof(
            np.random.rand(3, round(self.video_info["size"][0]),
                           round(self.video_info["size"][1]))) / 1024 / 1024) * 0.8)
        if self.frames_output_size < 100:
            self.frames_output_size = 100
        self.logger.info(f"Buffer Size to {self.frames_output_size}")
        self.frames_output = Queue(maxsize=self.frames_output_size)  # 补出来的帧序列队列（消费者）
        self.render_gap = self.args["render_gap"]  # 每个chunk的帧数
        self.frame_reader = None  # 读帧的迭代器／帧生成器
        self.render_thread = None  # 帧渲染器
        self.render_info_pipe = {"rendering": (0, 0, 0, 0)}  # 有关渲染的实时信息，目前只有一个key

        """Scene Detection"""
        self.scene_detection = TransitionDetection(int(0.3 * self.fps), **self.args)

        """Duplicate Frames Removal"""
        self.dup_skip_limit = int(0.5 * self.fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环

        """Main Thread Lock"""
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

        self.main_error = None
        pass

    def generate_frame_reader(self, start_frame=-1):
        """
        输入帧迭代器
        :param start_frame:
        :return:
        """
        """If input is sequence of frames"""
        if self.args["img_input"]:
            return ImgSeqIO(folder=self.input, is_read=True, start_frame=start_frame, **self.args)

        """If input is a video"""
        input_dict = {"-vsync": "0", "-hwaccel": "auto"}
        if self.args.get("start_point", None) is not None or self.args.get("end_point", None) is not None:
            time_fmt = "%H:%M:%S"
            start_point = datetime.datetime.strptime(self.args["start_point"], time_fmt)
            end_point = datetime.datetime.strptime(self.args["end_point"], time_fmt)
            if end_point > start_point:
                input_dict.update({"-ss": self.args['start_point'], "-to": self.args['end_point']})
                start_frame = -1
                clip_duration = end_point - start_point
                clip_fps = self.fps if not self.args["any_fps"] else self.target_fps
                self.all_frames_cnt = round(clip_duration.total_seconds() * clip_fps)
                self.logger.info(
                    f"Update Input Range: in {self.args['start_point']} -> out {self.args['end_point']}, all_frames_cnt -> {self.all_frames_cnt}")
            else:
                self.logger.warning(f"Bad Input Time Section: {start_point} -> {end_point}, change to origianl course")

        output_dict = {
            "-vframes": str(int(abs(self.all_frames_cnt * 100)))}  # use read frames cnt to avoid ffprobe, fuck

        output_dict.update(self.color_info)
        output_dict.update({"-r": f"{self.fps}"})  # TODO Check Danger

        vf_args = "copy"

        """任意帧率输出基础支持"""
        if self.args["any_fps"]:
            vf_args += f",minterpolate=fps={self.target_fps}:mi_mode=dup"
            output_dict.pop("-r")

        if len(self.args["resize"]):
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": self.args["resize"].replace(":", "x").replace("*", "x")})

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
        if self.args["any_fps"] and not self.args["ncnn"] and not self.args[
            "img_input"]:  # TODO: Img Seq supports any fps
            input_dict.update({"-r": f"{self.target_fps}"})
        output_dict.update(self.color_info)

        """Slow motion design"""
        if self.args["slow_motion"]:
            if self.args.get("slow_motion_fps", 0):
                input_dict.update({"-r": f"{self.args['slow_motion_fps']}"})
            else:
                input_dict.update({"-r": f"{self.fps}"})
            output_dict.pop("-r")
        vf_args = "copy"  # debug
        output_dict.update({"-vf": vf_args})

        """Assign Render Codec"""
        if self.args["encoder"] == "H264/AVC":
            if self.args["hwaccel_mode"] != "None":
                hwaccel_mode = self.args["hwaccel_mode"]
                if hwaccel_mode == "NVENC":
                    output_dict.update({"-c:v": "h264_nvenc", "-rc:v": "vbr_hq",
                                        f"-g": f"{int(self.target_fps * 3)}", "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                        f"-rc-lookahead": "120", "-no-scenecut": "1", "-forced-idr": "1",
                                        f"-spatial-aq": "1", "-temporal-aq": "1", "-strict_gop": "1",
                                        "-b_ref_mode": "2", })
                elif hwaccel_mode == "QSV":
                    output_dict.update({"-c:v": "h264_qsv",
                                        f"-g": f"{int(self.target_fps * 3)}", "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                        f"-rc-lookahead": "120", })
            else:
                output_dict.update({"-c:v": "libx264", })  # TODO H264 Encode Sets
        elif self.args["encoder"] == "H265/HEVC":
            if self.args["hwaccel_mode"] != "None":
                hwaccel_mode = self.args["hwaccel_mode"]
                if hwaccel_mode == "NVENC":
                    output_dict.update({"-c:v": "hevc_nvenc", "-rc:v": "vbr_hq",
                                        f"-g": f"{int(self.target_fps * 3)}", "-bf": "5", "-i_qfactor": "0.75",
                                        "-b_qfactor": "1.1",
                                        f"-rc-lookahead": "120", "-no-scenecut": "1", "-forced-idr": "1",
                                        f"-spatial-aq": "1", "-temporal-aq": "1", "-nonref_p": "1", "-strict_gop": "1",
                                        "-b_ref_mode": "2"})
                elif hwaccel_mode == "QSV":
                    output_dict.update({"-c:v": "hevc_qsv",
                                        f"-g": f"{int(self.target_fps * 3)}", "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                        f"-look_ahead": "120", })
            else:
                output_dict.update(
                    {"-c:v": "libx265",
                     "-x265-params": "ref=4:rd=3:no-rect=1:no-amp=1:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:"
                                     "weightb=1:no-strong-intra-smoothing=1:psy-rd=2.0:psy-rdoq=1.0:no-open-gop=1:"
                                     f"keyint={int(self.target_fps * 3)}:min-keyint=1:rc-lookahead=120:bframes=6:"
                                     f"aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:"
                                     f"deblock=-1:no-sao=1"})
        elif self.args["encoder"] == "ProRes":
            output_dict.pop("-preset")  # leave space for ProRes Profile
            output_dict.update({"-c:v": "prores_ks", "-profile:v": self.args["preset"], "-quant_mat": "hq"})

        if self.args["encoder"] not in ["ProRes"]:
            if self.args["crf"] and self.args["UseCRF".lower()]:
                if self.args["hwaccel_mode"] != "None":
                    hwaccel_mode = self.args["hwaccel_mode"]
                    if hwaccel_mode == "NVENC":
                        output_dict.update({"-cq:v": str(self.args["crf"])})
                    elif hwaccel_mode == "QSV":
                        output_dict.update({"-q": str(self.args["crf"])})
                else:
                    output_dict.update({"-crf": str(self.args["crf"])})
            if self.args["UseTargetBitrate".lower()] and self.args["bitrate"]:
                output_dict.update({"-b:v": f'{self.args["bitrate"]}M'})
                if self.args["hwaccel_mode"] == "QSV":
                    output_dict.update({"-maxrate": "200M"})

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

        self.logger.info("Found Previous Chunks")
        last_chunk = chunk_list[-1]  # select last chunk to assign start frames
        chunk_regex = rf"chunk-(\d+)-(\d+)-(\d+)\{self.output_ext}"
        match_result = re.findall(chunk_regex, last_chunk)[0]

        chunk = int(match_result[0])
        last_frame = int(match_result[2])
        return chunk + 1, last_frame + 1

    def render(self, chunk_cnt, render_start):
        """
        Render thread
        :param chunk_cnt:
        :param render_start:
        :return:
        """

        def rename_chunk():
            chunk_desc_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, render_start, now_frame,
                                                                       self.output_ext)
            chunk_desc_path = os.path.join(self.project_dir, chunk_desc_path)
            if os.path.exists(chunk_desc_path):
                os.remove(chunk_desc_path)
            os.rename(chunk_tmp_path, chunk_desc_path)

        def check_audio_concat():
            if not self.args["save_audio"]:
                return
            """Check Input file ext"""
            output_ext = os.path.splitext(self.input)[-1]
            if output_ext not in [".mp4", ".mov", ".mkv"]:
                output_ext = self.output_ext
            if self.args["encoder"] == "ProRes":
                output_ext = ".mov"

            concat_filepath = f"{os.path.join(self.output, 'concat_test')}" + output_ext
            map_audio = f'-i "{self.input}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy -shortest '
            ffmpeg_command = f'{self.ffmpeg} -hide_banner -i "{chunk_tmp_path}" {map_audio} -c:v copy {Utils.fillQuotation(concat_filepath)} -y'
            self.logger.info("Start Audio Concat Test")
            os.system(ffmpeg_command)
            if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath):
                self.logger.error(f"Concat Test Error, {output_ext}, empty output")
                self.main_error = FileExistsError("Concat Test Error, empty output, Check Output Extension!!!")
                raise FileExistsError("Concat Test Error, empty output, Check Output Extension!!!")
            self.logger.info("Audio Concat Test Success")
            os.remove(concat_filepath)

        concat_test_flag = True

        chunk_frame_cnt = 1  # number of frames of current output chunk
        chunk_tmp_path = os.path.join(self.project_dir, f"chunk-tmp{self.output_ext}")
        frame_writer = self.generate_frame_renderer(chunk_tmp_path)  # get frame renderer

        now_frame = render_start
        while True:
            if not self.main_event.is_set():
                self.logger.warning("Main interpolation thread Dead, break")  # 主线程已结束，这里的锁其实没用，调试用的
                frame_writer.close()
                rename_chunk()
                break

            frame_data = self.frames_output.get()
            if frame_data is None:
                frame_writer.close()
                if not self.args["img_output"]:
                    rename_chunk()
                break

            frame = frame_data[1]
            now_frame = frame_data[0]
            frame_writer.writeFrame(frame)

            chunk_frame_cnt += 1
            self.render_info_pipe["rendering"] = (chunk_cnt, render_start, now_frame, now_frame)  # update render info

            if not chunk_frame_cnt % self.render_gap:
                frame_writer.close()
                if concat_test_flag:
                    check_audio_concat()
                    concat_test_flag = False
                rename_chunk()
                chunk_cnt += 1
                render_start = now_frame + 1
                frame_writer = self.generate_frame_renderer(chunk_tmp_path)
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

    def nvidia_vram_test(self, img):
        try:
            if len(self.args["resize"]):
                w, h = list(map(lambda x: int(x), self.args["resize"].split("x")))
            else:
                h, w, _ = list(map(lambda x: round(x), img.shape))

            if w * h > 1920 * 1080:
                if self.args["scale"] > 0.5:
                    self.args["scale"] = 0.5
                    self.logger.warning(f"Big Resolution (>1080p) Input found: Reset Scale to {self.args['scale']}")

            self.logger.info(f"Start VRAM Test: {w}x{h} with scale {self.args['scale']}")

            test_img0, test_img1 = np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8), \
                                   np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8)
            self.rife_core.generate_interp(test_img0, test_img1, 1, self.args["scale"])
            self.logger.info(f"VRAM Test Success")
        except Exception as e:
            self.logger.error("VRAM Check Failed, PLS Lower your presets\n" + traceback.format_exc())
            raise e

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
        self.logger.info("Resuming Video Frames...")
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

        """VRAM Test"""
        self.nvidia_vram_test(img1)

        is_end = False
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
        pbar.update(n=start_frame)
        pbar.unpause()
        recent_scene = 0  # 最近的转场
        previous_cnt = now_frame  # 上一帧的计数
        scedet_info = {"0": 0, "1": 0, "1+": 0}  # 帧种类，0为转场，1为正常帧，1+为重复帧，即两帧之间的计数关系

        extract_cnt = 0

        """Update Mode Info"""
        if self.args["any_fps"]:
            self.args["dup_threshold"] = self.args["dup_threshold"] if self.args["dup_threshold"] > 0.01 else 0.01

        while True:
            if is_end or self.main_error:
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
                if self.scene_detection.check_scene(img0, img1):
                    self.feed_to_render(frames_list)  # no need to update scene flag
                    recent_scene = now_frame
                    scedet_info["0"] += 1
                    continue
                else:
                    if diff < self.args["dup_threshold"]:
                        before_img = img1
                        valid_skip = 0
                        while diff < self.args["dup_threshold"]:
                            skip += 1
                            valid_skip += 1
                            scedet_info["1+"] += 1
                            img1 = self.crop_read_img(Utils.gen_next(videogen))
                            extract_cnt += 1

                            if img1 is None:
                                img1 = before_img
                                is_end = True
                                break

                            diff = cv2.absdiff(img0, img1).mean()

                            self.scene_detection.check_scene(img0, img1, add_diff=True)  # update scene stack
                            if diff > 0.1 and valid_skip == int(self.dup_skip_limit * self.target_fps / self.fps):
                                """超过重复帧计数限额，直接跳出"""
                                break

                        # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                        if self.scene_detection.check_scene(img0, img1, no_diff=True):
                            skip -= 1  # 两帧间隔计数器-1
                            if skip:
                                interp_output = self.rife_core.generate_interp(img0, before_img, 0,
                                                                               self.args["scale"], n=skip, debug=_debug)
                                for x in interp_output:
                                    frames_list.append([now_frame, x])
                            frames_list.append([now_frame, before_img])
                            recent_scene = now_frame
                            scedet_info["0"] += 1
                            now_frame += skip + 1

                        elif skip != 0:
                            interp_output = self.rife_core.generate_interp(img0, img1, 0, self.args["scale"], n=skip,
                                                                           debug=_debug)
                            for x in interp_output:
                                frames_list.append([now_frame, x])
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
                if self.scene_detection.check_scene(img0, img1):
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
                            self.scene_detection.check_scene(img0, img1, add_diff=True)  # update scene stack
                            if skip == self.dup_skip_limit:
                                """超过重复帧计数限额，直接跳出"""
                                break

                        # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                        if self.scene_detection.check_scene(img0, img1, no_diff=True):
                            skip -= 1  # 两帧间隔计数器-1
                            if not skip:
                                gap_frames_list.append(img0)
                            else:
                                interp_output = self.rife_core.generate_interp(img0, img1, 0, self.args["scale"],
                                                                               n=skip, debug=_debug)
                                for x in interp_output:
                                    gap_frames_list.append(x)
                                gap_frames_list.append(before_img)

                            is_scene = True
                            scedet_info["0"] += 1

                        elif skip != 0:
                            interp_output = self.rife_core.generate_interp(img0, img1, 0, self.args["scale"], n=skip,
                                                                           debug=_debug)
                            for x in interp_output:
                                gap_frames_list.append(x)
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

                if self.scene_detection.check_scene(img0, img1):
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
            pbar.set_postfix({"Render": f"{rsq[3]}", "Current": f"{now_frame}", "Scene": f"{recent_scene}",
                              "SceneCnt": f"{self.scene_detection.scdet_cnt}"})
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
            pbar.set_postfix({"Render": f"{rsq[3]}", "Current": f"{now_frame}", "Scene": f"{recent_scene}",
                              "SceneCnt": f"{self.scene_detection.scdet_cnt}"})
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame
            time.sleep(0.1)
        pbar.close()

        """Check Finished Safely"""
        if self.main_error is not None:
            raise self.main_error

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
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
        pbar.update(n=start_frame)
        pbar.unpause()
        recent_scene = 0
        scedet_info = {"0": 0, "1": 0, "1+": 0}
        scenes_list = []
        previous_cnt = now_frame
        origianl_frame_cnt = 0  # to mark relation between original frame and interp frame
        frame_ioer = ImgSeqIO(self.input_dir, is_tool=True)

        while True:
            if is_end or self.main_error is not None:
                break
            img0 = img1
            img1 = self.crop_read_img(Utils.gen_next(videogen))
            if img1 is None:
                is_end = True

            if self.scene_detection.check_scene(img0, img1) and img1 is not None:
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
            pbar.set_postfix(
                {"Frame": f"{now_frame}", "Scene": f"{recent_scene}", "SceneCnt": f"{self.scene_detection.scdet_cnt}"})
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
                        pbar = tqdm.tqdm(total=self.rife_core.supervise_data["input_cnt"], unit="frames")
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
                    if self.main_error is not None:
                        raise self.main_error

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
                pbar = tqdm.tqdm(total=len(interp_list), unit="frames")

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
                    if self.main_error is not None:
                        raise self.main_error

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
                    if self.main_error is not None:
                        raise self.main_error

                """Clean out"""
                shutil.rmtree(self.input_dir)
                os.mkdir(self.input_dir)

                scenes_list.clear()
                origianl_frame_cnt = 0
                pbar.close()
                pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
                pbar.update(n=now_frame)
                pbar.unpause()
                if self.main_error is not None:
                    raise self.main_error

            if img1 is None:
                break

        pbar.close()
        self.logger.info(f"Scedet Status Quo: {scedet_info}")

        """Check Finished Safely"""
        if self.main_error is not None:
            raise self.main_error

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

        input_filenames = os.path.splitext(os.path.basename(self.input))

        concat_filepath = f"{os.path.join(self.output, input_filenames[0])}"
        if self.args["any_fps"]:
            concat_filepath += f"_{int(self.target_fps)}fps"
        else:
            concat_filepath += f"_{2 ** self.exp}x"
        if self.args["slow_motion"]:
            concat_filepath += f"_slowmo_{self.args['slow_motion_fps']}"
        concat_filepath += output_ext

        if self.args["save_audio"]:
            audio_path = self.input
            map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy -shortest '
            if self.args.get("start_point", None) is not None or self.args.get("end_point", None) is not None:
                time_fmt = "%H:%M:%S"
                start_point = datetime.datetime.strptime(self.args["start_point"], time_fmt)
                end_point = datetime.datetime.strptime(self.args["end_point"], time_fmt)
                if end_point > start_point:
                    self.logger.info(
                        f"Update Concat Audio Range: in {self.args['start_point']} -> out {self.args['end_point']}")
                    map_audio = f'-ss {self.args["start_point"]} -to {self.args["end_point"]} -i "{audio_path}" -map 0:v:0 -map 1:a? -c:a aac -ab 640k '
                else:
                    self.logger.warning(
                        f"Bad Input Time Section: {start_point} -> {end_point}, change to origianl course")

        else:
            map_audio = ""

        ffmpeg_command = f'{self.ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy {Utils.fillQuotation(concat_filepath)} -y'
        self.logger.debug(f"Concat command: {ffmpeg_command}")
        os.system(ffmpeg_command)
        if self.args["output_only"] and os.path.exists(concat_filepath):
            if not os.path.getsize(concat_filepath):
                self.logger.error(f"Concat Error, {output_ext}, empty output")
                raise FileExistsError("Concat Error, empty output, Check Output Extension!!!")
            self.check_chunk(del_chunk=True)
            Utils.make_dirs(self.env, rm=True)

    def concat_check(self, concat_list, concat_filepath):
        """
        Check if concat output is valid
        :param concat_filepath:
        :param concat_list:
        :return:
        """
        original_concat_size = 0
        for f in concat_list:
            original_concat_size += os.path.getsize(f)
        output_concat_size = os.path.getsize(concat_filepath)
        if output_concat_size < original_concat_size * 0.9:
            return False
        return True


interpworkflow = InterpWorkFlow(args)
interpworkflow.run()
sys.exit(0)
