# coding: utf-8
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

import cv2
import numpy as np
import tqdm
from skvideo.io import ffprobe, FFmpegWriter, FFmpegReader

import inference
import utils

Utils = utils.Utils()

"""Set Path Environment"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing working dir to {0}".format(dname))
os.chdir(os.path.dirname(dname))
print("Added {0} to temporary PATH".format(dname))
sys.path.append(dname)

parser = argparse.ArgumentParser(prog="#### RIFE Step by Step CLI tool/补帧分步设置命令行工具 from Jeanna ####",
                                 description='Interpolation for sequences of images')
stage1_parser = parser.add_argument_group(title="Basic Settings, Necessary")
stage1_parser.add_argument('-i', '--input', dest='input', type=str, default=None, required=True,
                           help="原视频/图片序列文件夹路径")
stage1_parser.add_argument('-o', '--output', dest='output', type=str, default=None, required=True,
                           help="成品输出的路径，注意默认在项目文件夹")
stage1_parser.add_argument('--img-input', dest='img_input', action='store_true', help='输入的是图片序列，支持png、jpg、jpeg')

stage1_parser.add_argument('--ffmpeg', dest='ffmpeg', type=str, default=dname,
                           help="ffmpeg三件套所在文件夹, 默认当前文件夹：%(default)s")
stage1_parser.add_argument('--fps', dest='fps', type=float, default=0,
                           help="原视频的帧率, 默认0(自动识别)")
stage1_parser.add_argument('--target-fps', dest='target_fps', type=float, default=0,
                           help="目标视频帧率, 默认0(fps * 2 ** exp)")
stage2_parser = parser.add_argument_group(title="Step by Step Settings")
stage2_parser.add_argument('-r', '--ratio', dest='ratio', type=int, choices=range(1, 4), default=2, required=True,
                           help="补帧系数, 2的几次方，23.976->95.904，填2")
stage2_parser.add_argument('--chunk', dest='chunk', type=int, default=1, help="新增视频的序号(auto)")
stage2_parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=1000,
                           help="每一个chunk包含的帧数量, 默认: %(default)s")
stage2_parser.add_argument('--interp-start', dest='interp_start', type=int, default=0,
                           help="用于补帧的原视频的帧序列起始帧数，默认：%(default)s")
stage2_parser.add_argument('--interp-cnt', dest='interp_cnt', type=int, default=1, help="成品帧序列起始帧数")
stage2_parser.add_argument('--hwaccel', dest='hwaccel', action='store_true', help='支持硬件加速编码(想搞快点就用上)')
stage2_parser.add_argument('--fp16', dest='fp16', action='store_true', help='支持fp16以精度稍低、占用显存更少的操作完成补帧')
stage2_parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='补帧精度倍数，越大越精确、越慢，4K推荐0.5或1.0')
stage2_parser.add_argument('--model', dest='model', type=str, default="", help="Select RIFE Model, default v2")
# TODO module selector
stage3_parser = parser.add_argument_group(title="Preference Settings")
stage3_parser_hardware_set = stage3_parser.add_mutually_exclusive_group()
# stage3_parser_step = stage3_parser.add_argument_group()
stage3_parser.add_argument('--img-output', dest='img_output', action='store_true', help='输出的是图片序列')
stage3_parser.add_argument('--HDR', dest='HDR', action='store_true', help='支持HDR补帧')
stage3_parser_hardware_set.add_argument('--use-gpu', dest='use_specific_gpu', type=int, default=0, help='指定GPU编号，从0开始')
stage3_parser_hardware_set.add_argument('--multicard', dest='use_multi_card', action='store_true', help='N卡多卡补帧，暂不支持')
stage3_parser_hardware_set.add_argument('--ncnn', dest='ncnn', action='store_true', help='NCNN补帧')
stage3_parser_hardware_set.add_argument('--cpu', dest='use_cpu', action='store_true', help='CPU补帧')
stage3_parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
# stage3_parser.add_argument('--pause', dest='pause', action='store_true', help='pause, 在各步暂停确认')
stage3_parser.add_argument('--no-scdet', dest='no_scdet', action='store_true', help='关闭转场识别（针对无转场素材）')
stage3_parser.add_argument('--no-concat', dest='no_concat', action='store_true', help='关闭自动合并（适合磁盘空间少使用）')
stage3_parser.add_argument('--remove-dup', dest='remove_dup', action='store_true', help='动态去除重复帧（预计会额外花很多时间）')
stage3_parser.add_argument('--quick-extract', dest='quick_extract', action='store_true', help='快速抽帧')
stage3_parser.add_argument('--output-only', dest='output_only', action='store_true', help='仅保留输出（删除其他）')
stage3_parser.add_argument('--audio', dest='save_audio', action='store_true', help='保留音频')
# stage3_parser_rife_set.add_argument('--accurate', dest='accurate', action='store_true', help='精确补帧')
stage3_parser_hardware_set.add_argument('--reverse', dest='reverse', action='store_true', help='反向光流')

stage4_parser = parser.add_argument_group(title="Output Settings")
stage4_parser.add_argument('--scdet-threshold', dest='scdet_threshold', type=float, default=30,
                           help="转场间隔阈值判定，要求相邻转场间隔大于该阈值")
stage4_parser.add_argument('--dup-threshold', dest='dup_threshold', type=float, default=0.8,
                           help='单次匹配重复帧最大重复数，默认: %(default)s')
stage4_parser.add_argument('--crop', dest='crop', type=str, default="0",
                           help="视频裁切参数，如3840:1608:0:276")
stage4_parser.add_argument('--resize', dest='resize', type=str, default="", help="ffmpeg -s 缩放参数，默认不开启（为空）")
stage4_parser.add_argument('-b', '--bitrate', dest='bitrate', type=str, default="80M", help="成品目标(最高)码率")
stage4_parser.add_argument('--preset', dest='preset', type=str, default="slow", help="压制预设，medium以下可用于收藏。硬件加速推荐hq")
stage4_parser.add_argument('--crf', dest='crf', type=int, default=9, help="恒定质量控制，12以下可作为收藏，16能看，默认：%(default)s")

args = parser.parse_args()

INPUT_FILEPATH = args.input
OUTPUT_DIR = args.output
SOURCE_FPS = args.fps
TARGET_FPS = args.target_fps
EXP = args.ratio

input_img = args.img_input
output_img = args.img_output
output_only = args.output_only

HDR = args.HDR
ncnn = args.ncnn
use_cpu = args.use_cpu
use_multi_card = args.use_multi_card

CHUNK_CNT = args.chunk
CHUNK_SIZE = args.chunk_size
bitrate = args.bitrate
CROP = args.crop
resize = args.resize
hwaccel = args.hwaccel
ffmpeg = os.path.join(args.ffmpeg, "ffmpeg.exe")
ffplay = os.path.join(args.ffmpeg, "ffplay.exe")
if not os.path.exists(ffmpeg):
    ffmpeg = "ffmpeg"
    print("[ARGS] Not find selected ffmpeg")

no_scdet = args.no_scdet
no_concat = args.no_concat

debug = args.debug
fp16 = args.fp16
interp_scale = args.scale
preset = args.preset
crf = args.crf
# accurate = args.accurate
reverse = args.reverse
interp_start = args.interp_start
interp_cnt = args.interp_cnt

scdet_threshold = args.scdet_threshold
remove_dup = args.remove_dup
dup_threshold = args.dup_threshold
quick_extract = args.quick_extract
save_audio = args.save_audio

failed_frames_cnt = False
if os.path.isfile(OUTPUT_DIR):
    PROJECT_DIR = os.path.dirname(OUTPUT_DIR)
else:
    PROJECT_DIR = OUTPUT_DIR
"""Set Logger"""
sys.path.append(PROJECT_DIR)
logger = Utils.get_logger("[ARGS]", PROJECT_DIR)
logger.info(f"Initial New Interpolation Project: PROJECT_DIR: %s, INPUT_FILEPATH: %s", PROJECT_DIR, INPUT_FILEPATH)

FRAME_INPUT_DIR = os.path.join(PROJECT_DIR, 'frames')
FRAME_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'interp')
SCENE_INPUT_DIR = os.path.join(PROJECT_DIR, 'scenes')
ENV = [SCENE_INPUT_DIR, FRAME_INPUT_DIR, FRAME_OUTPUT_DIR]


def clean_env():
    for DIR_ in ENV:
        if not os.path.exists(DIR_):
            os.mkdir(DIR_)


"""Frame Count"""
if not input_img:
    video_info = ffprobe(INPUT_FILEPATH)
    if "video" in video_info:
        video_info = video_info["video"]
    if not SOURCE_FPS:
        if "@r_frame_rate" in video_info:
            fps_info = video_info["@r_frame_rate"].split('/')
            SOURCE_FPS = int(fps_info[0]) / int(fps_info[1])
            logger.info("Auto Find FPS in r_frame_rate: %s", SOURCE_FPS)
        elif "@avg_frame_rate" in video_info:
            fps_info = video_info["@avg_frame_rate"].split('/')
            SOURCE_FPS = int(fps_info[0]) / int(fps_info[1])
            logger.warning("Auto Find FPS in avg_frame_rate: %s", SOURCE_FPS)
        else:
            logger.warning("Auto Find FPS Failed: and set it to 23.976", video_info)
            SOURCE_FPS = 24000 / 1001

    if "@nb_frames" in video_info:
        all_frames_cnt = int(video_info["@nb_frames"])
        logger.info(f"Auto Find frames cnt in nb_frames: {all_frames_cnt}")
    elif "@duration" in video_info:
        all_frames_cnt = round(video_info["@duration"]) * SOURCE_FPS
        logger.warning(f"Auto Find Frames Cnt by duration deduction: {all_frames_cnt}")
    else:
        failed_frames_cnt = True
        logger.warning("Not Find Frames Cnt")
        all_frames_cnt = 0
else:
    """FRAMES INPUT"""
    video_info = {}
    all_frames_cnt = len(os.listdir(INPUT_FILEPATH))
TARGET_FPS = (2 ** EXP) * SOURCE_FPS if not TARGET_FPS else TARGET_FPS  # Maintain Double Accuracy of FPS
logger.info(f"Check Interpolation Source, FPS: {SOURCE_FPS}, TARGET FPS: {TARGET_FPS}, FRAMES_CNT: {all_frames_cnt}")


class InterpWorkFlow:
    def __init__(self, __video_info, **kwargs):
        self.video_info = __video_info
        self.color_info = Utils.parse_video_info(HDR, __video_info)
        if ncnn:
            self.rife_core = inference.NCNNinterpolator(args)
        else:
            self.rife_core = inference.RifeInterpolation(args)
        # self.ffmpeg_reader = self.generate_ffmpeg_reader()
        self.logger = kwargs["logger"]
        self.duplicate_threshold = kwargs.get("dup_threshold", 1.0)
        self.scene_threshold = kwargs.get("scdet_threshold", 30)
        self.render_gap = kwargs.get("CHUNK_SIZE", 1000)
        self.start_frame = kwargs.get("start_frame", 0)
        self.start_cnt = kwargs.get("start_cnt", 1)
        self.frame_reader = None
        self.render_thread = None
        self.frames_output = Queue(maxsize=int(self.render_gap * 1.5))
        self.pos_map = Utils.generate_prebuild_map()
        # pprint(self.pos_map)
        self.render_info_pipe = {"rendering": (0, 0, 0, 0)}
        self.scene_stack_len = 6
        self.scene_stack = deque(maxlen=self.scene_stack_len)
        self.main_event = threading.Event()
        self.main_event.set()
        self.save_audio = kwargs.get("save_audio", True)
        self.input_img = kwargs.get("input_img", False)
        self.output_img = kwargs.get("output_img", False)
        self.output_only = kwargs.get("output_only", False)
        self.no_concat = kwargs.get("no_concat", False)
        self.no_scdet = kwargs.get("no_scdet", False)
        pass

    def generate_frame_reader(self, start_frame=-1):
        if self.input_img:
            return utils.ImgSeqIO(folder=INPUT_FILEPATH, is_read=True)
        input_dict = {"-vsync": "0", "-to": "00:00:01"}
        input_dict = {"-vsync": "0"}
        output_dict = {}
        vf_args = "copy"

        if len(resize):
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp", "-s": resize.replace(":", "x")})
        if start_frame not in [-1, 0]:
            vf_args += f",trim=start_frame={start_frame - 1}"
        if not quick_extract:
            vf_args += f",format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24"
        output_dict["-vf"] = vf_args
        # print(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=INPUT_FILEPATH, inputdict=input_dict, outputdict=output_dict)

    def generate_frame_renderer(self, output_path):
        if self.output_img:
            return utils.ImgSeqIO(folder=OUTPUT_DIR, is_read=False)
        input_dict = {"-vsync": "0", "-r": f"{SOURCE_FPS * 2 ** EXP}"}
        output_dict = {"-vsync": "0", "-r": f"{TARGET_FPS}", "-preset": preset}
        output_dict.update(self.color_info)
        vf_args = "copy"
        if CROP != "0":
            vf_args += f",crop={CROP}"
        output_dict.update({"-vf": vf_args})
        if HDR:
            if not hwaccel:
                output_dict.update({"-c:v": "libx265",
                                    "-tune": "grain",
                                    "-profile:v": "main10",
                                    "-pix_fmt": "yuv420p10le",
                                    "-x265-params": "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=0,0",
                                    "-crf": f"{crf}",
                                    })

            else:
                output_dict.update({"-c:v": "hevc_nvenc",
                                    "-rc:v": "vbr_hq",
                                    "-profile:v": "main10",
                                    "-pix_fmt": "p010le",
                                    "-cq:v": f"{crf}",
                                    })
                """GPU online, pause interpolation"""
        else:
            if not hwaccel:
                output_dict.update({"-c:v": "libx264",
                                    "-tune": "grain",
                                    "-pix_fmt": "yuv420p",
                                    "-crf": f"{crf}",
                                    })
            else:
                output_dict.update({"-c:v": "h264_nvenc",
                                    "-rc:v": "vbr_hq",
                                    "-pix_fmt": "yuv420p",
                                    "-cq:v": f"{crf}",
                                    })
        # print(f"writer: {input_dict} {output_dict}")
        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)

    def check_chunk(self, del_chunk=False):
        """
        Get Chunk Start
        :param: del_chunk: delete all chunks existed
        :return:
        """
        chunk_list = list()
        for f in os.listdir(PROJECT_DIR):
            if re.match("chunk-[\d+].*?\.mp4", f):
                if del_chunk:
                    os.remove(os.path.join(PROJECT_DIR, f))
                else:
                    chunk_list.append(f)
        if del_chunk:
            return
        if self.start_frame != 0 or self.start_cnt != 1:
            """Manually Prioritized"""
            return 1, self.start_frame
        if not len(chunk_list):
            return 1, 0
        """Remove last chunk(high possibility of dilapidation)"""
        chunk_list.sort(key=lambda x: int(x.split('-')[2]))
        os.remove(os.path.join(PROJECT_DIR, chunk_list[-1]))
        chunk_list.pop(-1)
        if not len(chunk_list):
            return 1, 0
        logger.info("Found Previous Chunks")
        last_chunk = chunk_list[-1]
        match_result = re.findall("chunk-(\d+)-(\d+)-(\d+)\.mp4", last_chunk)[0]

        chunk = int(match_result[0])
        last_frame = int(match_result[2])
        return chunk + 1, last_frame + 1

    def render(self, chunk_cnt, start_frame):
        end_frame = int(start_frame + self.render_gap / (2 ** EXP) - 1)
        output_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}.mp4".format(chunk_cnt, start_frame, end_frame)
        output_path = os.path.join(PROJECT_DIR, output_path)
        if not self.output_img:
            logger.info(f"First Chunk Render Path: {output_path}")
        chunk_round = 1
        frame_writer = self.generate_frame_renderer(output_path)
        while True:
            if not self.main_event.is_set():
                logger.info("Main interpolation thread Dead, break")
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
                end_frame = int(start_frame + self.render_gap / (2 ** EXP) - 1)
                output_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}.mp4".format(chunk_cnt, start_frame, end_frame)
                output_path = os.path.join(PROJECT_DIR, output_path)
                frame_writer.close()
                frame_writer = self.generate_frame_renderer(output_path)
            pass

    def feed_to_render(self, frames_list, is_scene=False, is_end=False):
        frames_list_len = len(frames_list)
        for frame_i in range(frames_list_len):
            self.frames_output.put(frames_list[frame_i])
            if frame_i == frames_list_len - 1:
                if is_end:
                    self.frames_output.put(None)
                    logger.info("Put None to write_buffer")
                    return
                if is_scene:
                    for put_i in range(2 ** EXP - 1):
                        self.frames_output.put(frames_list[frame_i])
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
        if after_measure > before_measure + 10:
            """Detect new scene"""
            self.scene_stack.clear()
            return True
        else:
            self.scene_stack.append(diff)
            return False

    def run(self):
        _debug = False
        self.rife_core.initiate_rife(args)
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.frame_reader = self.generate_frame_reader(start_frame)
        self.render_thread = threading.Thread(target=self.render, name="[ARGS] RenderThread",
                                              args=(chunk_cnt, start_frame,))
        self.render_thread.start()
        videogen = self.frame_reader.nextFrame()
        img1 = Utils.gen_next(videogen)
        now_frame = start_frame
        if img1 is None:
            logger.critical(f"Input file not valid: {INPUT_FILEPATH}")
            self.main_event.clear()
            sys.exit()

        is_end = False
        pbar = tqdm.tqdm(total=all_frames_cnt)
        pbar.update(n=start_frame)
        slot_img = None
        recent_scene = 0
        previous_cnt = now_frame
        scedet_info = {"0": 0, "1": 0, "1+": 0}
        while True:
            if is_end:
                break
            if slot_img is not None:
                img0 = slot_img
                slot_img = None
            else:
                img0 = img1
            img1 = None
            frames_list = []
            if remove_dup:
                now_gap = 0
                is_scene = False
                while True:
                    if now_gap == 8:
                        break
                    before_img1 = img1
                    img1 = Utils.gen_next(videogen)
                    if img1 is None:
                        img1 = before_img1
                        is_end = True
                        break
                    diff = cv2.absdiff(img0, img1).mean()
                    if self.check_scene(diff):
                        slot_img = img1
                        img1 = before_img1
                        is_scene = True
                        recent_scene = now_frame + now_gap
                        break

                    if diff < self.duplicate_threshold:
                        now_gap += 1
                        continue

                    now_gap += 1
                    break

                if now_gap == 0:
                    """Normal frames Encounter scenes"""
                    frames_list.append((now_frame, img0))
                    self.feed_to_render(frames_list, is_scene=is_scene, is_end=is_end)
                    scedet_info["0"] += 1
                    if is_end:
                        break
                    continue
                elif now_gap == 1:
                    # print(f"check img0 == img1 {img0} {img1}, {img0 == img1}")
                    frames_list.append((now_frame, img0))
                    interp_output = self.rife_core.generate_interp(img0, img1, EXP, interp_scale, _debug)
                    for interp in interp_output:
                        frames_list.append((now_frame, interp))
                    now_frame += 1
                    self.feed_to_render(frames_list, is_scene=is_scene, is_end=is_end)
                    scedet_info["1"] += 1

                    if is_end:
                        break
                    continue
                elif now_gap > 1:
                    # print(f"intertp between {now_frame} - {now_frame + now_gap}")
                    """exists gap > 1"""
                    exp = round(math.sqrt(now_gap))
                    gap_output = self.rife_core.generate_interp(img0, img1, exp, interp_scale)
                    gap_input = [img0]
                    for gap_i in range(now_gap - 1):
                        gap_input.append(gap_output[self.pos_map[(now_gap, gap_i)]])
                    for gap_i in range(len(gap_input) - 1):
                        interp_output = self.rife_core.generate_interp(gap_input[gap_i], gap_input[gap_i + 1], EXP,
                                                                       interp_scale, _debug)
                        frames_list.append((now_frame + gap_i, gap_input[gap_i]))
                        for interp_i in range(len(interp_output)):
                            frames_list.append((now_frame + gap_i, interp_output[interp_i]))
                        frames_list.append((now_frame + gap_i, gap_input[gap_i + 1]))

                    now_frame += now_gap
                    self.feed_to_render(frames_list, is_scene=is_scene, is_end=is_end)
                    scedet_info["1+"] += 1
                    if is_end:
                        break

            else:
                img1 = Utils.gen_next(videogen)
                if img1 is None:
                    frames_list.append((now_frame, img0))
                    self.feed_to_render(frames_list, is_end=True)
                    # is_end = True
                    break
                if not self.no_scdet:
                    diff = cv2.absdiff(img0, img1).mean()
                    # print(f"now_frame {now_frame} - {now_frame + 1}, diff {diff}")
                    if self.check_scene(diff):
                        """!!!scene"""
                        # print(f"Find Scene now_frame {now_frame} - {now_frame + 1}")
                        frames_list.append((now_frame, img0))
                        self.feed_to_render(frames_list, is_scene=True)
                        recent_scene = now_frame
                        now_frame += 1  # to next frame img0 = img1
                        scedet_info["0"] += 1
                        continue
                frames_list.append((now_frame, img0))
                interp_output = self.rife_core.generate_interp(img0, img1, EXP, interp_scale)
                for interp in interp_output:
                    frames_list.append((now_frame, interp))
                self.feed_to_render(frames_list)
                now_frame += 1  # to next frame img0 = img1
                scedet_info["1"] += 1

            rsq = self.render_info_pipe["rendering"]  # render status quo
            """(chunk_cnt, start_frame, end_frame, frame_cnt)"""
            pbar.set_description(
                f"Process at chunk {rsq[0]:0>3d}, {rsq[1]:0>8d} - {rsq[2]:0>8d}, frame_cnt {rsq[3]:0>8d}, now_frame {now_frame:0>8d}, recent_scene {recent_scene:0>8d}")
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame
        pbar.close()
        logger.info(f"Scedet Status Quo: {scedet_info}")
        while self.render_thread.is_alive():
            time.sleep(0.1)
        self.concat_all()
        pass

    def concat_all(self):
        if self.no_concat:
            return
        os.chdir(PROJECT_DIR)
        concat_path = os.path.join(PROJECT_DIR, "concat.txt")
        self.logger.info("Final Round Finished, Start Concating")
        concat_list = list()
        for f in os.listdir(PROJECT_DIR):
            if re.match("chunk-[\d+].*?\.mp4", f):
                concat_list.append(os.path.join(PROJECT_DIR, f))
            else:
                self.logger.debug(f"concat escape {f}")
        concat_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))  # sort as start-frame
        if os.path.exists(concat_path):
            os.remove(concat_path)
        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")
        concat_filepath = f"{os.path.splitext(os.path.basename(INPUT_FILEPATH))[0]}_{2 ** EXP}x" + \
                          os.path.splitext(INPUT_FILEPATH)[-1]
        if self.save_audio:
            map_audio = f'-i "{INPUT_FILEPATH}" -map 0:v:0 -map 1:a:0 -c:a copy -shortest '
        else:
            map_audio = ""
        ffmpeg_command = f'{ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy {concat_filepath} -y'
        self.logger.info(f"Concat command: {ffmpeg_command}")
        os.system(ffmpeg_command)
        if self.output_only:
            self.check_chunk(del_chunk=True)


interpworkflow = InterpWorkFlow(video_info,
                                scdet_threshold=scdet_threshold, dup_threshold=dup_threshold, CHUNK_SIZE=CHUNK_SIZE,
                                logger=logger, start_frame=interp_start, start_cnt=interp_cnt, save_audio=save_audio,
                                input_img=input_img, output_img=output_img, output_only=output_only,
                                no_scdet=no_scdet, no_concat=no_concat)
interpworkflow.run()
logger.info(f"Program finished at {datetime.datetime.now()}")
sys.exit()
