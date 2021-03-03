# coding: utf-8
import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pprint import pprint

"""Set Path Environment"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing working dir to {0}".format(dname))
os.chdir(os.path.dirname(dname))
print("Added {0} to temporary PATH".format(dname))
sys.path.append(dname)

parser = argparse.ArgumentParser(prog="#### RIFE Step by Step CLI tool/补帧分步设置命令行工具 from Jeanna ####",
                                 description='Interpolation for sequences of images')
# TODO: Use "Click" Commander
stage1_parser = parser.add_argument_group(title="Basic Settings, Necessary")
stage1_parser.add_argument('-i', '--input', dest='input', type=str, default=None, required=True,
                           help="原视频路径, 补帧项目将在视频所在文件夹建立")
stage1_parser.add_argument('-o', '--output', dest='output', type=str, default=None, required=True,
                           help="成品输出的路径，注意默认在项目文件夹")
stage1_parser.add_argument('--rife', dest='rife', type=str, default="inference_img_only[v1].py",
                           help="inference_img_only.py的路径")
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
stage2_parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=1000, help="每一个chunk包含的帧数量, 默认: %(default)s")
stage2_parser.add_argument('--interp-start', dest='interp_start', type=int, default=0,
                           help="用于补帧的原视频的帧序列起始帧数，默认：%(default)s")
stage2_parser.add_argument('--interp-cnt', dest='interp_cnt', type=int, default=1, help="成品帧序列起始帧数")
stage2_parser.add_argument('--hwaccel', dest='hwaccel', action='store_true', help='支持硬件加速编码(想搞快点就用上)')
stage2_parser.add_argument('--model', dest='model', type=int, default=2, help="Select RIFE Model, default v2")

stage3_parser = parser.add_argument_group(title="Preference Settings")
stage3_parser_paradox = stage3_parser.add_mutually_exclusive_group()
stage3_parser_paradox_2 = stage3_parser.add_mutually_exclusive_group()
stage3_parser.add_argument('--UHD', dest='UHD', action='store_true', help='支持UHD补帧')
stage3_parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
stage3_parser.add_argument('--pause', dest='pause', action='store_true', help='pause, 在各步暂停确认')
stage3_parser.add_argument('--quick-extract', dest='quick_extract', action='store_true', help='快速抽帧')
stage3_parser_paradox.add_argument('--extract-only', dest='extract_only', action='store_true', help='只进行帧序列提取操作')
stage3_parser_paradox.add_argument('--rife-only', dest='rife_only', action='store_true', help='只进行补帧操作')
stage3_parser_paradox.add_argument('--render-only', dest='render_only', action='store_true', help='只进行渲染操作')
stage3_parser_paradox.add_argument('--concat-only', dest='concat_only', action='store_true', help='只进行音视频合并操作')
stage3_parser_paradox_2.add_argument('--accurate', dest='accurate', action='store_true', help='精确补帧')
stage3_parser_paradox_2.add_argument('--reverse', dest='reverse', action='store_true', help='反向光流')

stage4_parser = parser.add_argument_group(title="Output Settings")
stage4_parser.add_argument('--scdet', dest='scdet', type=float, default=3, help="转场识别灵敏度，越小越准确，人工介入也会越多")
stage4_parser.add_argument('--scdet-threshold', dest='scdet_threshold', type=float, default=0.2,
                           help="转场间隔阈值判定，要求相邻转场间隔大于该阈值")
stage4_parser.add_argument('--UHD-crop', dest='UHDcrop', type=str, default="3840:1608:0:276",
                           help="UHD裁切参数，默认开启，填0不裁，默认：%(default)s")
stage4_parser.add_argument('--HD-crop', dest='HDcrop', type=str, default="1920:804:0:138",
                           help="QHD裁切参数，默认：%(default)s")
stage4_parser.add_argument('--resize', dest='resize', type=str, default="", help="ffmpeg -s 缩放参数，默认不开启（为空）")
stage4_parser.add_argument('-b', '--bitrate', dest='bitrate', type=str, default="80M", help="成品目标(最高)码率")
stage4_parser.add_argument('--preset', dest='preset', type=str, default="slow", help="压制预设，medium以下可用于收藏。硬件加速推荐hq")
stage4_parser.add_argument('--crf', dest='crf', type=int, default=9, help="恒定质量控制，12以下可作为收藏，16能看，默认：%(default)s")

args = parser.parse_args()

INPUT_FILEPATH = args.input
OUTPUT_FILE_PATH = args.output
SOURCE_FPS = args.fps
TARGET_FPS = args.target_fps
EXP = args.ratio

UHD = args.UHD
RIFE_PATH = args.rife
CHUNK_CNT = args.chunk
CHUNK_SIZE = args.chunk_size
scdet = args.scdet
scdet_threshold = args.scdet_threshold
bitrate = args.bitrate
UHDcrop = args.UHDcrop
HDcrop = args.HDcrop
ffmpeg = os.path.join(args.ffmpeg, "ffmpeg.exe")
ffprobe = os.path.join(args.ffmpeg, "ffprobe.exe")
ffplay = os.path.join(args.ffmpeg, "ffplay.exe")
debug = args.debug
hwaccel = args.hwaccel
preset = args.preset
crf = args.crf
accurate = args.accurate
reverse = args.reverse
resize = args.resize
pause = args.pause
interp_start = args.interp_start
interp_cnt = args.interp_cnt
model_select = args.model

extract_only = args.extract_only
rife_only = args.rife_only
render_only = args.render_only
concat_only = args.concat_only
quick_extract = args.quick_extract
failed_frames_cnt = False

PROJECT_DIR = os.path.dirname(OUTPUT_FILE_PATH)

"""Set Logger"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')

logger_path = os.path.join(PROJECT_DIR, f"{datetime.datetime.now().date()}-{EXP}-{interp_start}-{interp_cnt}.txt")
txt_handler = logging.FileHandler(logger_path)
txt_handler.setLevel(level=logging.INFO)
txt_handler.setFormatter(logger_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(level=logging.INFO)
console_handler.setFormatter(logger_formatter)

logger.addHandler(console_handler)
logger.addHandler(txt_handler)
logger.info(f"Initial New Interpolation Project: PROJECT_DIR: %s, INPUT_FILEPATH: %s", PROJECT_DIR, INPUT_FILEPATH)

FRAME_INPUT_DIR = os.path.join(PROJECT_DIR, 'frames')
FRAME_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'interp')
SCENE_INPUT_DIR = os.path.join(PROJECT_DIR, 'scenes')
if not os.path.exists(SCENE_INPUT_DIR):
    os.mkdir(SCENE_INPUT_DIR)

ENV = [FRAME_INPUT_DIR, FRAME_OUTPUT_DIR]




def clean_env():
    for DIR_ in ENV:
        if not os.path.exists(DIR_):
            os.mkdir(DIR_)


class FFmpegQuickReader:
    def __init__(self, output_txt):
        self.output_txt = os.path.join(PROJECT_DIR, output_txt)
        self.tool_list = {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "ffplay": ffplay}
        pass

    def get_tool(self, tool):
        if tool not in self.tool_list:
            logger.warning(f"Not Recognize tool: {tool}")
            return ""
        else:
            return self.tool_list[tool]

    def execute(self, tool, command):
        tool = self.get_tool(tool)
        os.system(f'{tool} {command} > {self.output_txt} 2>&1')
        with open(self.output_txt, "r", encoding="utf-8") as tool_read:
            content = tool_read.read()
        return content


"""Frame Count"""
ffmpeg_reader = FFmpegQuickReader("video_info.txt")
video_info = ffmpeg_reader.execute("ffprobe",
                                   f"-v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate,nb_frames {INPUT_FILEPATH}").splitlines()
if not SOURCE_FPS:
    if video_info[0] != "N/A":
        logger.warning("Auto Find FPS Failed: %s and set it to 23.976")
        SOURCE_FPS = 24000/1001

    if '/' in video_info[0]:
        fps_info = video_info[0].split('/')
        SOURCE_FPS = int(fps_info[0]) / int(fps_info[1])
        logger.warning("Auto Find FPS in fraction: %s", SOURCE_FPS)
    elif len(video_info[0]):
        SOURCE_FPS = float(video_info[0])
        logger.warning("Auto Find FPS in decimal: %s", SOURCE_FPS)

if video_info[1] == "N/A":
    logger.warning("Not Find Frames Cnt")
    failed_frames_cnt = True
    all_frames_cnt = 0
else:
    all_frames_cnt = int(video_info[1])
TARGET_FPS = (2 ** EXP) * SOURCE_FPS if not TARGET_FPS else TARGET_FPS  # Maintain Double Accuracy of FPS
logger.info(f"Check Interpolation Source, FPS: {SOURCE_FPS}, TARGET FPS: {TARGET_FPS}, FRAMES_CNT: {all_frames_cnt}")


class SceneDealer:
    def __init__(self):
        self.scenes_data = {}
        self.scene_json = os.path.join(PROJECT_DIR, "scene_json.json")
        self.tool_list = {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "ffplay": ffplay}

    def get_scenes_data(self):
        if not self.scdet_read():
            self.scdet_get()
        return self.scenes_data

    def scdet_get(self):
        scene_reader = FFmpegQuickReader("scenes.txt")
        scene_content = scene_reader.execute("ffmpeg",
                                             f'-hide_banner -hwaccel auto -i "{INPUT_FILEPATH}" -an -vsync 0 -frame_pts true -copyts -vf scdet=t={scdet}:sc_pass=1,scale=iw/4:ih/4 -compression_level 7 {os.path.join(SCENE_INPUT_DIR, "%08d.png")}')
        scenes_time = re.findall(".*?lavfi\.scd\.time:\s{,5}([\d\.]+).*?", scene_content)
        scenes_list = sorted(os.listdir(SCENE_INPUT_DIR), key=lambda x: int(x[:-4]))
        scenes_tuple = zip(scenes_list, scenes_time)  # 文件名, 时间
        scene_t = 1e-2
        _last_scene_time = 0
        for _scene in reversed(list(scenes_tuple)):
            scene_pts, scene_time = int(_scene[0][:-4]), float(_scene[1])
            if scene_pts / SOURCE_FPS - scene_time > scene_t:
                logger.warning(f"Shifted Scene pts: {scene_pts}, time: {scene_time}, discarded")
                os.remove(os.path.join(SCENE_INPUT_DIR, _scene[0]))
                continue
            if abs(scene_time - _last_scene_time) < scdet_threshold:
                logger.warning(f"Too Near Scene pts: {scene_pts}, time: {scene_time} -> {_last_scene_time}, discarded")
                os.remove(os.path.join(SCENE_INPUT_DIR, _scene[0]))
                continue
            _last_scene_time = scene_time
        # TODO: Upgrade specified situations for "Pause"
        if pause:
            input("Wait for Finish of Manually Cleaning Scenes")
        scenes_list = sorted(os.listdir(SCENE_INPUT_DIR), key=lambda x: int(x[:-4]), reverse=True)
        for _scene in scenes_list:
            self.scenes_data[str(_scene[:-4])] = int(_scene[:-4]) / SOURCE_FPS
        logger.info(f"Get New Fresh Scene Data: {len(self.scenes_data)}")
        pprint(self.scenes_data)
        self.scdet_dump()

    def scdet_read(self):
        if os.path.exists(self.scene_json):
            with open(self.scene_json, "r") as r_:
                self.scenes_data = json.load(r_)
                logger.info(f"Loaded Scenes: {len(self.scenes_data)}")
                pprint(self.scenes_data)
                return True
        return False

    def scdet_dump(self):
        with open(self.scene_json, "w") as w:
            json.dump(self.scenes_data, w)


class ExtractFrames:
    def __init__(self):
        pass

    def run(self):
        if not self.check_frames():
            self.extract_frames()

    def check_frames(self):
        global all_frames_cnt
        check = True
        if failed_frames_cnt:
            if pause:
                proceed = input("Failed to count all frames, press 'Y' or 'y' to proceed: ")
                if proceed not in ["Y", "y"]:
                    return False
            all_frames_cnt = len(os.listdir(FRAME_INPUT_DIR))
            if not all_frames_cnt:
                logger.warning("Initial FRAME_INPUT_DIR Empty")
                return False
            logger.warning("Update all_frames_cnt: %d", all_frames_cnt)
            return True
        for frame in range(all_frames_cnt):
            frame_path = "{:0>8d}.png".format(frame)
            if not os.path.exists(os.path.join(FRAME_INPUT_DIR, frame_path)):
                logger.warning(f"[Frame Check]: PNG {frame} not exists, ReExtraction is Necessary")
                check = False
                break
        return check

    def extract_frames(self):
        global all_frames_cnt
        logger.info(
            f"\n\n\nExtract All the Frames at once from 0 to {all_frames_cnt - 1}, UHD: {UHD}")
        frame_input = os.path.join(FRAME_INPUT_DIR, "%08d.png")
        extract_head = f'{ffmpeg} -hide_banner -hwaccel auto -r {SOURCE_FPS}  -i {INPUT_FILEPATH} -vsync 0 -copyts -frame_pts true -vf'
        extract_color_filters = "format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24" if not quick_extract else "copy"
        UHDcrop_args = f",crop={UHDcrop}" if UHDcrop != "0" else ""
        HDcrop_args = f",crop={HDcrop}" if HDcrop != "0" else ""
        if len(resize):
            low_res = f"-s {resize}"
        else:
            low_res = f"-s 720x302" if debug else ""
        if UHD:
            ffmpeg_command = f'{extract_head} {extract_color_filters}{UHDcrop_args} {low_res} -compression_level 2 -pix_fmt rgb24 "{frame_input}" -y'
        else:
            ffmpeg_command = f'{extract_head} {extract_color_filters}{HDcrop_args} {low_res} -compression_level 2 -pix_fmt rgb24  "{frame_input}" -y'
        logger.info(f"Extract Frames FFmpeg Command: {ffmpeg_command}")
        os.system(ffmpeg_command)
        all_frames_cnt = len(os.listdir(FRAME_INPUT_DIR))
        return


def interpolation():
    logger.info(f"\n\n\nInterpolation: UHD: {UHD}, Frame {interp_cnt} -> {interp_start}")
    UHD_args = "--UHD" if UHD else ""
    accurate_args = "--accurate" if accurate else ""
    reverse_args = "--reverse " if reverse else ""
    if os.path.splitext(RIFE_PATH)[1] == ".exe":
        """Executable RIFE"""
        logger.info("Use Executable RIFE")
        python_command = f"{RIFE_PATH} --img {FRAME_INPUT_DIR} --exp {EXP} {accurate_args} {reverse_args} {UHD_args} --imgformat png --output {FRAME_OUTPUT_DIR} --cnt {interp_cnt} --start {interp_start} --model {model_select}"
    else:
        python_command = f"python {RIFE_PATH} --img {FRAME_INPUT_DIR} --exp {EXP} {accurate_args} {reverse_args} {UHD_args} --imgformat png --output {FRAME_OUTPUT_DIR} --cnt {interp_cnt} --start {interp_start} --model {model_select}"
    logger.info(f"Interpolation Python Command: {python_command}")
    os.system(python_command)
    logger.info("Interpolation process is over")


class RenderThread(threading.Thread):
    def __init__(self, interp_finished, input_dir, output_dir, **kwargs):
        threading.Thread.__init__(self)
        self.interp_finished = interp_finished
        self.scene_data = kwargs["scene_data"]
        self.exp = kwargs["exp"]
        self.start_frame = 0
        self.end_frame = 0
        self.chunk_cnt = kwargs["chunk_cnt"]
        self.render_gap = kwargs["chunk_size"]
        self.target_fps = kwargs["target_fps"]
        self.source_fps = kwargs["source_fps"]
        self.all_frames_cnt = kwargs["all_frames_cnt"]
        self.ffmpeg = kwargs["ffmpeg"]
        self.logger = kwargs["logger"]
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.project_dir = os.path.dirname(input_dir)
        self.rife_current_png = None
        self.render_init_path = ""
        self.rendered_list = list()
        self.final_round = False
        self.render_only = False
        self.concat_only = False

    def generate_render_ini(self):
        rife_rendered_list = sorted(os.listdir(self.output_dir), key=lambda x: int(x[:-4]))
        if not len(rife_rendered_list):
            """No enough interpolated frames yet"""
            return False
        rife_first_png = rife_rendered_list[0]
        rife_last_png = rife_rendered_list[-1]
        rife_first = int(rife_first_png[:-4])
        rife_last = int(rife_last_png[:-4])

        """Overwrite authority with render_only flag"""
        if self.render_only:
            self.logger.info(f"Final Round for Render only Detected, {rife_first} -> {rife_last}")
            self.final_round = True
        else:
            if rife_last - rife_first + 1 >= self.render_gap:
                """There are many pictures to render, update rife_last"""
                rife_last = rife_first + self.render_gap - 1
            elif not self.interp_finished.is_set():
                """less than int(gap) pictures to render, and interp process is done"""
                self.logger.info(f"Final Round Detected, {rife_first} -> {rife_last}")
                self.final_round = True
            else:
                """less than int(gap) pictures to render, and interp process is not done, continue waiting"""
                return False

        scene_false = list()
        render_list = list()
        for _scene in self.scene_data:
            scene_pts = int(_scene)
            if scene_pts * (2 ** self.exp) > rife_last:
                break
            for i in range(2 ** self.exp - 1):
                scene_false.append(scene_pts * (2 ** self.exp) - i)
        """Start_Frame is calculated"""
        self.start_frame = round((rife_first - 1) / (2 ** self.exp))
        self.end_frame = round(rife_last / (2 ** self.exp) - 1)
        last_render_frame = 0
        self.rendered_list.clear()
        for render_frame in range(rife_first, rife_last + 1):
            render_file_path = os.path.join(self.output_dir, '{:0>8d}.png'.format(render_frame))
            self.rendered_list.append(render_file_path)
            if render_frame in scene_false:
                render_frame = last_render_frame
            else:
                last_render_frame = render_frame
            render_file_path = os.path.join(self.output_dir, '{:0>8d}.png'.format(render_frame))
            concat_line = f"file '{render_file_path}'\n"
            render_list.append(concat_line)

        self.render_init_path = os.path.join(os.path.dirname(self.output_dir),
                                             "chunk-{:0>3d}-{:0>8d}-{:0>8d}.ini".format(self.chunk_cnt,
                                                                                        self.start_frame,
                                                                                        self.end_frame))
        with open(self.render_init_path, "w") as w:
            w.writelines(render_list)
        self.logger.debug("Write %d render_path to ini for render", len(render_list))
        return True

    def render(self):
        output_chunk_path = f"{os.path.splitext(self.render_init_path)[0]}.mp4"
        ffmpeg_command = f'{ffmpeg}  -hide_banner -loglevel error -vsync 0 -f concat -safe 0 -r {self.source_fps * (2 ** self.exp)} -i "{self.render_init_path}" '
        render_start = datetime.datetime.now()
        if UHD:
            if not hwaccel:
                ffmpeg_command = ffmpeg_command + f'-c:v libx265 -preset {preset} -tune grain -profile:v main10 -pix_fmt yuv420p10le ' \
                                                  f'-color_range tv -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc ' \
                                                  f'-x265-params "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=0,0" ' \
                                                  f'-crf {crf} "{output_chunk_path}" -y'
            else:
                ffmpeg_command = ffmpeg_command + f'-c:v hevc_nvenc -preset {preset} -rc:v vbr_hq -cq:v {crf} -rc-lookahead:v 32 ' \
                                                  f'-refs:v 16 -bf:v 3 -b:v 100M -b_ref_mode:v middle -profile:v main10 ' \
                                                  f'-pix_fmt p010le ' \
                                                  f'-color_range tv -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc "{output_chunk_path}" -y'
        else:
            if not hwaccel:
                ffmpeg_command = ffmpeg_command + f'-c:v libx264 -pix_fmt yuv420p -preset {preset} -crf {crf} "{output_chunk_path}" -y'
            else:
                ffmpeg_command = ffmpeg_command + f'-c:v h264_nvenc -preset {preset} -rc:v vbr_hq -cq:v {crf} -rc-lookahead:v 32 ' \
                                                  f'-refs:v 16 -bf:v 3 -b:v 100M -b_ref_mode:v middle ' \
                                                  f'-pix_fmt yuv420p -preset {preset} "{output_chunk_path}" -y'
        self.logger.debug("Render FFmpeg Command: %s" % ffmpeg_command)
        subprocess.Popen(ffmpeg_command).wait()
        render_end = datetime.datetime.now()
        self.logger.info(f"Render: UHD: {UHD}, from {self.start_frame} -> {self.end_frame}, to {os.path.basename(output_chunk_path)} in {str(render_end - render_start)}, {len(os.listdir(self.output_dir))} interp frames left")

    def clean(self):
        try:
            for rendered_frame in self.rendered_list:
                if os.path.exists(rendered_frame):
                    pass
                    os.remove(rendered_frame)
            self.logger.info(f"Cleaned Rendered list: {len(self.rendered_list)}")
            self.rendered_list.clear()
            pass
        except Exception:
            self.logger.exception("Remove Frames Failed")
            time.sleep(0.5)
            self.clean()

    def concat(self):
        os.chdir(self.project_dir)
        concat_path = os.path.join(self.project_dir, "concat.txt")
        self.logger.info("Final Round Finished, Start Concating")
        concat_list = list()
        for f in os.listdir(self.project_dir):
            if re.match("chunk-[\d+].*?\.mp4", f):
                concat_list.append(f)
            else:
                self.logger.debug(f"concat escape {f}")
        concat_list.sort(key=lambda x: int(x.split('-')[2]))  # sort as start-frame
        if os.path.exists(concat_path):
            os.remove(concat_path)
        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")
        os.system(f'{self.ffmpeg} -f concat -safe 0 -i "{concat_path}" -c copy concat_all.mp4 -y')
        pass

    def run(self, _render_only=False, _concat_only=False):
        self.render_only = _render_only
        self.concat_only = _concat_only
        if self.concat_only:
            self.concat()
            return
        while True:
            if not self.generate_render_ini():
                time.sleep(0.5)
                if not self.interp_finished.is_set() and len(os.listdir(self.output_dir)):
                    self.logger.info("Interp Thread Already Terminated, Render Thread Break")
                    break
                continue
            self.render()
            self.clean()
            self.chunk_cnt += 1
            if self.final_round:
                self.concat()
                break
            time.sleep(5)
        pass


"""Instinct Procesd and Exit"""
Finished = threading.Event()
Finished.set()  # set means alive
clean_env()
scene_dealer = SceneDealer()
extract_frames = ExtractFrames()
extract_frames.check_frames()  # get all_frames_cnt right
render_frames = RenderThread(Finished, FRAME_INPUT_DIR, FRAME_OUTPUT_DIR, chunk_cnt=CHUNK_CNT, chunk_size=CHUNK_SIZE,
                             exp=EXP,
                             scene_data=scene_dealer.get_scenes_data(), target_fps=TARGET_FPS, source_fps=SOURCE_FPS,
                             all_frames_cnt=all_frames_cnt, ffmpeg=ffmpeg, logger=logger)

if extract_only:
    logger.info("Extract Frames Only")
    extract_frames.run()
    Finished.clear()
    logger.info(f"Program finished at {datetime.datetime.now()}")
    sys.exit()
if rife_only:
    logger.info("interpolation Only")
    interpolation()
    Finished.clear()
    logger.info(f"Program finished at {datetime.datetime.now()}")
    sys.exit()
if render_only:
    logger.info("Render Only")
    Finished.clear()
    render_frames.run(_render_only=True)
    logger.info(f"Program finished at {datetime.datetime.now()}")
    sys.exit()
if concat_only:
    logger.info("Concat Only")
    Finished.clear()
    render_frames.concat()
    logger.info(f"Program finished at {datetime.datetime.now()}")
    sys.exit()


extract_frames.run()
render_frames.start()
interpolation()
Finished.clear()
while render_frames.is_alive():
    time.sleep(0.1)
logger.info(f"Program finished at {datetime.datetime.now()}")
