# coding: utf-8
from pprint import pprint
import argparse
import os
import re
import shutil
import sys
import subprocess
import json
import datetime
import numpy  as np
import math
from PIL import Image
from skimage.metrics import structural_similarity as ssim2


def ssim(img1, img2, multichannel=True):
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img1 = img1.resize((int(img1.size[0] / 3), int(img1.size[1] / 3)), Image.ANTIALIAS)
    img2 = img2.resize((int(img2.size[0] / 3), int(img2.size[1] / 3)), Image.ANTIALIAS)
    img1 = np.array(img1)
    img2 = np.array(img2)
    return ssim2(img1, img2, multichannel=multichannel)


parser = argparse.ArgumentParser(prog="#### RIFE Step by Step CLI tool/补帧分步设置命令行工具 from Jeanna ####",
                                 description='Interpolation for sequences of images')

stage1_parser = parser.add_argument_group(title="Basic Settings, Necessary")
stage1_parser.add_argument('-i', '--input', dest='input', type=str, default=None, required=True,
                           help="原视频路径，记得打双引号。补帧项目将在视频所在文件夹建立")
stage1_parser.add_argument('-o', '--output', dest='output', type=str, default=None, required=True,
                           help="成品输出的路径，注意默认在项目文件夹")
stage1_parser.add_argument('--rife', dest='rife', type=str, default=None, required=True,
                           help="inference_img_only.py的路径")
stage1_parser.add_argument('--ffmpeg', dest='ffmpeg', type=str, default="ffmpeg", required=True, help="ffmpeg.exe 所在路径")
stage1_parser.add_argument('--fps', dest='fps', type=float, default=24000 / 1001,
                           help="原视频的帧率, 请使用分数，23.976 = 24000/1001")

stage2_parser = parser.add_argument_group(title="Step by Step Settings")
stage2_parser.add_argument('-r', '--ratio', dest='ratio', type=int, choices=range(1, 4), default=2, required=True,
                           help="补帧系数, 2的几次方，23.976->96，填2")
stage2_parser.add_argument('--start', dest='start', type=int, default=1, required=True, help="起始补帧的原视频帧数")
stage2_parser.add_argument('--chunk', dest='chunk', type=int, default=1, required=True, help="新增视频的序号")
stage2_parser.add_argument('--scene', dest='scene', type=int, default=1, required=True, help="开局转场编号")
stage2_parser.add_argument('--end', dest='end', type=int, default=0, help="结束补帧的原视频帧数")
stage2_parser.add_argument('--round', dest='round', type=int, default=100000000, help="要处理的转场数(一个转场将素材补成一个视频)")
stage2_parser.add_argument('--interp_start', dest='interp_start', type=int, default=1, help="起始补帧的帧序列帧数，默认：%(default)s")
stage2_parser.add_argument('--interp_cnt', dest='interp_cnt', type=int, default=1, help="补帧帧序列起始帧数")
stage2_parser.add_argument('--hwaccel', dest='hwaccel', action='store_true', help='支持硬件加速编码（想搞快点就可以用这个）')
stage2_parser.add_argument('--model', dest='model', type=int, default=2, help="Select RIFE Model, default v2")

stage3_parser = parser.add_argument_group(title="Preference Settings")
stage3_parser.add_argument('--UHD', dest='UHD', action='store_true', help='支持UHD补帧')
stage3_parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
stage3_parser.add_argument('--pause', dest='pause', action='store_true', help='pause, 删除项目缓存前提供确认操作')
stage3_parser.add_argument('--rifeonly', dest='rifeonly', action='store_true', help='只进行补帧操作，其余手动，要求文件夹目录齐全')
stage3_parser.add_argument('--renderonly', dest='renderonly', action='store_true', help='只进行渲染操作，其余手动，要求文件夹目录齐全')
stage3_parser.add_argument('--accurate', dest='accurate', action='store_true', help='精确补帧')
stage3_parser.add_argument('--reverse', dest='reverse', action='store_true', help='反向光流')

stage4_parser = parser.add_argument_group(title="Output Settings")
stage4_parser.add_argument('-s', '--scdet', dest='scdet', type=float, default=3, help="转场识别灵敏度，越小越准确，人工介入也会越多")
stage4_parser.add_argument('--UHDcrop', dest='UHDcrop', type=str, default="3840:1608:0:276",
                           help="UHD裁切参数，默认开启，填0不裁，默认：%(default)s")
stage4_parser.add_argument('--HDcrop', dest='HDcrop', type=str, default="1920:804:0:138", help="QHD裁切参数，默认：%(default)s")
stage4_parser.add_argument('-b', '--bitrate', dest='bitrate', type=str, default="80M", help="成品目标（最高）码率")
stage4_parser.add_argument('--preset', dest='preset', type=str, default="slow", help="压制预设，medium以下可用于收藏。硬件加速推荐hq")
stage4_parser.add_argument('--crf', dest='crf', type=int, default=9, help="恒定质量控制，12以下可作为收藏，16能看，默认：%(default)s")

args = parser.parse_args()

INPUT_FILEPATH = args.input
OUTPUT_FILE_PATH = args.output
SOURCE_FPS = args.fps
UHD = "--UHD" if args.UHD else ""
START_FRAME = args.start
scene_start = args.scene
RIFE_PATH = args.rife
CHUNK_CNT = args.chunk
scdet = args.scdet
bitrate = args.bitrate
EXP = args.ratio
UHDcrop = args.UHDcrop
HDcrop = args.HDcrop
ffmpeg = args.ffmpeg
debug = args.debug
hwaccel = args.hwaccel
preset = args.preset
crf = args.crf
accurate = "--accurate " if args.accurate else ""
reverse_ = "--reverse " if args.reverse else ""

# TARGET_FPS = (2 ** EXP) * round(SOURCE_FPS)
TARGET_FPS = (2 ** EXP) * SOURCE_FPS  # Maintain Double Accuracy of FPS

PROJECT_DIR = os.path.dirname(OUTPUT_FILE_PATH)
print(f"Project Dir: {PROJECT_DIR}")
HDcrop = f",crop={HDcrop}" if HDcrop != "0" else ""
UHDcrop = f",crop={UHDcrop}" if UHDcrop != "0" else ""
low_res = f"-s 720x302" if debug else ""
FRAME_INPUT_DIR = os.path.join(PROJECT_DIR, 'frames')
FRAME_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'interp')
SCENE_INPUT_DIR = os.path.join(PROJECT_DIR, 'scenes')
if not os.path.exists(SCENE_INPUT_DIR):
    os.mkdir(SCENE_INPUT_DIR)
ENV = [FRAME_INPUT_DIR, FRAME_OUTPUT_DIR]

"""Set Path Environment"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing working dir to {0}".format(dname))
os.chdir(os.path.dirname(dname))
print("Added {0} to temporary PATH".format(dname))
sys.path.append(dname)

"""Frame Count"""

print(f"Source, FPS: {SOURCE_FPS}, TARGET FPS: {TARGET_FPS}")
start_frame = int(START_FRAME)
if args.end:
    end_frame = args.end
process_round = args.round
last_batch = False
scene = 0


def extract_frames():
    print(f"\n\n\nExtract Frames between scenes: last_scene_time: {last_scene_time}, scene_time: {scene}, UHD: {UHD}")
    frame_input = os.path.join(FRAME_INPUT_DIR, "%08d.png")
    end_trim = f"-to {scene}" if not last_batch else ""
    extract_head = f'{ffmpeg} -hide_banner -hwaccel auto -ss {last_scene_time} {end_trim} -i {INPUT_FILEPATH} -vf'
    if UHD:
        os.system(
            f'{extract_head} zscale=matrixin=input:chromal=input:cin=input,format=yuv444p10le,format=rgb48be,format=rgb24{UHDcrop} {low_res} -q:v 2 -pix_fmt rgb24 "{frame_input}" -y')
    else:
        os.system(
            f'{extract_head} zscale=matrix=709:matrixin=709:chromal=input:cin=input,format=yuv444p10le,format=rgb48be,format=rgb24{HDcrop} {low_res} -q:v 2 -pix_fmt rgb24  "{frame_input}" -y')

    clean_false_frames(os.path.basename(FRAME_INPUT_DIR))
    return


def interpolation():
    print(f"Interpolation: UHD: {UHD}, last_scene_time: {last_scene_time}, scene_time: {scene}")
    os.system(
        f"python {RIFE_PATH} --img {FRAME_INPUT_DIR} --exp {EXP} {UHD} {accurate} {reverse_} --imgformat png --output {FRAME_OUTPUT_DIR} --cnt {args.interp_cnt} --start {args.interp_start} --model {args.model}")


def render():
    output_chunk_path = os.path.join(PROJECT_DIR,
                                     "chunk-{:0>3d}-{:0>4d}.mp4".format(CHUNK_CNT, scene_cnt))
    frame_output = os.path.join(FRAME_OUTPUT_DIR, "%08d.png")
    print(f"Render: UHD: {UHD}, last_scene_time: {last_scene_time}, scene: {scene}")
    ffmpeg_command = f'{ffmpeg}  -hide_banner -threads 0 -f image2 -r {TARGET_FPS} -i "{frame_output}" -r {TARGET_FPS} '
    if UHD:
        if not hwaccel:
            subprocess.Popen(
                ffmpeg_command + f'-c:v libx265 -preset {preset} -tune grain -profile:v main10 -pix_fmt yuv420p10le '
                                 f'-color_range tv -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc '
                                 f'-x265-params "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=0,0" '
                                 f'-crf {crf} "{output_chunk_path}" -y').wait()
        else:
            subprocess.Popen(
                ffmpeg_command + f'-c:v hevc_nvenc -preset hq -rc:v vbr_hq -cq:v {crf} -rc-lookahead:v 32 '
                                 f'-refs:v 16 -bf:v 3 -b:v 100M -b_ref_mode:v middle -profile:v main10 '
                                 f'-pix_fmt p010le '
                                 f'-color_range tv -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc "{output_chunk_path}" -y').wait()

    else:
        if not hwaccel:
            subprocess.Popen(
                ffmpeg_command + f'-c:v libx264 -pix_fmt yuv420p -preset {preset} -crf {crf} "{output_chunk_path}" -y').wait()
        else:
            subprocess.Popen(
                ffmpeg_command + f'-c:v h264_nvenc -preset hq -rc:v vbr_hq -cq:v {crf} -rc-lookahead:v 32 '
                                 f'-refs:v 16 -bf:v 3 -b:v 100M -b_ref_mode:v middle '
                                 f'-pix_fmt yuv420p -preset {preset} "{output_chunk_path}" -y').wait()


"""Scene Detection"""
for DIR in ENV:
    if not os.path.exists(DIR):
        os.mkdir(DIR)
scenes_data = {}
scene_json = os.path.join(PROJECT_DIR, "scenes.json")


def scdet_read():
    global scenes_data
    if os.path.exists(scene_json):
        with open(scene_json, "r") as r_:
            scenes_data = json.load(r_)
            print("Loaded Scenes")
            pprint(scenes_data)


scdet_read()


def scdet_dump():
    with open(scene_json, "w") as w:
        json.dump(scenes_data, w)


if not scenes_data:
    scene_txt = os.path.join(PROJECT_DIR, "scenes.txt")
    os.system(
        f'{ffmpeg} -hide_banner -hwaccel auto -i "{INPUT_FILEPATH}" -an  -frame_pts true  -lavfi scdet=t={scdet}:sc_pass=1,mpdecimate,setpts=N/FRAME_RATE/TB -q:v 9 -s 320x180 {os.path.join(SCENE_INPUT_DIR, "%08d.png")} > {scene_txt} 2>&1')
    with open(scene_txt, "r") as r:
        scene_find = r.read()
    scenes = re.findall(".*?lavfi\.scd\.time:\s{,5}([\d\.]+).*?", scene_find)

    scene_cnt = 1
    last_scene_time = 0
    for scene in scenes:
        if last_scene_time and scene - last_scene_time < 1:
            print(f"SCENE TOO NEAR and DROP!: {scene}")
            continue
        last_scene_time = scene
        scenes_data[str(scene_cnt)] = scene
        scene_cnt += 1
    print("Calculated Scenes")
    pprint(scenes_data)

scdet_dump()


def clean_env():
    for DIR_ in ENV:
        if os.path.exists(DIR_):
            shutil.rmtree(DIR_)
        if not os.path.exists(DIR_):
            os.mkdir(DIR_)


"""Start Sectioning"""
sections = []

last_scene_time = 0
# scenes_data = sorted(scenes_data.values(), key=lambda x: float(x))
scene_cnt = 1
round_ = 1

"""Instinct Procesd and Exit"""
if args.renderonly:
    render()
    sys.exit()

if args.rifeonly:
    interpolation()
    sys.exit()

last_start = 0  # timestamp, seconds
all_frames_cnt = 0
os.chdir(PROJECT_DIR)
last_false_png = None  # 非上一序列的错帧


def clean_false_frames(save_dir, ):
    global last_false_png

    png_list = sorted(os.listdir(save_dir), key=lambda x: int(x[:-4]))

    now_first_png = os.path.join(save_dir, png_list[0])
    now_first_2png = os.path.join(save_dir, png_list[1])
    if sum(Image.open(now_first_2png).convert("L").getextrema()) not in [0, 2]:
        result = ssim(now_first_png, now_first_2png, multichannel=True)
        if result == 1.0:
            """完全匹配"""
            os.remove(now_first_2png)
            print(f"Remove First False PNG: {now_first_2png}")
            png_list = sorted(os.listdir(save_dir), key=lambda x: int(x[:-4]))
            for ri in png_list[1:]:
                ri = os.path.join(save_dir, ri)
                new_path_id = int(os.path.basename(ri)[:-4]) - 1
                shutil.move(ri, os.path.join(save_dir, '{:0>8d}.png'.format(new_path_id)))
            png_list = sorted(os.listdir(save_dir), key=lambda x: int(x[:-4]))

    """处理上一图片序列的错帧情况"""
    if last_false_png is not None:
        result = ssim(now_first_png, last_false_png, multichannel=True)
        if result == 1.0:
            """完全匹配"""
            os.remove(last_false_png)
            print(f"Remove Last False PNG: {last_false_png}")
            last_false_png = None
        elif result > 0.95:
            """上一序列的最后一张图片是当前序列的第一张图片"""
            for ri in reversed(png_list):
                ri = os.path.join(save_dir, ri)
                new_path_id = int(os.path.basename(ri)[:-4]) + 1
                shutil.move(ri, os.path.join(save_dir, '{:0>8d}.png'.format(new_path_id)))
            shutil.move(last_false_png, os.path.join(save_dir, '{:0>8d}.png'.format(1)))
            print(f"Move Last False PNG: {last_false_png}")
            png_list = sorted(os.listdir(save_dir), key=lambda x: int(x[:-4]))
            last_false_png = None

    now_last_png = os.path.join(save_dir, png_list[-1])
    now_last_2png = os.path.join(save_dir, png_list[-2])

    if ssim(now_last_png, now_last_2png, multichannel=True) < 0.7:
        """最后一张图片非当前序列"""
        move_now_last_png = os.path.join(PROJECT_DIR, os.path.basename(now_last_png))
        shutil.move(now_last_png, move_now_last_png)
        last_false_png = move_now_last_png
        print(f"Find Last False PNG: {now_last_png}")

    return len(os.listdir(save_dir))


for scene_, scene in sorted(scenes_data.items(), key=lambda x: float(x[1])):
    scene_cnt = int(scene_)
    scene = float(scene)
    if last_scene_time and scene - last_scene_time < 1:
        print(f"\n\nSCENE TOO NEAR: No. {scene_cnt}, scene: {scene} <- last: {last_scene_time}\n\n")
        continue
    print(f"SCENE Process: No. {scene_cnt}, scene time: {scene}")
    extract_start_time = start_frame / SOURCE_FPS
    # extract_end_time = end_frame / SOURCE_FPS
    if scene_cnt < scene_start:
        """Not till target scene start """
        last_scene_time = scene
        continue
    if scene <= extract_start_time:
        """Not till end of previous section"""
        continue
    if round_ > process_round:
        print(f"Finish SCENE Process before: No. {scene_cnt}, scene time: {scene}")
        process_round = 0
        break

    clean_env()

    """Step directly into rendering"""

    start_time = datetime.datetime.now()

    """Extract Frames"""
    extract_frames()

    """interpolation"""
    interpolation()

    """render"""
    render()

    if args.pause:
        input("About to remove, Proceed?: ")

    """remove"""
    shutil.rmtree(FRAME_OUTPUT_DIR)
    shutil.rmtree(FRAME_INPUT_DIR)

    CHUNK_CNT += 1
    round_ += 1
    last_scene_time = scene

    end_time = datetime.datetime.now()
    print(f"Round {round_} finished by {end_time - start_time}")

if process_round:
    """process between last scene to last frame"""
    last_batch = True
    print(f"Processing Last Batch: last_scene_time: {last_scene_time}")
    """Step directly into rendering"""
    clean_env()
    extract_frames()
    interpolation()
    render()

print(f"Program finished at {datetime.datetime.now()}")
