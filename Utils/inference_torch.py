import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import psutil
import time
from queue import Queue, Empty
from skvideo.io import FFmpegWriter
import sys
import skvideo

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='对图片序列进行补帧')
parser.add_argument('--img', dest='img', type=str, default='input',help='图片目录')
parser.add_argument('--output', dest='output', type=str, default='out',help='保存目录')
parser.add_argument('--start',dest='start',type=int,default=0,help='从第start张图片开始补帧')
parser.add_argument('--resume', dest='resume', action='store_true', help='自动计算count并恢复渲染')

parser.add_argument('--device_id',dest='device_id',type=int,default=0, help='设备ID')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='模型目录')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='FP16速度更快，质量略差')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='4K时建议0.5')
parser.add_argument('--rbuffer', dest='rbuffer', type=int, default=0,help='读写缓存')
parser.add_argument('--predict_mode', dest='predict_mode', type=str, default="safe",help = "safe/performance/medium , 36HW/24HW/30HW")
parser.add_argument('--wthreads', dest='wthreads', type=int, default=4,help='写入线程')

parser.add_argument('--out_video', dest='out_video', action='store_true', help='直接输出为视频')
parser.add_argument('--out_format', dest='out_format', type=str, default='mp4', help='视频格式')
parser.add_argument('--preset', dest='preset', type=str, default="slow", help="压制预设，medium以下可用于收藏。硬件加速推荐hq")
parser.add_argument('--crf', dest='crf', type=int, default=16, help="恒定质量控制，12以下可作为收藏，16能看，默认16")
parser.add_argument('--read_fps', dest='read_fps', type=float, default=23.976,help='读取帧率')
parser.add_argument('--output_fps', dest='output_fps', type=float, default=47.952,help='导出帧率')
parser.add_argument('--HDR', dest='HDR', action='store_true', help='开启HDR模式')
parser.add_argument('--hwaccel', dest='hwaccel', action='store_true', help='开启硬件加速')
parser.add_argument('--audio', dest='audio', type=str, default="", help="音频文件")
parser.add_argument('--ffmpeg', dest='ffmpeg', type=str, default='ffmpeg.exe', help='ffmpeg路径')

parser.add_argument('--scene', dest='scene', type=float, default=50,help='场景识别阈值')
parser.add_argument('--rescene', dest='rescene', type=str, default="mix",help="copy/mix   帧复制/帧混合")
parser.add_argument('--exp', dest='exp', type=int, default=1,help='补2的exp次方-1帧')

args = parser.parse_args()
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
spent = time.time()
#skvideo.setFFmpegPath(args.ffmpeg)

if not os.path.exists(args.output) and not args.out_video:
    os.mkdir(args.output)

model = None
if args.device_id != -1:
    device = torch.device("cuda")
    torch.cuda.set_device(args.device_id)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)    
    try:
        from Utils.model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from Utils.model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
    if model is None:
        sys.exit()

model.eval()
model.device()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

def generate_frame_renderer(output_path):
        input_dict = {"-vsync": "0", "-r": f"{args.read_fps}"}
        output_dict = {"-vsync": "0", "-r": f"{args.output_fps}", "-preset": args.preset}
        if args.HDR:
            if not args.hwaccel:
                output_dict.update({"-c:v": "libx265",
                                    "-tune": "grain",
                                    "-profile:v": "main10",
                                    "-pix_fmt": "yuv420p10le",
                                    "-x265-params": "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=0,0",
                                    "-crf": f"{args.crf}",
                                    })
            else:
                output_dict.update({"-c:v": "hevc_nvenc",
                                    "-rc:v": "vbr_hq",
                                    "-profile:v": "main10",
                                    "-pix_fmt": "p010le",
                                    "-cq:v": f"{args.crf}",
                                    })
        else:
            if not args.hwaccel:
                output_dict.update({"-c:v": "libx264",
                                    "-tune": "grain",
                                    "-pix_fmt": "yuv420p",
                                    "-crf": f"{args.crf}",
                                    })
            else:
                output_dict.update({"-c:v": "h264_nvenc",
                                    "-rc:v": "vbr_hq",
                                    "-pix_fmt": "yuv420p",
                                    "-cq:v": f"{args.crf}",
                                    })
        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)

start = args.start
if args.resume:
    if args.out_video:
        print("暂未实现直接导出为视频的恢复渲染功能...")
        sys.exit(0)
    maxc = 0
    for f in os.listdir(args.output):
        tempcnt = int(os.path.splitext(f)[0])
        if tempcnt > maxc:
            maxc = tempcnt
    if maxc != 0:
        start = int(((maxc - 1) / (2 ** args.exp))) + 1
    start += args.start

videogen = [f for f in os.listdir(args.img)]
tot_frame = len(videogen)
if start != 0:
    templist = []
    pos = start - 1
    end = len(videogen)
    while pos != end:
        templist.append(videogen[pos])
        pos = pos + 1
    videogen = templist
passed = tot_frame - len(videogen)
videogen.sort()
lastframe = cv2.imdecode(np.fromfile(os.path.join(args.img, videogen[0]), dtype=np.uint8), 1)[:, :, ::-1].copy()
videogen = videogen[1:]
h, w, _ = lastframe.shape
    
def clear_write_buffer(user_args, write_buffer):
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if args.out_video:
            global frame_writer
            frame_writer.writeFrame(item[1])
        else:
            num = item[0]
            content = item[1]
            cv2.imencode('.png', content[:, :, ::-1])[1].tofile('{}/{:0>9d}.png'.format(user_args.output,num))
            #cnt += 1

def build_read_buffer(dir_path, read_buffer, videogen):
    try:
        for frame in videogen:
             frame = cv2.imdecode(np.fromfile(os.path.join(dir_path, frame), dtype=np.uint8), 1)[:, :, ::-1].copy()
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

def make_inference(im0, im1, exp):
    I0 = torch.from_numpy(np.transpose(im0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = torch.from_numpy(np.transpose(im1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I0 = pad_image(I0)
    I1 = pad_image(I1)
    global model
    middle = model.inference(I0, I1, args.scale)
    I0 = 0
    I1 = I0
    mid = (((middle[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))[:h, :w][:, :, ::-1][:, :, ::-1].copy()
    if exp == 1:
        return [mid]
    first_half = make_inference(im0, mid, exp=exp - 1)
    second_half = make_inference(mid, im1, exp=exp - 1)
    return [*first_half, mid, *second_half]

tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
pbar.update(passed)
rb = args.rbuffer
wb = rb
if rb < 1:
    ram = float(psutil.virtual_memory().free)
    print("ramsize:{}".format(ram))
    try:
        num = 36
        if args.predict_mode == "medium":
            num = 30
        elif args.predict_mode == "performence":
            num = 24
        wb = rb = int((0.9*ram) / (num * h * w))
    except:
        wb = rb = 100
if rb < 1:
    rb = 2
    wb = 1
read_buffer = Queue(maxsize=rb)
write_buffer = Queue(maxsize=wb)
print("IO_buffer_size:{}".format(rb))
frame_writer = 0
_thread.start_new_thread(build_read_buffer, (args.img, read_buffer, videogen))
if args.out_video:
    frame_writer = generate_frame_renderer(args.output)
    _thread.start_new_thread(clear_write_buffer, (args,write_buffer))
else:
    for _ in range(args.wthreads):
        _thread.start_new_thread(clear_write_buffer, (args,write_buffer))

cnt = 0
cnt = 0 if start == 0 else (start - 1) * (2 ** args.exp) + 1
cnt += 1

while True:
    frame = read_buffer.get()
    if frame is None:
        break
    diff = cv2.absdiff(lastframe[:, :, ::-1], frame[:, :, ::-1]).mean()
    if diff > args.scene:
        output = []
        if args.rescene == "mix":
            step = 1 / (2 ** args.exp)
            alpha = 0
            for _ in range((2 ** args.exp) - 1):
                alpha += step
                beta = 1-alpha
                mix = cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
                output.append(mix)
        else:
            for _ in range((2 ** args.exp) - 1):
                output.append(frame)
    else:
        output = make_inference(lastframe, frame, args.exp)
    write_buffer.put([cnt,lastframe])
    for mid in output:
        write_buffer.put([cnt,mid])
        cnt += 1
    pbar.update(1)
    lastframe = frame
write_buffer.put([cnt,lastframe])
if not args.out_video:
    while(not os.path.exists('{}/{:0>9d}.png'.format(args.output,cnt))):
        time.sleep(1)
pbar.close()
print("spent {}s".format(time.time()-spent))