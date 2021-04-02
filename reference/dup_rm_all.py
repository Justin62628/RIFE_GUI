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
import sys
import skvideo
from skvideo.io import FFmpegWriter, FFmpegReader
import math

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='对图片序列进行补帧')
parser.add_argument('--img', dest='img', type=str,
                    default='', help='图片路径')
parser.add_argument('--output', dest='output', type=str,
                    default='out', help='保存目录')
parser.add_argument('--start', dest='start', type=int,
                    default=0, help='从第start张图片开始补帧')
parser.add_argument('--resume', dest='resume',
                    action='store_true', help='自动计算count并恢复渲染')
parser.add_argument('--device_id', dest='device_id',
                    type=int, default=0, help='设备ID')
parser.add_argument('--model', dest='modelDir', type=str,
                    default='train_log', help='模型目录')
parser.add_argument('--fp16', dest='fp16',
                    action='store_true', help='FP16速度更快，质量略差')
parser.add_argument('--scale', dest='scale', type=float,
                    default=1.0, help='4K时建议0.5')
parser.add_argument('--rbuffer', dest='rbuffer',
                    type=int, default=0, help='读写缓存')
parser.add_argument('--predict_mode', dest='predict_mode', type=str,
                    default="safe", help="safe/performance/medium , 36HW/24HW/30HW")
parser.add_argument('--wthreads', dest='wthreads',
                    type=int, default=4, help='写入线程')
parser.add_argument('--redup', dest='redup',
                    action='store_true', help='去除重复帧并补足')
parser.add_argument('--dup', dest='dup', type=float, default=1.0, help='dup数值')
parser.add_argument('--scene', dest='scene', type=float,
                    default=50, help='场景识别阈值')
parser.add_argument('--rescene', dest='rescene', type=str,
                    default="mix", help="copy/mix   帧复制/帧混合")
parser.add_argument('--exp', dest='exp', type=int,
                    default=2, help='补2的exp次方-1帧')

args = parser.parse_args()
assert args.scale in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
spent = time.time()

if not os.path.exists(args.output):
    os.mkdir(args.output)

if args.device_id != -1:
    device = torch.device("cuda")
    torch.cuda.set_device(args.device_id)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
else:
    device = torch.device("cpu")
    try:
        from model_cpu.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from model_cpu.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")

model.eval()
model.device()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

start = args.start
if args.resume:
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
        num = item[0]
        content = item[1]
        cv2.imencode('.png', content[:, :, ::-1])[1].tofile('{}/{:0>9d}.png'.format(user_args.output, num))

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
    I0 = torch.from_numpy(np.transpose(im0, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = torch.from_numpy(np.transpose(im1, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I0 = pad_image(I0)
    I1 = pad_image(I1)
    global model
    middle = model.inference(I0, I1, args.scale)
    I0 = 0
    I1 = I0
    mid = (((middle[0] * 255.).byte().cpu().numpy().transpose(1,2, 0)))[:h, :w][:, :, ::-1][:, :, ::-1].copy()
    if exp == 1:
        return [mid]
    first_half = make_inference(im0, mid, exp=exp - 1)
    second_half = make_inference(mid, im1, exp=exp - 1)
    return [*first_half, mid, *second_half]

def rescene(im0,im1,REQ):
    out = []
    if args.rescene == "mix":
        step = 1 / (REQ+1)
        alpha = 0
        for _ in range(REQ):
            alpha += step
            beta = 1-alpha
            mix = cv2.addWeighted(
                im0[:, :, ::-1], alpha, im1[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
            out.append(mix)
    else:
        for _ in range(REQ):
            out.append(im0[:, :, ::-1])
    return out

def drop(EXP,req):
    I_step = 1 / (2 ** EXP)
    IL = [x*I_step for x in range(1,2**EXP)]
    N_step = 1 / (req + 1)
    NL = [x*N_step for x in range(1,req+1)]
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

def calc_diff(im0,im1):
    try:
        return cv2.absdiff(im0[:, :, ::-1], im1[:, :, ::-1]).mean()
    except:
        return -1
    return -1

tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=(tot_frame * (2**args.exp) -1))
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
for _ in range(args.wthreads):
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))

cnt = 0
cnt = 0 if start == 0 else (start - 1) * (2 ** args.exp) + 1
cnt += 1

while True:
    frame = read_buffer.get()
    if frame is None:
        break

    skip = 0 #用于记录跳过的帧数

    diff = calc_diff(lastframe,frame)

    #帧读取完了，lastframe或frame会变成none type，值返回-1 跳出循环
    if diff == -1:
        print('...')
        break

    output = []
    post = []
    post.append(lastframe) #放入起始帧

    if diff > args.scene:
        #转场（这一块要改）(我觉得应该放到正常补帧队列里)
        output = rescene(frame,lastframe,(2**args.exp-1))

        #补出来的转场放到post中
        for mid in output:
            post.append(mid)

    else:

        if diff < args.dup:

            #一直向下一帧，直到非重复帧位置，阈值为args.dup
            while diff < args.dup:
                skip += 1
                frame = read_buffer.get()
                diff = calc_diff(lastframe,frame)

            #除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
            if diff > args.scene:
                #转场（尽量换方差法）
                output = rescene(frame,lastframe,skip)
                for mid in output:
                    post.append(mid)
            else:
                #推导 exp
                exp = int(math.log(skip,2))+1 
                output = make_inference(lastframe, frame, exp)
                kpl = drop(exp,skip) #打表（exp，需要的帧数）
                for x in kpl:
                    post.append(output[x])
    output = [] #丢弃output的值
    

    #post进行到正常补帧序列（可修改）
    write_buffer.put([cnt,lastframe])
    cnt += 1
    i0 = 0
    lp = len(post)
    for i1 in range(1,lp):
        output = make_inference(post[i0],post[i1],args.exp)
        for mid in output:
            write_buffer.put([cnt,mid])
            cnt += 1
            pbar.update(1)
        i0 = i1

    lastframe = frame #起始帧 = 结尾帧

write_buffer.put([cnt, lastframe]) #程序结束时放入最后一帧


pbar.update(1)
while(not os.path.exists('{}/{:0>9d}.png'.format(args.output, cnt))):
    time.sleep(1)
pbar.close()
print("spent {}s".format(time.time()-spent)) 