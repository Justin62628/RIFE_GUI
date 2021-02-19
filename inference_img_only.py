import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue
import time

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--output', required=True, dest='output', type=str)
parser.add_argument('--img', dest='img', type=str, default=None, help="interp output path")
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--reverse', dest='reverse', action='store_true', help='Reversed Optical Flow')
parser.add_argument('-a', '--accurate', dest='accurate', action='store_true', help='Accurate Design RIFE(Beta)')
parser.add_argument('--fps', dest='fps', type=int, default=None, help="Source fps")
parser.add_argument('--exp', dest='exp', type=int, default=1, help="2 ** exp")
parser.add_argument('--imgformat', default="png")
#
parser.add_argument('--start', dest='start', type=int, default=1)
parser.add_argument('--end', dest='end', type=int, default=0)
parser.add_argument('--cnt', dest='cnt', type=int, default=1)
parser.add_argument('--thread', dest='thread', type=int, default=5, help="Write Buffer Thread")
parser.add_argument('--model', dest='model', type=int, default=2, help="Select RIFE Modle, default v2")

args = parser.parse_args()


def clear_write_buffer(user_args, write_buffer_):
    # Threading
    cnt = user_args.cnt
    while True:
        item = write_buffer_.get()
        if item is None:
            print(f"Found None write_buffer at {cnt}, break")
            break
        path = item[0]
        frame_ = item[1]
        # print('{} => {:0>8d}.png'.format(os.path.basename(path), cnt))
        cv2.imwrite("{}/{:0>8d}.png".format(user_args.output, cnt), frame_[:, :, ::-1])
        cnt += 1


def build_read_buffer(user_args, read_buffer_, video_open_):
    video_open_.sort(key=lambda x: int(x[:-4]))
    for frame_ in video_open_:
        path = os.path.join(user_args.img, frame_)
        frame_ = cv2.imread(path)[:, :, ::-1].copy()
        read_buffer_.put((path, frame_))
    read_buffer_.put(None)


def make_inference(I0, I1, exp, sec_batch=False):
    # Interpolation Process
    global model
    if args.accurate and not sec_batch:
        middle_backward = model.inference(I1, I0, args.UHD)
        middle_forward = model.inference(I0, I1, args.UHD)
        fs0 = cv2.absdiff(I0, middle_forward).mean()
        bs0 = cv2.absdiff(I0, middle_backward).mean()
        fs1 = cv2.absdiff(middle_forward, I1).mean()
        bs1 = cv2.absdiff(middle_backward, I1).mean()
        fow = (fs0 + fs1) / 2
        bac = (bs0 + bs1) / 2
        print(f"First Trial: fow: {fow}, bac: {bac}")
        if fow < bac:
            middle = middle_forward
        else:
            middle = middle_backward
    else:
        if args.reverse:
            middle = model.inference(I1, I0, args.UHD)
        else:
            middle = model.inference(I0, I1, args.UHD)
    if exp == 1:
        return [middle]
    # interpolation progression
    first_half = make_inference(I0, middle, exp=exp - 1, sec_batch=True)
    second_half = make_inference(middle, I1, exp=exp - 1, sec_batch=True)
    # return 3 imgs
    return [*first_half, middle, *second_half]

if args.img:
    args.png = True

if args.model == 1:
    from model.RIFE_HD import Model
else:
    from model.RIFE_HDv2 import Model

model = Model()
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log')

# Pls Manually Mkdir for train_log\1.8, train_log\2.0 to store datasets for RIFE
model_v1 = os.path.join(model_path, "1.8")
model_v2 = os.path.join(model_path, "2.0")
if os.path.exists(model_v1) and args.model == 1:
    model.load_model(model_v1, -1)
elif os.path.exists(model_v2) and args.model == 2:
    model.load_model(model_v2, -1)
else:
    model.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log'), -1)
model.eval()
model.device()

# Read Source PNG sequence
video_open = []

img_list = sorted(os.listdir(args.img), key=lambda x: int(x[:-4]))
if not args.end:
    # select the last img as the end of batch
    args.end = int(img_list[-1][:-4])
for f in img_list:
    if 'png' in f:
        if args.start <= int(f[:-4]) <= args.end:
            video_open.append(f)
total_frame = len(video_open)  # length of sequence, not counting in the first
video_open.sort(key=lambda x: int(x[:-4]))
lastframe_path = os.path.join(args.img, video_open[0])
lastframe = cv2.imread(lastframe_path)[:, :, ::-1].copy()  # read first img
video_open = video_open[1:]  # deposit the others

h, w, _ = lastframe.shape

# mkdir for output pngs
if not os.path.exists(args.output):
    print("Manually make dir for png output")
    os.mkdir(args.output)

if args.UHD:
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
else:
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
skip_frame = 1
# write_buffer = multiprocessing.Queue()
if __name__ == "__main__":
    write_buffer = Queue(maxsize=1000)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, video_open))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))

    # Feed first frame to torch
    I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = F.pad(I1, padding)

    print(f"Start Interpolation, --reverse: {args.reverse}, --UHD: {args.UHD}, --accurate: {args.accurate}, --model: {args.model}")
    pbar = tqdm(total=total_frame)

    frame_cnt = 1
    while True:
        frame = read_buffer.get()  # path, cv.read()[]
        if frame is None:
            print(f"Read Buffer get None, Break")
            break
        frame_path = frame[0]
        frame = frame[1]

        I0 = I1  # I0 is start frame before frame from read_buffer
        # Feed next frame to torch
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = F.pad(I1, padding)

        output = make_inference(I0, I1, args.exp)
        # put last_frame(1 -> first frame to write, identically copy)
        write_buffer.put((lastframe_path, lastframe))
        pool_cnt = 1
        for mid in output:
            # for exp = 2, len(output) = 3
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put((lastframe_path, mid[:h, :w]))
            pool_cnt += 1
        pbar.set_description("Interpolate at %s" % os.path.basename(lastframe_path))
        pbar.update(1)
        lastframe = frame
        lastframe_path = frame_path

    pool_cnt = 1
    for i in range(2 ** args.exp):
        write_buffer.put((lastframe_path, lastframe))
        pool_cnt += 1

    write_buffer.put(None)
    while not write_buffer.empty():
        time.sleep(0.1)
    pbar.close()

    print("Interpolation is over\n\n\n\n")
