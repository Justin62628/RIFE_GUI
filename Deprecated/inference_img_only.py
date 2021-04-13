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
import traceback
import json
import datetime
import math
import shutil

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', type=str, default=None, help="interp input path")
parser.add_argument('--output', required=True, dest='output', type=str)
parser.add_argument('--reverse', dest='reverse', action='store_true', help='Reversed Optical Flow')
parser.add_argument('--accurate', dest='accurate', action='store_true', help='Accurate Design RIFE(Beta)')
parser.add_argument('--remove-dup', dest='remove_dup', action='store_true',
                    help='Generate Removed Duplicated Frames Sequence')
parser.add_argument('--fp16', dest='fp16', action='store_true',
                    help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--fps', dest='fps', type=int, default=None, help="Source fps")
parser.add_argument('--exp', dest='exp', type=int, default=1, help="output fps = fps * 2 ** exp")
parser.add_argument('--scale', dest='scale', type=float, default=1.0,
                    help='actual times of scaled resolution: scale = 2 => 1080p -> 4K to achieve more accurate interpolation')
parser.add_argument('--imgformat', default="png")
parser.add_argument('--start', dest='start', type=int, default=0, help="start pts of input dir")
parser.add_argument('--end', dest='end', type=int, default=0)
parser.add_argument('--cnt', dest='cnt', type=int, default=1)
parser.add_argument('--thread', dest='thread', type=int, default=8, help="Write Buffer Thread")
# TODO: model select
parser.add_argument('--model', dest='model', type=int, default=2, help="Select RIFE Modle, default v2")
parser.add_argument('--ncnn', dest='ncnn', action='store_true', help='Appoint NCNN interpolation which supports AMD card')

args = parser.parse_args()
# assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
project_dir = os.path.dirname(args.img)

rendering_flag = True

class InterpArgs:
    def __init__(self, args_=None):
        pass
    img = args.img


class RifeInterpolation:
    def __init__(self):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("use cpu to interpolate")
        else:
            self.device = torch.device("cuda")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if args.fp16:
                try:
                    torch.set_default_tensor_type(torch.cuda.HalfTensor)
                    print("FP16 mode switch success")
                except Exception as e:
                    print("FP16 mode switch failed")
                    traceback.print_exc()
                    args.fp16 = False

        torch.set_grad_enabled(False)

        if args.img:
            args.png = True

        if args.model == 1:
            from Utils import Model
        else:
            from Utils import Model

        self.model = Model()
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log')

        # Pls Manually Mkdir for train_log\1.8, train_log\2.0 to store datasets for RIFE
        self.model.load_model(self.model_path, -1)
        print(f"Load model at {self.model_path}")
        self.model.eval()
        self.model.device()
        self.user_args = args
        self.input = args.img
        self.output = args.output
        self.padding = (0,0,0,0)
        self.input_start = args.start
        self.interp_start = args.cnt
        self.input_end = args.end
        pass

    def clear_write_buffer(self, user_args, write_buffer_, thread_id, to_input=False):
        # Threading
        thread_png_cnt = 1
        output_dir = user_args.output if not to_input else user_args.img
        while True:
            item = write_buffer_.get()
            if item is None:
                print(f"[T{thread_id}] Found None write_buffer at {thread_png_cnt}, break")
                break
            _frame_cnt = item[0]
            frame_ = item[1]
            cv2.imwrite("{}/{:0>8d}.png".format(output_dir, _frame_cnt), frame_[:, :, ::-1])
            thread_png_cnt += 1

    def build_read_buffer(self, user_args, read_buffer_, video_open_):
        video_open_.sort(key=lambda x: int(x[:-4]))
        for frame_ in video_open_:
            path = os.path.join(user_args.img, frame_)
            frame_ = cv2.imread(path)[:, :, ::-1].copy()
            read_buffer_.put((path, frame_))
        read_buffer_.put(None)

    def make_inference(self, I0, I1, exp, sec_batch=False):
        if args.reverse:
            middle = self.model.inference(I1, I0, args.scale)
        else:
            middle = self.model.inference(I0, I1, args.scale)

        if exp == 1:
            return [middle]
        # interpolation progression
        first_half = self.make_inference(I0, middle, exp=exp - 1, sec_batch=True)
        second_half = self.make_inference(middle, I1, exp=exp - 1, sec_batch=True)
        # return 3 imgs
        return [*first_half, middle, *second_half]

    def check_is_rendering(self):
        global rendering_flag
        status_json_path = os.path.join(project_dir, "interp_status.json")
        if os.path.exists(status_json_path):
            with open(status_json_path, "r", encoding="utf-8") as r:
                status_json = json.load(r)
            if bool(status_json["interp"]):
                if rendering_flag:
                    print(f"Detect Interp signal, postpone interpolation at {datetime.datetime.now()}")
                    rendering_flag = False
                return True
            else:
                if not rendering_flag:
                    print(f"Detect Pause signal, postpone interpolation at {datetime.datetime.now()}")
                    rendering_flag = True
                return False
        else:
            return True
        pass

    def generate_padding(self, img):
        """

        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        tmp = max(32, int(32 / args.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        self.padding = (0, pw - w, 0, ph - h)
        return h,w
        pass

    def generate_torch_img(self, img):
        """

        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        img_torch = torch.from_numpy(np.transpose(img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(
            0).float() / 255.
        return self.pad_image(img_torch)

    def complete_run(self, input_dir, output_dir):
        """
        from img folder to interp folder
        :return:
        """

        # Read Source PNG sequence
        video_open = []
        img_list = sorted(os.listdir(input_dir), key=lambda x: int(x[:-4]))
        if not self.input_end:
            # select the last img as the end of batch
            self.input_end = int(img_list[-1][:-4])
        for f in img_list:
            if 'png' in f:
                if self.input_start <= int(f[:-4]) <= self.input_end:
                    video_open.append(f)
        total_frame = len(video_open)  # length of sequence, not counting in the first
        if not len(video_open):
            print(f"Find No Frames in {os.path.basename(input_dir)} to Interpolate: {len(os.listdir(args.img))}")
        video_open.sort(key=lambda x: int(x[:-4]))
        lastframe_path = os.path.join(args.img, video_open[0])
        lastframe = cv2.imread(lastframe_path)[:, :, ::-1].copy()  # read first img
        video_open = video_open[1:]  # deposit the others
        # mkdir for output pngs
        if not os.path.exists(output_dir):
            print("Manually make dir for png output")
            os.mkdir(output_dir)

        h,w = self.generate_padding(lastframe)

        write_buffer = Queue(maxsize=1000)
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self.build_read_buffer, (self.user_args, read_buffer, video_open))

        for x in range(args.thread):
            _thread.start_new_thread(self.clear_write_buffer, (self.user_args, write_buffer, x))

        # Feed first frame to torch
        I1 = self.generate_torch_img(lastframe)

        print(f"Start Interpolation from {os.path.basename(input_dir)} to {os.path.basename(output_dir)}, --reverse: {args.reverse}, --accurate: {args.accurate}, --model: {args.model}")

        pbar = tqdm(total=total_frame)
        frame_cnt = args.cnt
        while True:
            if not self.check_is_rendering():
                time.sleep(1)
                continue
            frame = read_buffer.get()  # path, cv.read()[]
            if frame is None:
                print(f"Read Buffer get None, Break")
                break
            frame_path = frame[0]
            frame = frame[1]

            I0 = I1  # I0 is start frame before frame from read_buffer
            I1 = self.generate_torch_img(frame)

            output = self.make_inference(I0, I1, args.exp)
            # put last_frame(1 -> first frame to write, identically copy)
            write_buffer.put((frame_cnt, lastframe))
            frame_cnt += 1
            pool_cnt = 1
            for mid in output:
                # for exp = 2, len(output) = 3
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put((frame_cnt, mid[:h, :w]))
                frame_cnt += 1
                pool_cnt += 1
            pbar.set_description("Interpolate at {}, {:0>8d}".format(os.path.basename(lastframe_path), frame_cnt))
            pbar.update(1)
            lastframe = frame
            lastframe_path = frame_path

        write_buffer.put((frame_cnt, lastframe))
        for i in range(args.thread):
            write_buffer.put(None)
        while not write_buffer.empty():
            time.sleep(0.1)
        pbar.close()
        print(f"\n\nfrom {os.path.basename(input_dir)} to {os.path.basename(output_dir)}\nInterpolation is over\n\n")

    def pad_image(self, img):
        if args.fp16:
            return F.pad(img, self.padding).half()
        else:
            return F.pad(img, self.padding)

    def run(self):
        pass


class NCNNinterpolator:
    def __init__(self):
        self.rife_ncnn_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rife-ncnn")
        self.rife_ncnn = os.path.join(self.rife_ncnn_root, "rife-ncnn-vulkan.exe")
        self.exp = args.exp
        self.input_list = list()
        self.input_root = os.path.dirname(args.img)
        self.generate_input_list()
        print(f"Use NCNN to interpolate from {self.rife_ncnn_root} with input list {str(self.input_list)}")
        pass

    def generate_input_list(self):
        self.input_list.append(args.img)
        for e in range(self.exp-1):
            new_input = os.path.join(self.input_root, f"mid_interp_{e + 1}x")
            self.input_list.append(new_input)
            if not os.path.exists(new_input):
                os.mkdir(new_input)
        self.input_list.append(os.path.join(self.input_root, "interp"))

        pass

    def ncnn_interpolate(self):
        dir_cnt = 1
        for input_dir in self.input_list:
            if os.path.basename(input_dir) == "interp":
                break
            create_command = f"{self.rife_ncnn}  -i {input_dir} -o {self.input_list[dir_cnt]} -m {os.path.join(self.rife_ncnn_root, 'rife-v2.4')}"
            os.system(create_command)
            print(f"Round {os.path.basename(self.input_list[dir_cnt])} finished")
            if input_dir != self.input_list[0]:
                """Erase all mid interpolations except the img input"""
                shutil.rmtree(input_dir)
            dir_cnt += 1
        pass




class GenerateGapFrames(RifeInterpolation):

    def __init__(self):
        super(GenerateGapFrames, self).__init__()
        self.pair_path = os.path.join(project_dir, "dup_json.json")
        self.available_flag = True
        if not os.path.exists(self.pair_path):
            print("Not found dup_json, break first round")
            self.available_flag = False
            return
        self.interp_t = dict()
        self.real_t = dict()
        self.pos_map = dict()
        self.generate_prebuild_map()
        pass

    def generate_prebuild_map(self):
        for gap in range(2, 8 + 1):
            exp = round(math.sqrt(gap - 1))
            interp_t = {i: (i + 1) / (2 ** exp) for i in range(2 ** exp - 1)}
            real_t = {i: (i + 1) / gap for i in range(gap - 1)}
            for rt in real_t:
                min_ans = (99999999, -1)
                for inp in interp_t:
                    tmp_ = abs(interp_t[inp] - real_t[rt])
                    if tmp_ <= min_ans[0]:
                        min_ans = (tmp_, inp)
                self.pos_map[(gap, rt)] = min_ans[1]
                pass
        # print(self.pos_map)

    def get_png_path(self, png_cnt):
        return os.path.join(args.img, "{:0>8d}.png".format(png_cnt))
        pass

    def generate_gap_frames(self):
        with open(self.pair_path, "r", encoding="utf-8") as r:
            pair_json = json.load(r)
        print(f"Start interpolate duplicated frames from {self.pair_path}")

        write_buffer = Queue(maxsize=1000)
        for x in range(args.thread):
            _thread.start_new_thread(self.clear_write_buffer, (self.user_args, write_buffer, x, True))

        pbar = tqdm(total=len(pair_json))
        for pair in pair_json:
            pair = list(map(lambda x: int(x), pair.strip("(").strip(")").split(",")))
            # TODO what's wrong with regex
            exp = round(math.sqrt(int(pair[1]) - int(pair[0])))
            self.pair_interpolate(pair, exp, write_buffer)
            pbar.set_description("Interpolate for {}".format(str(pair)))
            pbar.update(1)
            pass

        for x in range(args.thread):
            write_buffer.put(None)

        while not write_buffer.empty():
            time.sleep(0.2)
        pbar.close()
        print("Duplicate Frames interpolated")


        pass

    def pair_interpolate(self, pair, exp, write_buffer):
        img1_path = self.get_png_path(pair[0])
        img2_path = self.get_png_path(pair[1])
        img1 = cv2.imread(img1_path)[:, :, ::-1].copy()  # read first img
        img2 = cv2.imread(img2_path)[:, :, ::-1].copy()  # read first img
        h,w = self.generate_padding(img1)
        I0_ = self.generate_torch_img(img1)
        I1_ = self.generate_torch_img(img2)
        mid_inference = self.make_inference(I0_, I1_, exp)
        cnt = pair[0] + 1
        gap = pair[1] - pair[0]

        for cnt_i in range(gap - 1):
            # for exp = 2, len(output) = 3
            mid_ = mid_inference[self.pos_map[(gap, cnt_i)]]
            mid_ = ((mid_[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))[:h, :w]
            write_buffer.put((cnt, mid_))
            # cv2.imwrite("{}/{:0>8d}.png".format(args.img, cnt), mid_[:, :, ::-1])
            cnt += 1

        pass


if __name__ == "__main__":
    import sys

    if args.remove_dup:
        ggf = GenerateGapFrames()
        if ggf.available_flag:
            ggf.generate_gap_frames()
        sys.exit()

    if args.ncnn:
        ni = NCNNinterpolator()
        ni.ncnn_interpolate()
        sys.exit()

    ri = RifeInterpolation()
    ri.complete_run(args.img, args.output)
