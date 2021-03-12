import os
import shutil
import traceback
import warnings

import numpy as np
import torch
from torch.nn import functional as F

from model.RIFE_HDv2 import Model

warnings.filterwarnings("ignore")


class InterpArgs:
    def __init__(self):
        self.img = ""
        self.output = ""
        self.reverse = False
        self.accurate = False
        self.remove_dup = False
        self.fp16 = False
        self.fps = 24000 / 1001
        self.exp = 1
        self.scale = 1.0
        self.imgformat = "png"
        self.start = 0
        self.end = 0
        self.cnt = 1
        self.thread = 8
        self.model = ""
        self.ncnn = False
        self.png = False
        self.use_multi_card = False
        self.use_cpu = False
        self.use_specific_gpu = 0


args = InterpArgs()


class RifeInterpolation:
    def __init__(self, __args):
        self.initiated = False
        if __args is not None:
            self.args = __args
        else:
            self.args = InterpArgs()
        pass

    def initiate_rife(self, __args=None):
        if __args is not None:
            """Update Args"""
            self.args = __args
        if self.initiated:
            return
        # if self.args.use_cpu:
        #     self.device = torch.device("cpu")
        # elif not torch.cuda.is_available():
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("use cpu to interpolate")
        elif self.args.use_specific_gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.args.use_specific_gpu}"
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if self.args.fp16:
            try:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
                print("FP16 mode switch success")
            except Exception as e:
                print("FP16 mode switch failed")
                traceback.print_exc()
                self.args.fp16 = False

        torch.set_grad_enabled(False)
        self.model = Model()
        if self.args.model == "":
            self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log')
        else:
            self.model_path = self.args.model
        self.model.load_model(self.model_path, -1)
        print(f"Load model at {self.model_path}")
        self.model.eval()
        self.model.device()
        self.initiated = True

    def make_inference(self, i1, i2, scale, exp):
        if self.args.reverse:
            middle = self.model.inference(i2, i1, scale)
        else:
            middle = self.model.inference(i1, i2, scale)

        if exp == 1:
            return [middle]
        # interpolation progression
        first_half = self.make_inference(i1, middle, scale, exp=exp - 1)
        second_half = self.make_inference(middle, i2, scale, exp=exp - 1)
        return [*first_half, middle, *second_half]

    def generate_padding(self, img):
        """

        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        tmp = max(32, int(32 / self.args.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        return padding, h, w

    def generate_torch_img(self, img, padding):
        """
        :param img: cv2.imread [:, :, ::-1]
        :param padding:
        :return:
        """
        try:
            img_torch = torch.from_numpy(np.transpose(img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(
                0).float() / 255.
            return self.pad_image(img_torch, padding)
        except Exception as e:
            print(img)
            traceback.print_exc()
            raise e

    def pad_image(self, img, padding):
        if self.args.fp16:
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    def generate_interp(self, img1, img2, exp, scale, debug=False):
        """

        :param img1: cv2.imread
        :param img2:
        :param exp:
        :param scale:
        :return: list of interp cv2 image
        """
        if debug:
            output_gen = list()
            for i in range(2 ** exp):
                output_gen.append(img1)
            return output_gen
        padding, h, w = self.generate_padding(img1)
        i1 = self.generate_torch_img(img1, padding)
        i2 = self.generate_torch_img(img2, padding)
        interp_gen = self.make_inference(i1, i2, scale, exp)
        output_gen = list()
        for mid in interp_gen:
            mid = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))
            output_gen.append(mid[:h, :w])
        return output_gen

    def run(self):
        pass


class NCNNinterpolator:

    def __init__(self, __args):
        if __args is not None:
            self.args = __args
        else:
            self.args = InterpArgs()
        # same with this file
        self.rife_ncnn_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rife-ncnn")
        self.rife_ncnn = os.path.join(self.rife_ncnn_root, "rife-ncnn-vulkan.exe")
        self.exp = self.args.exp
        self.input_list = list()
        self.input_root = os.path.dirname(self.args.img)
        self.generate_input_list()
        print(f"Use NCNN to interpolate from {self.rife_ncnn_root} with input list {str(self.input_list)}")
        pass

    def initiate_rife(self):
        pass

    def generate_input_list(self):
        self.input_list.append(self.args.img)
        for e in range(self.exp - 1):
            new_input = os.path.join(self.input_root, f"mid_interp_{e + 1}x")
            self.input_list.append(new_input)
            if not os.path.exists(new_input):
                os.mkdir(new_input)
        self.input_list.append(os.path.join(self.input_root, "interp"))

        pass

    def supervise_ncnn(self):
        """
        Supervise Interpolation process, detect output dir to check
        :return:
        """
        # TODO Supervise NCNN

    def ncnn_interpolate(self):
        dir_cnt = 1
        for input_dir in self.input_list:
            if os.path.basename(input_dir) == "interp":
                break
            # TODO manually adjust
            create_command = f"{self.rife_ncnn}  -i {input_dir} -o {self.input_list[dir_cnt]} -m {os.path.join(self.rife_ncnn_root, 'rife-v2.4')} -j 2:4:4"
            os.system(create_command)
            print(f"[NCNN] Round {os.path.basename(self.input_list[dir_cnt])} finished")
            if input_dir != self.input_list[0]:
                """Erase all mid interpolations except the img input"""
                shutil.rmtree(input_dir)
            dir_cnt += 1
        pass


if __name__ == "__main__":
    pass
