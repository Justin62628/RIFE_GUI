import os
import shutil
import traceback
import warnings

import numpy as np
import time
import threading
import torch
from torch.nn import functional as F
from Utils.utils import Utils

warnings.filterwarnings("ignore")
Utils = Utils()

class RifeInterpolation:
    def __init__(self, __args):
        self.initiated = False
        self.args = {}
        if __args is not None:
            """Update Args"""
            self.args = __args
        else:
            raise NotImplementedError("Args not sent in")

        self.device = None
        self.model = None
        self.model_path = ""
        pass

    def initiate_rife(self, __args=None):
        if self.initiated:
            return

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("use cpu to interpolate")
        else:
            self.device = torch.device("cuda")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if self.args["fp16"]:
            try:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
                print("FP16 mode switch success")
            except Exception as e:
                print("FP16 mode switch failed")
                traceback.print_exc()
                self.args["fp16"] = False

        torch.set_grad_enabled(False)
        from Utils.model.RIFE_HDv2 import Model
        self.model = Model()
        if self.args["SelectedModel".lower()] == "":
            self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log')
        else:
            self.model_path = self.args["SelectedModel".lower()]
        self.model.load_model(self.model_path, -1)
        print(f"Load model at {self.model_path}")
        self.model.eval()
        self.model.device()
        self.initiated = True

    def make_inference(self, img1, img2, scale, exp):
        padding, h, w = self.generate_padding(img1)
        i1 = self.generate_torch_img(img1, padding)
        i2 = self.generate_torch_img(img2, padding)
        if self.args["reverse"]:
            mid = self.model.inference(i1, i2, scale)
        else:
            mid = self.model.inference(i2, i1, scale)
        del i1, i2
        mid = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))[:h, :w].copy()
        if exp == 1:
            return [mid]
        first_half = self.make_inference(img1, mid, scale, exp=exp - 1)
        second_half = self.make_inference(mid, img2, scale, exp=exp - 1)
        return [*first_half, mid, *second_half]

    def generate_padding(self, img):
        """

        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        tmp = max(32, int(32 / self.args["scale"]))
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
        if self.args["fp16"]:
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
        interp_gen = self.make_inference(img1, img2, scale, exp)
        return interp_gen

    def run(self):
        pass




if __name__ == "__main__":
    pass
