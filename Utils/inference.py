import os
import traceback
import warnings

import numpy as np
import torch
from torch.nn import functional as F
from Utils.utils import ArgumentManager

warnings.filterwarnings("ignore")


class RifeInterpolation:
    def __init__(self, __args: ArgumentManager):
        self.initiated = False
        self.ARGS = __args

        self.auto_scale = self.ARGS.use_rife_auto_scale
        self.device = None
        self.model = None
        self.model_path = ""
        self.model_version = 0
        self.tta_mode = self.ARGS.use_rife_tta_mode
        pass

    def initiate_rife(self, __args=None):
        if self.initiated:
            return

        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.ARGS.use_rife_fp16 = False
            print("INFO - use cpu to interpolate")
        else:
            self.device = torch.device("cuda")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            if self.ARGS.use_rife_fp16:
                try:
                    torch.set_default_tensor_type(torch.cuda.HalfTensor)
                    print("INFO - FP16 mode switch success")
                except Exception as e:
                    print("INFO - FP16 mode switch failed")
                    traceback.print_exc()
                    self.ARGS.use_rife_fp16 = False

        if self.ARGS.rife_model == "":
            self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log')
        else:
            self.model_path = self.ARGS.rife_model

        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(self.model_path, -1)
                self.model_version = 2
                print("INFO - Loaded v2.x HD model.")
            except:
                from model.RIFE_HDv3 import Model
                model = Model(forward_ensemble=self.ARGS.use_rife_forward_ensemble)
                model.load_model(self.model_path, -1)
                self.model_version = 3
                print("INFO - Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(self.model_path, -1)
            self.model_version = 1
            print("INFO - Loaded v1.x HD model")

        self.model = model

        self.model.eval()
        self.model.device()
        # print(f"INFO - Load model at {self.model_path}")
        self.initiated = True

    def __inference(self, i1, i2, scale):
        if self.ARGS.is_rife_reverse:
            mid = self.model.inference(i1, i2, scale)
        else:
            mid = self.model.inference(i2, i1, scale)
        return mid

    def __make_n_inference(self, img1, img2, scale, n):
        padding, h, w = self.generate_padding(img1, scale)
        i1 = self.generate_torch_img(img1, padding)
        i2 = self.generate_torch_img(img2, padding)
        mid = self.__inference(i1, i2, scale)
        if self.tta_mode:
            mid1 = self.__inference(i1, mid, scale)
            mid2 = self.__inference(mid, i2, scale)
            mid = self.__inference(mid1, mid2, scale)
        del i1, i2
        mid = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))[:h, :w].copy()
        if n == 1:
            return [mid]
        first_half = self.__make_n_inference(img1, mid, scale, n=n // 2)
        second_half = self.__make_n_inference(mid, img2, scale, n=n // 2)
        if n % 2:
            return [*first_half, mid, *second_half]
        else:
            return [*first_half, *second_half]

    def generate_padding(self, img, scale: float):
        """

        :param scale:
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        tmp = max(32, int(32 / scale))
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
        if self.ARGS.use_rife_fp16:
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    def generate_n_interp(self, img1, img2, n, scale, debug=False, test=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen
        interp_gen = self.__make_n_inference(img1, img2, scale, n=n)
        return interp_gen

    def run(self):
        pass


if __name__ == "__main__":
    pass
