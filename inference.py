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


class NCNNinterpolator(threading.Thread):

    def __init__(self, __args):
        super().__init__()
        if __args is not None:
            self.args = __args

        self.rife_ncnn_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rife-ncnn")
        self.rife_ncnn = os.path.join(self.rife_ncnn_root, "rife-ncnn-vulkan.exe")
        self.exp = self.args["exp"]
        self.j_settings = self.args["j_settings"]
        self.input_list = list()
        self.input_root = os.path.dirname(self.args["input_dir"])
        self.supervise_thread = None
        self.supervise_data = {}
        print(f"Use NCNN to interpolate from {self.rife_ncnn_root} with input list {str(self.input_list)}")
        pass

    def initiate_rife(self):
        pass

    def generate_input_list(self):
        self.input_list.append(self.args["input_dir"])
        for e in range(self.exp - 1):
            new_input = os.path.join(self.input_root, f"mid_interp_{2 ** (e + 1)}x")
            self.input_list.append(new_input)
            if not os.path.exists(new_input):
                os.mkdir(new_input)
        self.input_list.append(os.path.join(self.input_root, "interp"))
        pass

    def run(self):
        self.generate_input_list()
        dir_cnt = 1

        shutil.rmtree(os.path.join(self.input_root, "interp"))
        os.mkdir(os.path.join(self.input_root, "interp"))

        for input_dir in self.input_list:
            if os.path.basename(input_dir) == "interp":
                """Ignore interp"""
                break
            output_dir = self.input_list[dir_cnt]
            input_cnt = len(os.listdir(input_dir))
            ncnn_thread = threading.Thread(target=self.ncnn_interpolate, name="NCNN-Interpolate Thread",
                                           args=(input_dir, output_dir,))
            ncnn_thread.start()
            while ncnn_thread.is_alive():
                time.sleep(0.1)
                output_frames_cnt = len(os.listdir(output_dir))
                now_cnt = int(output_frames_cnt/2)
                self.supervise_data.update({"now_cnt": now_cnt, "now_dir": os.path.basename(output_dir),
                                            "input_cnt": input_cnt})

            print(f"INFO - [NCNN] Round {output_dir} finished")
            dir_cnt += 1

        for input_dir in self.input_list:
            if input_dir != self.input_list[0] and os.path.basename(input_dir) != "interp":
                """Erase all mid interpolations except the img input"""
                shutil.rmtree(input_dir)
        pass

    def ncnn_interpolate(self, input_dir, output_dir):
        if len(self.j_settings):
            j_settings = f"-j {self.j_settings}"
        else:
            j_settings = ""
        create_command = f"{self.rife_ncnn}  -i {input_dir} -o {output_dir} " \
                         f"-m {Utils.fillQuotation(os.path.join(self.rife_ncnn_root, 'rife-v2.4'))} " \
                         f"{j_settings}"
        os.system(create_command)
        pass

    def stop(self):
        """
        stop ncnn interpolation
        :return:
        """


if __name__ == "__main__":
    pass
