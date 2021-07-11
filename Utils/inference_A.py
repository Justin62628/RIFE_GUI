import os
import warnings

from Utils.utils import ArgumentManager
from ncnn.rife import rife_ncnn_vulkan

warnings.filterwarnings("ignore")
raw = rife_ncnn_vulkan.raw


class RifeInterpolation(rife_ncnn_vulkan.RIFE):
    def __init__(self, __args: ArgumentManager):
        self.ARGS = __args
        uhd_mode = True if self.ARGS.rife_exp < 1 else False
        super().__init__(self.ARGS.ncnn_gpu, os.path.basename(self.ARGS.rife_model),
                         uhd_mode=uhd_mode, num_threads=self.ARGS.ncnn_thread)
        self.initiated = False

        self.device = None
        self.model = None
        self.tta_mode = self.ARGS.use_rife_tta_mode
        self.model_path = ""

    def initiate_rife(self, __args=None):
        if self.initiated:
            return
        self.initiated = True

    def generate_input_img(self, img):
        """
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        return img

    def __inference(self, i1, i2):
        if self.ARGS.is_rife_reverse:
            mid = self.process(i1, i2)[0]
        else:
            mid = self.process(i2, i1)[0]
        return mid

    def __make_n_inference(self, i1, i2, scale, n):
        mid = self.__inference(i1, i2)
        if self.tta_mode:
            mid1 = self.__inference(i1, mid)
            mid2 = self.__inference(mid, i2)
            mid = self.__inference(mid1, mid2)
        if n == 1:
            return [mid]
        first_half = self.__make_n_inference(i1, mid, scale, n=n // 2)
        second_half = self.__make_n_inference(mid, i2, scale, n=n // 2)
        if n % 2:
            return [*first_half, mid, *second_half]
        else:
            return [*first_half, *second_half]

    def generate_n_interp(self, img1, img2, n, scale, debug=False, test=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen
        img1 = self.generate_input_img(img1)
        img2 = self.generate_input_img(img2)
        interp_gen = self.__make_n_inference(img1, img2, scale, n)
        return interp_gen

    def run(self):
        pass


if __name__ == "__main__":
    pass
