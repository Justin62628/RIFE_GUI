# coding: utf-8

import cv2
import numpy as np
from PIL import Image

from ncnn.sr.realSR.realsr_ncnn_vulkan import RealSR
from ncnn.sr.waifu2x.waifu2x_ncnn_vulkan import Waifu2x


class SvfiWaifu(Waifu2x):
    def __init__(self, model="", scale=1, num_threads=4, **kwargs):
        super().__init__(gpuid=0,
                         model=model,
                         tta_mode=False,
                         num_threads=num_threads,
                         scale=scale,
                         noise=0,
                         tilesize=0, )

    def svfi_process(self, img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = self.process(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image


class SvfiRealSR(RealSR):
    def __init__(self, model="", scale=1, num_threads=4, **kwargs):
        super().__init__(gpuid=0,
                         model=model,
                         tta_mode=False,
                         scale=scale,
                         tilesize=0, )

    def svfi_process(self, img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = self.process(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image
