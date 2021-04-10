import os
import shutil
import traceback
import warnings

import numpy as np
import time
import threading
from Utils.utils import Utils

warnings.filterwarnings("ignore")
Utils = Utils()


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

            print(f"INFO - [NCNN] Round {output_dir.encode('utf-8', 'ignore').decode('utf-8')} finished")
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
