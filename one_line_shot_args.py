# coding: utf-8
import argparse
import re
import sys

import psutil
import tqdm
from skvideo.io import FFmpegWriter, FFmpegReader

from Utils.utils import *

print("INFO - ONE LINE SHOT ARGS 6.8.2 2021/7/16")
"""
Update Log at 6.8.2
1. Greatly Optimize Tasks Pipe
2. Optimize SVFI GUI toolbox utility(multi thread)
3. Fix pretty much bugs on work flow
"""

"""设置环境路径"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(os.path.dirname(dname))
sys.path.append(dname)

"""输入命令行参数"""
parser = argparse.ArgumentParser(prog="#### RIFE CLI tool/补帧分步设置命令行工具 by Jeanna ####",
                                 description='Interpolation for sequences of images')
basic_parser = parser.add_argument_group(title="Basic Settings, Necessary")
basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                          help="原视频/图片序列文件夹路径")
basic_parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                          help="成品输出的路径，注意默认在项目文件夹")
basic_parser.add_argument("-c", '--config', dest='config', type=str, required=True, help="配置文件路径")
basic_parser.add_argument("-t", '--task-id', dest='task_id', type=str, required=True, help="任务id")
basic_parser.add_argument('--concat-only', dest='concat_only', action='store_true', help='只执行合并已有区块操作')
basic_parser.add_argument('--extract-only', dest='extract_only', action='store_true', help='只执行拆帧操作')
basic_parser.add_argument('--render-only', dest='render_only', action='store_true', help='只执行渲染操作')

args_read = parser.parse_args()
cp = DefaultConfigParser(allow_no_value=True)  # 把SVFI GUI传来的参数格式化
cp.read(args_read.config, encoding='utf-8')
cp_items = dict(cp.items("General"))
args = Tools.clean_parsed_config(cp_items)
args.update(vars(args_read))  # update -i -o -c，将命令行参数更新到config生成的字典
ARGS = ArgumentManager(args)

"""设置可见的gpu"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if int(ARGS.use_specific_gpu) != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{ARGS.use_specific_gpu}"

"""强制使用CPU"""
if ARGS.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f""


class InterpWorkFlow:
    def __init__(self, __args: ArgumentManager, **kwargs):
        self.ARGS = __args

        """获得补帧输出路径"""
        if os.path.isfile(self.ARGS.output_dir):
            self.ARGS.output_dir = os.path.dirname(self.ARGS.output_dir)
        self.project_dir = os.path.join(self.ARGS.output_dir,
                                        f"{Tools.get_filename(self.ARGS.input)}_{self.ARGS.task_id}")

        if not os.path.exists(self.project_dir):
            os.mkdir(self.project_dir)

        """Set Logger"""
        sys.path.append(self.project_dir)
        self.logger = Tools.get_logger("[ARGS]", self.project_dir, debug=self.ARGS.debug)

        self.logger.info(f"Initial New Interpolation Project: project_dir: %s, INPUT_FILEPATH: %s", self.project_dir,
                         self.ARGS.input)
        self.logger.info("Changing working dir to {0}".format(dname))

        """Set FFmpeg"""
        self.ffmpeg = Tools.fillQuotation(os.path.join(self.ARGS.ffmpeg, "ffmpeg.exe"))
        self.ffplay = Tools.fillQuotation(os.path.join(self.ARGS.ffmpeg, "ffplay.exe"))
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.logger.warning("Not find selected ffmpeg, use default")

        """Set input output and initiate environment"""
        self.input = self.ARGS.input
        self.output = self.ARGS.output_dir
        self.input_dir = os.path.join(self.project_dir, 'frames')
        self.interp_dir = os.path.join(self.project_dir, 'interp')
        self.scene_dir = os.path.join(self.project_dir, 'scenes')
        self.env = [self.input_dir, self.interp_dir, self.scene_dir]

        self.ARGS.is_img_input = not os.path.isfile(self.input)

        """Load Interpolation Exp"""
        self.rife_exp = self.ARGS.rife_exp

        """Get input's info"""
        self.video_info_instance = VideoInfo(file_input=self.input, logger=self.logger, project_dir=self.project_dir,
                                             ffmpeg=self.ARGS.ffmpeg, img_input=self.ARGS.is_img_input,
                                             strict_mode=self.ARGS.is_hdr_strict_mode, exp=self.ARGS.rife_exp)
        self.video_info = self.video_info_instance.get_info()
        if self.ARGS.batch and not self.ARGS.is_img_input:  # 检测到批处理，且输入不是文件夹，使用检测到的帧率
            self.input_fps = self.video_info["fps"]
        elif self.ARGS.input_fps:
            self.input_fps = self.ARGS.input_fps
        else:  # 用户有毒，未发现有效的输入帧率，用检测到的帧率
            if self.video_info["fps"] is None or not self.video_info["fps"]:
                raise OSError("Not Find FPS, Input File not valid")
            self.input_fps = self.video_info["fps"]

        if self.ARGS.is_img_input:
            self.target_fps = self.ARGS.target_fps
            self.ARGS.is_save_audio = False
            # but assigned output fps will be not touched
        else:
            if self.ARGS.target_fps:
                self.target_fps = self.ARGS.target_fps
            else:
                self.target_fps = (2 ** self.rife_exp) * self.input_fps  # default

        """Update All Frames Count"""
        self.all_frames_cnt = min(abs(int(self.video_info["duration"] * self.target_fps)),
                                  10 ** 10)  # constrain frames cnt

        """Crop Video"""
        self.crop_param = [0, 0]  # crop parameter, 裁切参数
        crop_param = self.ARGS.crop.replace("：", ":")
        if crop_param not in ["", "0", None]:
            width_black, height_black = crop_param.split(":")
            width_black = int(width_black)
            height_black = int(height_black)
            self.crop_param = [width_black, height_black]
            self.logger.info(f"Update Crop Parameters to {self.crop_param}")

        """initiation almost ready"""
        self.logger.info(
            f"Check Interpolation Source, FPS: {self.input_fps}, TARGET FPS: {self.target_fps}, "
            f"FRAMES_CNT: {self.all_frames_cnt}, EXP: {self.rife_exp}")

        """RIFE Core"""
        self.rife_core = RifeInterpolation(self.ARGS)  # 用于补帧的模块

        """Guess Memory and Render System"""
        if self.ARGS.use_manual_buffer:
            # 手动指定内存占用量
            free_mem = self.ARGS.manual_buffer_size * 1024
        else:
            mem = psutil.virtual_memory()
            free_mem = round(mem.free / 1024 / 1024)
        if self.ARGS.resize_width != 0 and self.ARGS.resize_height != 0:
            if self.ARGS.resize_width % 2 != 0:
                self.ARGS.resize_width += 1
            if self.ARGS.resize_height % 2 != 0:
                self.ARGS.resize_height += 1
            self.frames_output_size = round(free_mem / (sys.getsizeof(
                np.random.rand(3, round(self.ARGS.resize_width),
                               round(self.ARGS.resize_height))) / 1024 / 1024) * 0.8)
        else:
            self.frames_output_size = round(free_mem / (sys.getsizeof(
                np.random.rand(3, round(self.video_info["size"][0]),
                               round(self.video_info["size"][1]))) / 1024 / 1024) * 0.8)
        if not self.ARGS.use_manual_buffer:
            self.frames_output_size = int(max(10.0, self.frames_output_size * 0.9))
        self.logger.info(f"Buffer Size to {self.frames_output_size}")

        self.frames_output = Queue(maxsize=self.frames_output_size)  # 补出来的帧序列队列（消费者）
        self.rife_task_queue = Queue(maxsize=self.frames_output_size)  # 补帧任务队列（生产者）
        self.rife_thread = None  # 帧插值预处理线程（生产者）
        self.rife_work_event = threading.Event()
        self.rife_work_event.clear()
        self.sr_module = SuperResolution()  # 超分类

        if self.ARGS.use_sr:
            try:
                import Utils.SuperResolutionModule
                input_resolution = self.video_info["size"][0] * self.video_info["size"][1]
                output_resolution = self.ARGS.resize_width * self.ARGS.resize_height
                resolution_rate = output_resolution / input_resolution
                if input_resolution and resolution_rate > 1:
                    sr_scale = Tools.get_exp_edge(resolution_rate)
                    if self.ARGS.use_sr_algo == "waifu2x":
                        self.sr_module = Utils.SuperResolutionModule.SvfiWaifu(model=self.ARGS.use_sr_model,
                                                                               scale=sr_scale,
                                                                               num_threads=self.ARGS.ncnn_thread)
                    elif self.ARGS.use_sr_algo == "realSR":
                        self.sr_module = Utils.SuperResolutionModule.SvfiRealSR(model=self.ARGS.use_sr_model,
                                                                                scale=sr_scale)
                    self.logger.info(
                        f"Load AI SR at {self.ARGS.use_sr_algo}, {self.ARGS.use_sr_model}, scale = {sr_scale}")
                else:
                    self.logger.warning("Abort to load AI SR since Resolution Rate < 1")
            except ImportError:
                self.logger.error(f"Import SR Module failed\n{traceback.format_exc()}")

        self.frame_reader = None  # 读帧的迭代器／帧生成器
        self.render_gap = self.ARGS.render_gap  # 每个chunk的帧数
        self.render_thread = None  # 帧渲染器
        self.task_info = {"chunk_cnt": -1, "render": -1, "now_frame": -1}  # 有关渲染的实时信息

        """Scene Detection"""
        if self.ARGS.scdet_mode == 0:
            """Old Mode"""
        self.scene_detection = TransitionDetection_ST(self.project_dir, int(0.5 * self.input_fps),
                                                      scdet_threshold=self.ARGS.scdet_threshold,
                                                      no_scdet=self.ARGS.is_no_scdet,
                                                      use_fixed_scdet=self.ARGS.use_scdet_fixed,
                                                      fixed_max_scdet=self.ARGS.scdet_fixed_max,
                                                      scdet_output=self.ARGS.is_scdet_output)
        """Duplicate Frames Removal"""
        self.dup_skip_limit = int(0.5 * self.input_fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环

        """Main Thread Lock"""
        self.main_event = threading.Event()
        self.render_lock = threading.Event()  # 渲染锁，没有用
        self.main_event.set()

        """Set output's color info"""
        self.color_info = {}
        for k in self.video_info:
            if k.startswith("-"):
                self.color_info[k] = self.video_info[k]

        """maintain output extension"""
        self.output_ext = "." + self.ARGS.output_ext
        if "ProRes" in self.ARGS.render_encoder and not self.ARGS.is_img_output:
            self.output_ext = ".mov"

        self.main_error = None
        self.first_hdr_check_report = True

    def generate_frame_reader(self, start_frame=-1, frame_check=False):
        """
        输入帧迭代器
        :param frame_check:
        :param start_frame:
        :return:
        """
        """If input is sequence of frames"""
        if self.ARGS.is_img_input:
            img_io = ImgSeqIO(folder=self.input, is_read=True,
                              start_frame=self.ARGS.interp_start, logger=self.logger,
                              output_ext=self.ARGS.output_ext, )
            self.all_frames_cnt = img_io.get_frames_cnt()
            self.logger.info(f"Img Input, update frames count to {self.all_frames_cnt}")
            return img_io

        """If input is a video"""
        input_dict = {"-vsync": "0", }
        if self.ARGS.render_hwaccel_mode:
            input_dict.update({"-hwaccel": "auto"})
        if self.ARGS.input_start_point is not None or self.ARGS.input_end_point is not None:
            """任意时段补帧"""
            time_fmt = "%H:%M:%S"
            start_point = datetime.datetime.strptime(self.ARGS.input_start_point, time_fmt)
            end_point = datetime.datetime.strptime(self.ARGS.input_end_point, time_fmt)
            if end_point > start_point:
                input_dict.update({"-ss": self.ARGS.input_start_point, "-to": self.ARGS.input_end_point})
                start_frame = -1
                clip_duration = end_point - start_point
                clip_fps = self.target_fps
                self.all_frames_cnt = round(clip_duration.total_seconds() * clip_fps)
                self.logger.info(
                    f"Update Input Range: in {self.ARGS.input_start_point} -> out {self.ARGS.input_end_point}, all_frames_cnt -> {self.all_frames_cnt}")
            else:
                self.logger.warning(f"Input Time Section change to original course")

        output_dict = {
            "-vframes": str(10 ** 10), }  # use read frames cnt to avoid ffprobe, fuck

        output_dict.update(self.color_info)

        if frame_check:
            """用以一拍二一拍N除重模式的预处理"""
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": "256x256"})
        elif len(self.ARGS.resize) and not self.ARGS.use_sr:
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": self.ARGS.resize.replace(":", "x").replace("*", "x")})
        vf_args = f"copy"
        if self.ARGS.use_deinterlace:
            vf_args += f",yadif=parity=auto"
        vf_args += f",minterpolate=fps={self.target_fps}:mi_mode=dup"
        if start_frame not in [-1, 0]:
            input_dict.update({"-ss": f"{start_frame / self.target_fps:.3f}"})

        """Quick Extraction"""
        if not self.ARGS.is_quick_extract:
            vf_args += f",format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24"

        """Update video filters"""
        output_dict["-vf"] = vf_args
        self.logger.debug(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=self.input, inputdict=input_dict, outputdict=output_dict)

    def generate_frame_renderer(self, output_path, start_frame=0):
        """
        渲染帧
        :param start_frame: for IMG IO, select start_frame to generate IO instance
        :param output_path:
        :return:
        """
        params_265 = ("ref=4:rd=3:no-rect=1:no-amp=1:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:"
                      "weightb=1:no-strong-intra-smoothing=1:psy-rd=2.0:psy-rdoq=1.0:no-open-gop=1:"
                      f"keyint={int(self.target_fps * 3)}:min-keyint=1:rc-lookahead=120:bframes=6:"
                      f"aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:"
                      f"deblock=-1:no-sao=1")

        def HDRChecker():
            nonlocal params_265
            if self.ARGS.is_img_input:
                return

            if self.ARGS.is_hdr_strict_mode:
                self.logger.warning("Strict Mode, Skip HDR Check")
                return

            if "color_transfer" not in self.video_info["video_info"]:
                if self.first_hdr_check_report:
                    self.logger.warning("Not Find Color Transfer\n%s" % str(self.video_info["video_info"]))
                return

            color_trc = self.video_info["video_info"]["color_transfer"]

            if "smpte2084" in color_trc or "bt2020" in color_trc:
                hdr = True
                self.ARGS.render_encoder = "H265, 10bit"
                self.ARGS.render_encoder_preset = "slow"
                self.ARGS.render_hwaccel_mode = "CPU"
                if "master-display" in str(self.video_info["video_info"]):
                    self.ARGS.render_hwaccel_mode = "CPU"
                    params_265 += ":hdr10-opt=1:repeat-headers=1"
                    if self.first_hdr_check_report:
                        self.logger.warning("\nDetect HDR10+ Content, Switch to CPU Render Compulsorily")
                else:
                    if self.first_hdr_check_report:
                        self.logger.warning(
                            "\nPQ or BT2020 Content Detected, Switch to CPU Render Compulsorily")

            elif "arib-std-b67" in color_trc:
                hdr = True
                self.ARGS.render_encoder = "H265, 10bit"
                self.ARGS.render_encoder_preset = "slow"
                self.ARGS.render_hwaccel_mode = "CPU"
                if self.first_hdr_check_report:
                    self.logger.warning("\nHLG Content Detected, Switch to CPU Render Compulsorily")
            pass

        """If output is sequence of frames"""
        if self.ARGS.is_img_output:
            img_io = ImgSeqIO(folder=self.output, is_read=False,
                              start_frame=start_frame, logger=self.logger,
                              output_ext=self.ARGS.output_ext, )
            return img_io

        """HDR Check"""
        if self.first_hdr_check_report:
            HDRChecker()
            self.first_hdr_check_report = False

        """Output Video"""
        input_dict = {"-vsync": "cfr"}

        output_dict = {"-r": f"{self.target_fps}", "-preset": self.ARGS.render_encoder_preset,
                       "-metadata": f'title="Made By SVFI {self.ARGS.version}"'}

        output_dict.update(self.color_info)

        if not self.ARGS.is_img_input:
            input_dict.update({"-r": f"{self.target_fps}"})
        else:
            """Img Input"""
            input_dict.update({"-r": f"{self.input_fps * 2 ** self.rife_exp}"})

        """Slow motion design"""
        if self.ARGS.is_render_slow_motion:
            if self.ARGS.render_slow_motion_fps:
                input_dict.update({"-r": f"{self.ARGS.render_slow_motion_fps}"})
            else:
                input_dict.update({"-r": f"{self.target_fps}"})
            output_dict.pop("-r")

        vf_args = "copy"  # debug
        output_dict.update({"-vf": vf_args})

        if self.ARGS.use_sr:
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": self.ARGS.resize.replace(":", "x").replace("*", "x")})

        """Assign Render Codec"""
        """CRF / Bitrate Controll"""
        if self.ARGS.render_hwaccel_mode == "CPU":
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "libx264", "-preset:v": self.ARGS.render_encoder_preset})
                if "8bit" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "high",
                                        "-weightb": "1", "-weightp": "2", "-mbtree": "1", "-forced-idr": "1",
                                        "-coder": "1",
                                        "-x264-params": "keyint=200:min-keyint=1:bframes=6:b-adapt=2:no-open-gop=1:"
                                                        "ref=8:deblock='-1:-1':rc-lookahead=50:chroma-qp-offset=-2:"
                                                        "aq-mode=1:aq-strength=0.8:qcomp=0.75:me=umh:merange=24:"
                                                        "subme=10:psy-rd='1:0.1'",
                                        })
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "high10",
                                        "-weightb": "1", "-weightp": "2", "-mbtree": "1", "-forced-idr": "1",
                                        "-coder": "1",
                                        "-x264-params": "keyint=200:min-keyint=1:bframes=6:b-adapt=2:no-open-gop=1:"
                                                        "ref=8:deblock='-1:-1':rc-lookahead=50:chroma-qp-offset=-2:"
                                                        "aq-mode=1:aq-strength=0.8:qcomp=0.75:me=umh:merange=24:"
                                                        "subme=10:psy-rd='1:0.1'",
                                        })
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "libx265", "-preset:v": self.ARGS.render_encoder_preset})
                if "8bit" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "main",
                                        "-x265-params": params_265})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "main10",
                                        "-x265-params": params_265})
            else:
                """ProRes"""
                if "-preset" in output_dict:
                    output_dict.pop("-preset")
                output_dict.update({"-c:v": "prores_ks", "-profile:v": self.ARGS.render_encoder_preset, })
                if "422" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv422p10le"})
                else:
                    output_dict.update({"-pix_fmt": "yuv444p10le"})

        elif self.ARGS.render_hwaccel_mode == "NVENC":
            output_dict.update({"-pix_fmt": "yuv420p"})
            if "10bit" in self.ARGS.render_encoder:
                output_dict.update({"-pix_fmt": "yuv420p10le"})
                pass
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({f"-g": f"{int(self.target_fps * 3)}", "-c:v": "h264_nvenc", "-rc:v": "vbr_hq", })
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "hevc_nvenc", "-rc:v": "vbr_hq",
                                    f"-g": f"{int(self.target_fps * 3)}", })

            if self.ARGS.render_encoder_preset != "loseless":
                hwacccel_preset = self.ARGS.render_hwaccel_preset
                if hwacccel_preset != "None":
                    output_dict.update({"-i_qfactor": "0.71", "-b_qfactor": "1.3", "-keyint_min": "1",
                                        f"-rc-lookahead": "120", "-forced-idr": "1", "-nonref_p": "1",
                                        "-strict_gop": "1", })
                    if hwacccel_preset == "5th":
                        output_dict.update({"-bf": "0"})
                    elif hwacccel_preset == "6th":
                        output_dict.update({"-bf": "0", "-weighted_pred": "1"})
                    elif hwacccel_preset == "7th+":
                        output_dict.update({"-bf": "4", "-temporal-aq": "1", "-b_ref_mode": "2"})
            else:
                output_dict.update({"-preset": "10", })

        else:
            """QSV"""
            output_dict.update({"-pix_fmt": "yuv420p"})
            if "10bit" in self.ARGS.render_encoder:
                output_dict.update({"-pix_fmt": "yuv420p10le"})
                pass
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "h264_qsv",
                                    "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-rc-lookahead": "120", })
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "hevc_qsv",
                                    f"-g": f"{int(self.target_fps * 3)}", "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-look_ahead": "120", })

        if "ProRes" not in self.ARGS.render_encoder and self.ARGS.render_encoder_preset != "loseless":

            if self.ARGS.render_crf and self.ARGS.use_crf:
                if self.ARGS.render_hwaccel_mode != "CPU":
                    hwaccel_mode = self.ARGS.render_hwaccel_mode
                    if hwaccel_mode == "NVENC":
                        output_dict.update({"-cq:v": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "QSV":
                        output_dict.update({"-q": str(self.ARGS.render_crf)})
                else:  # CPU
                    output_dict.update({"-crf": str(self.ARGS.render_crf)})

            if self.ARGS.render_bitrate and self.ARGS.use_bitrate:
                output_dict.update({"-b:v": f'{self.ARGS.render_bitrate}M'})
                if self.ARGS.render_hwaccel_mode == "QSV":
                    output_dict.update({"-maxrate": "200M"})

        if self.ARGS.use_manual_encode_thread:
            output_dict.update({"-threads": f"{self.ARGS.render_encode_thread}"})

        self.logger.debug(f"writer: {output_dict}, {input_dict}")

        """Customize FFmpeg Render Command"""
        ffmpeg_customized_command = {}
        if len(self.ARGS.render_ffmpeg_customized):
            shlex_out = shlex.split(self.ARGS.render_ffmpeg_customized)
            if len(shlex_out) % 2 != 0:
                self.logger.warning(f"Customized FFmpeg is invalid: {self.ARGS.render_ffmpeg_customized}")
            else:
                for i in range(int(len(shlex_out) / 2)):
                    ffmpeg_customized_command.update({shlex_out[i * 2]: shlex_out[i * 2 + 1]})
        self.logger.debug(ffmpeg_customized_command)
        output_dict.update(ffmpeg_customized_command)

        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)

    def check_chunk(self, del_chunk=False):
        """
        Get Chunk Start
        :param: del_chunk: delete all chunks existed
        :return: chunk, start_frame
        """
        if self.ARGS.is_img_output:
            """IMG OUTPUT"""
            img_io = ImgSeqIO(folder=self.output, is_tool=True,
                              start_frame=self.ARGS.interp_start, logger=self.logger,
                              output_ext=self.ARGS.output_ext, )
            last_img = img_io.get_start_frame()
            if last_img == 0:
                return 1, 0
            else:
                if self.ARGS.interp_start not in [-1, ] or self.ARGS.output_chunk_cnt not in [-1, 0]:
                    return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)  # Manually Prioritized

        chunk_regex = rf"chunk-[\d+].*?\{self.output_ext}"

        """获得现有区块"""
        if del_chunk:
            for f in os.listdir(self.project_dir):
                if re.match(chunk_regex, f):
                    os.remove(os.path.join(self.project_dir, f))
            return 1, 0

        """If remove only"""
        if del_chunk:
            return 1, 0

        chunk_info_path = os.path.join(self.project_dir, "chunk.json")

        if not os.path.exists(chunk_info_path):
            return 1, 0

        with open(chunk_info_path, "r", encoding="utf-8") as r:
            chunk_info = json.load(r)
        """
        key: project_dir, input filename, chunk cnt, chunk list, last frame
        """
        chunk_cnt = chunk_info["chunk_cnt"]
        """Not find previous chunk"""
        if not chunk_cnt:
            return 1, 0

        last_frame = chunk_info["last_frame"]

        """Manually Prioritized"""
        if self.ARGS.interp_start not in [-1, ] or self.ARGS.output_chunk_cnt not in [-1, 0]:
            if chunk_cnt + 1 != self.ARGS.output_chunk_cnt or last_frame + 1 != self.ARGS.interp_start:
                try:
                    os.remove(chunk_info_path)
                except FileNotFoundError:
                    pass
            return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)
        return chunk_cnt + 1, last_frame + 1

    def render(self, chunk_cnt, start_frame):
        """
        Render thread
        :param chunk_cnt:
        :param start_frame: render start
        :return:
        """

        def rename_chunk():
            """Maintain Chunk json"""
            if self.ARGS.is_img_output:
                return
            chunk_desc_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, now_frame,
                                                                       self.output_ext)
            chunk_desc_path = os.path.join(self.project_dir, chunk_desc_path)
            if os.path.exists(chunk_desc_path):
                os.remove(chunk_desc_path)
            os.rename(chunk_tmp_path, chunk_desc_path)
            chunk_path_list.append(chunk_desc_path)
            chunk_info_path = os.path.join(self.project_dir, "chunk.json")

            with open(chunk_info_path, "w", encoding="utf-8") as w:
                chunk_info = {
                    "project_dir": self.project_dir,
                    "input": self.input,
                    "chunk_cnt": chunk_cnt,
                    "chunk_list": chunk_path_list,
                    "last_frame": now_frame,
                    "target_fps": self.target_fps,
                }
                json.dump(chunk_info, w)
            """
            key: project_dir, input filename, chunk cnt, chunk list, last frame
            """
            if is_end:
                if os.path.exists(chunk_info_path):
                    os.remove(chunk_info_path)

        def check_audio_concat():
            """Check Input file ext"""
            if not self.ARGS.is_save_audio:
                return
            if self.ARGS.is_img_output:
                return
            output_ext = os.path.splitext(self.input)[-1]
            if output_ext not in SupportFormat.vid_outputs:
                output_ext = self.output_ext
            if "ProRes" in self.ARGS.render_encoder:
                output_ext = ".mov"

            concat_filepath = f"{os.path.join(self.output, 'concat_test')}" + output_ext
            map_audio = f'-i "{self.input}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy -shortest '
            ffmpeg_command = f'{self.ffmpeg} -hide_banner -i "{chunk_tmp_path}" {map_audio} -c:v copy ' \
                             f'{Tools.fillQuotation(concat_filepath)} -y'

            self.logger.info("Start Audio Concat Test")
            sp = Tools.popen(ffmpeg_command)
            sp.wait()
            if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath):
                self.logger.error(f"Concat Test Error, {output_ext}, empty output")
                self.main_error = FileExistsError("Concat Test Error, empty output, Check Output Extension!!!")
                raise FileExistsError("Concat Test Error, empty output, Check Output Extension!!!")
            self.logger.info("Audio Concat Test Success")
            os.remove(concat_filepath)

        concat_test_flag = True

        chunk_frame_cnt = 1  # number of frames of current output chunk
        chunk_path_list = list()
        chunk_tmp_path = os.path.join(self.project_dir, f"chunk-tmp{self.output_ext}")
        frame_writer = self.generate_frame_renderer(chunk_tmp_path, start_frame)  # get frame renderer

        now_frame = start_frame
        is_end = False
        while True:
            if not self.main_event.is_set():
                self.logger.warning("Main interpolation thread Dead, break")  # 主线程已结束，这里的锁其实没用，调试用的
                frame_writer.close()
                is_end = True
                rename_chunk()
                break

            frame_data = self.frames_output.get()
            if frame_data is None:
                frame_writer.close()
                is_end = True
                rename_chunk()
                break

            frame = frame_data[1]
            now_frame = frame_data[0]

            if self.ARGS.use_fast_denoise:
                frame = cv2.fastNlMeansDenoising(frame)

            frame_writer.writeFrame(frame)

            chunk_frame_cnt += 1
            self.task_info.update({"chunk_cnt": chunk_cnt, "render": now_frame})  # update render info

            if not chunk_frame_cnt % self.render_gap:
                frame_writer.close()
                if concat_test_flag:
                    check_audio_concat()
                    concat_test_flag = False
                rename_chunk()
                chunk_cnt += 1
                start_frame = now_frame + 1
                frame_writer = self.generate_frame_renderer(chunk_tmp_path, start_frame)
        return

    def feed_to_render(self, frames_list: list, is_end=False):
        """
        维护输出帧数组的输入（往输出渲染线程喂帧
        :param frames_list:
        :param is_end: 是否是视频结尾
        :return:
        """
        frames_list_len = len(frames_list)

        for frame_i in range(frames_list_len):
            if frames_list[frame_i] is None:
                self.frames_output.put(None)
                self.logger.info("Put None to write_buffer in advance")
                return
            self.frames_output.put(frames_list[frame_i])  # 往输出队列（消费者）喂正常的帧
            if frame_i == frames_list_len - 1:
                if is_end:
                    self.frames_output.put(None)
                    self.logger.info("Put None to write_buffer")
                    return
        pass

    def feed_to_rife(self, now_frame: int, img0, img1, n=0, exp=0, is_end=False, add_scene=False, ):
        """
        创建任务，输出到补帧任务队列消费者
        :param now_frame:当前帧数
        :param add_scene:加入转场的前一帧（避免音画不同步和转场鬼畜）
        :param img0:
        :param img1:
        :param n:要补的帧数
        :param exp:使用指定的补帧倍率（2**exp）
        :param is_end:是否是任务结束
        :return:
        """

        def psnr(i1, i2):
            i1 = np.float64(i1)
            i2 = np.float64(i2)
            mse = np.mean((i1 - i2) ** 2)
            if mse == 0:
                return 100
            pixel_max = 255.0
            return 20 * math.log10(pixel_max / math.sqrt(mse))

        scale = self.ARGS.rife_scale
        if self.ARGS.use_rife_auto_scale:
            """使用动态光流"""
            if img0 is None or img1 is None:
                scale = 1.0
            else:
                x = psnr(cv2.resize(img0, (256, 256)), cv2.resize(img1, (256, 256)))
                y25 = 0.0000136703 * (x ** 3) - 0.000407396 * (x ** 2) - 0.0129 * x + 0.62621
                y50 = 0.00000970763 * (x ** 3) - 0.0000908092 * (x ** 2) - 0.02095 * x - 0.69068
                y100 = 0.0000134965 * (x ** 3) - 0.000246688 * (x ** 2) - 0.01987 * x - 0.70953
                m = min(y25, y50, y100)
                scale = {y25: 0.25, y50: 0.5, y100: 1.0}[m]

        self.rife_task_queue.put(
            {"now_frame": now_frame, "img0": img0, "img1": img1, "n": n, "exp": exp, "scale": scale,
             "is_end": is_end, "add_scene": add_scene})

    def crop_read_img(self, img):
        """
        Crop using self.crop parameters
        :param img:
        :return:
        """
        if img is None:
            return img

        h, w, _ = img.shape
        if self.crop_param[0] > w or self.crop_param[1] > h:
            """奇怪的黑边参数，不予以处理"""
            return img
        return img[self.crop_param[1]:h - self.crop_param[1], self.crop_param[0]:w - self.crop_param[0]]

    def nvidia_vram_test(self):
        """
        显存测试
        :return:
        """
        try:
            if len(self.ARGS.resize):
                w, h = list(map(lambda x: int(x), self.ARGS.resize.split("x")))
            else:

                w, h = list(map(lambda x: round(x), self.video_info["size"]))

            # if w * h > 1920 * 1080:
            #     if self.ARGS.rife_scale > 0.5:
            #         """超过1080p锁光流尺度为0.5"""
            #         self.ARGS.rife_scale = 0.5
            #     self.logger.warning(f"Big Resolution (>1080p) Input found")
            # else:
            self.logger.info(f"Start VRAM Test: {w}x{h} with scale {self.ARGS.rife_scale}")

            test_img0, test_img1 = np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8), \
                                   np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8)
            self.rife_core.generate_n_interp(test_img0, test_img1, 1, self.ARGS.rife_scale)
            self.logger.info(f"VRAM Test Success")
            del test_img0, test_img1
        except Exception as e:
            self.logger.error("VRAM Check Failed, PLS Lower your presets\n" + traceback.format_exc())
            raise e

    def remove_duplicate_frames(self, videogen_check: FFmpegReader.nextFrame, init=False) -> (list, list, dict):
        """
        获得新除重预处理帧数序列
        :param init: 第一次重复帧
        :param videogen_check:
        :return:
        """
        flow_dict = dict()
        canny_dict = dict()
        predict_dict = dict()
        resize_param = (40, 40)

        def get_img(i0):
            if i0 in check_frame_data:
                return check_frame_data[i0]
            else:
                return None

        def calc_flow_distance(pos0: int, pos1: int, _use_flow=True):
            if not _use_flow:
                return diff_canny(pos0, pos1)
            if (pos0, pos1) in flow_dict:
                return flow_dict[(pos0, pos1)]
            if (pos1, pos0) in flow_dict:
                return flow_dict[(pos1, pos0)]

            prev_gray = cv2.cvtColor(cv2.resize(get_img(pos0), resize_param), cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(cv2.resize(get_img(pos1), resize_param), cv2.COLOR_BGR2GRAY)
            flow0 = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow1 = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow = (flow0 - flow1) / 2
            x = flow[:, :, 0]
            y = flow[:, :, 1]
            dis = np.linalg.norm(x) + np.linalg.norm(y)
            flow_dict[(pos0, pos1)] = dis
            return dis

        def diff_canny(pos0, pos1):
            if (pos0, pos1) in canny_dict:
                return canny_dict[(pos0, pos1)]
            if (pos1, pos0) in canny_dict:
                return canny_dict[(pos1, pos0)]
            canny_diff = cv2.Canny(cv2.absdiff(get_img(pos0), get_img(pos0)), 100, 200).mean()
            canny_dict[(pos0, pos1)] = canny_diff
            return canny_diff

        def predict_scale(pos0, pos1):
            if (pos0, pos1) in predict_dict:
                return predict_dict[(pos0, pos1)]
            if (pos1, pos0) in predict_dict:
                return predict_dict[(pos1, pos0)]

            w, h, _ = get_img(pos0).shape
            diff = cv2.Canny(cv2.absdiff(get_img(pos0), get_img(pos0)), 100, 200)
            mask = np.where(diff != 0)
            try:
                xmin = min(list(mask)[0])
            except:
                xmin = 0
            try:
                xmax = max(list(mask)[0]) + 1
            except:
                xmax = w
            try:
                ymin = min(list(mask)[1])
            except:
                ymin = 0
            try:
                ymax = max(list(mask)[1]) + 1
            except:
                ymax = h
            W = xmax - xmin
            H = ymax - ymin
            S0 = w * h
            S1 = W * H
            prediction = -2 * (S1 / S0) + 3
            predict_dict[(pos0, pos1)] = prediction
            return prediction

        use_flow = True
        check_queue_size = min(self.frames_output_size, 1000)  # 预处理长度
        check_frame_list = list()  # 采样图片帧数序列,key ~ LabData
        scene_frame_list = list()  # 转场图片帧数序列,key,和check_frame_list同步
        input_frame_data = dict()  # 输入图片数据
        check_frame_data = dict()  # 用于判断的采样图片数据
        if init:
            self.logger.info("Initiating Duplicated Frames Removal Process...This might take some time")
            pbar = tqdm.tqdm(total=check_queue_size, unit="frames")

        else:
            pbar = None
        """
            check_frame_list contains key, check_frame_data contains (key, frame_data)
        """
        check_frame_cnt = -1

        while len(check_frame_list) < check_queue_size:
            check_frame_cnt += 1
            check_frame = Tools.gen_next(videogen_check)
            if check_frame is None:
                break
            if init:
                pbar.update(1)
                pbar.set_description(
                    f"Process at Extract Frame {check_frame_cnt}")
            if len(check_frame_list):  # len>1
                if Tools.get_norm_img_diff(input_frame_data[check_frame_list[-1]],
                                           check_frame) < 0.001:
                    # do not use pure scene check to avoid too much duplication result
                    # duplicate frames
                    continue
            check_frame_list.append(check_frame_cnt)  # key list
            input_frame_data[check_frame_cnt] = check_frame
            # check_frame_data[check_frame_cnt] = cv2.resize(cv2.GaussianBlur(check_frame, (3, 3), 0), (256, 256))
            check_frame_data[check_frame_cnt] = cv2.resize(cv2.GaussianBlur(check_frame, (3, 3), 0), (256, 256))
        if not len(check_frame_list):
            if init:
                pbar.close()
            return [], [], {}

        if init:
            pbar.close()
            pbar = tqdm.tqdm(total=len(check_frame_list), unit="frames")
        """Scene Batch Detection"""
        for i in range(len(check_frame_list) - 1):
            if init:
                pbar.update(1)
                pbar.set_description(
                    f"Process at Scene Detect Frame {i}")
            i1 = check_frame_data[check_frame_list[i]]
            i2 = check_frame_data[check_frame_list[i + 1]]
            result = self.scene_detection.check_scene(i1, i2)
            if result:
                scene_frame_list.append(check_frame_list[i + 1])  # at i find scene

        if init:
            pbar.close()
            self.logger.info("Start Remove First Batch of Duplicated Frames")

        max_epoch = self.ARGS.remove_dup_mode  # 一直去除到一拍N，N为max_epoch，默认去除一拍二
        opt = []  # 已经被标记，识别的帧
        for queue_size, _ in enumerate(range(1, max_epoch), start=4):
            Icount = queue_size - 1  # 输入帧数
            Current = []  # 该轮被标记的帧
            i = 1
            try:
                while i < len(check_frame_list) - Icount:
                    c = [check_frame_list[p + i] for p in range(queue_size)]  # 读取queue_size帧图像 ~ 对应check_frame_list中的帧号
                    first_frame = c[0]
                    last_frame = c[-1]
                    count = 0
                    for step in range(1, queue_size - 2):
                        pos = 1
                        while pos + step <= queue_size - 2:
                            m0 = c[pos]
                            m1 = c[pos + step]
                            d0 = calc_flow_distance(first_frame, m0, use_flow)
                            d1 = calc_flow_distance(m0, m1, use_flow)
                            d2 = calc_flow_distance(m1, last_frame, use_flow)
                            value_scale = predict_scale(m0, m1)
                            if value_scale * d1 < d0 and value_scale * d1 < d2:
                                count += 1
                            pos += 1
                    if count == (queue_size * (queue_size - 5) + 6) / 2:
                        Current.append(i)  # 加入标记序号
                        i += queue_size - 3
                    i += 1
            except:
                self.logger.error(traceback.format_exc())
            for x in Current:
                if x not in opt:  # 优化:该轮一拍N不可能出现在上一轮中
                    for t in range(queue_size - 3):
                        opt.append(t + x + 1)
        delgen = sorted(set(opt))  # 需要删除的帧
        for d in delgen:
            if check_frame_list[d] not in scene_frame_list:
                check_frame_list[d] = -1

        max_key = np.max(list(input_frame_data.keys()))
        if max_key not in check_frame_list:
            check_frame_list.append(max_key)
        if 0 not in check_frame_list:
            check_frame_list.insert(0, 0)
        check_frame_list = [i for i in check_frame_list if i > -1]
        return check_frame_list, scene_frame_list, input_frame_data

    def rife_run(self):
        """
        Go through all procedures to produce interpolation result
        :return:
        """

        self.logger.info("Activate Remove Duplicate Frames Mode")

        """Get Start Info"""
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.logger.info("Resuming Video Frames...")
        frame_reader = self.generate_frame_reader(start_frame)
        frame_check_reader = self.generate_frame_reader(start_frame, frame_check=True)

        """Get Frames to interpolate"""
        videogen = frame_reader.nextFrame()
        videogen_check = frame_check_reader.nextFrame()

        img1 = self.crop_read_img(Tools.gen_next(videogen_check))
        now_frame_key = start_frame
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}, img_input: {self.ARGS.is_img_input}")

        is_end = False

        self.rife_work_event.set()
        """Start Process"""
        run_time = time.time()
        first_run = True
        while True:
            if is_end or self.main_error:
                break

            if not self.render_thread.is_alive():
                self.logger.critical("Render Thread Dead Unexpectedly")
                break

            if self.ARGS.multi_task_rest and self.ARGS.multi_task_rest_interval and \
                    time.time() - run_time > self.ARGS.multi_task_rest_interval * 3600:
                self.logger.info(
                    f"\n\n INFO - Exceed Run Interval {self.ARGS.multi_task_rest_interval} hour. Time to Rest for 5 minutes!")
                time.sleep(600)
                run_time = time.time()

            check_frame_list, scene_frame_list, input_frame_data = self.remove_duplicate_frames(videogen,
                                                                                                init=first_run)
            input_frame_data = dict(input_frame_data)
            first_run = False
            if not len(check_frame_list):
                while True:
                    img1 = self.crop_read_img(Tools.gen_next(videogen))
                    if img1 is None:
                        is_end = True
                        self.feed_to_rife(now_frame_key, img1, img1, n=0,
                                          is_end=is_end)
                        break
                    self.feed_to_rife(now_frame_key, img1, img1, n=0)
                break

            else:
                img0 = input_frame_data[check_frame_list[0]]
                last_frame_key = check_frame_list[0]
                for frame_cnt in range(1, len(check_frame_list)):
                    img1 = input_frame_data[check_frame_list[frame_cnt]]
                    now_frame_key = check_frame_list[frame_cnt]
                    self.task_info.update({"now_frame": now_frame_key})
                    if now_frame_key in scene_frame_list:
                        self.scene_detection.update_scene_status(now_frame_key, "scene")
                        potential_key = check_frame_list[frame_cnt] - 1
                        if potential_key > 0 and potential_key in input_frame_data:
                            before_img = input_frame_data[potential_key]
                        else:
                            before_img = img0

                        # Scene Review, should be annoted
                        # title = f"try:"
                        # comp_stack = np.hstack((img0, before_img, img1))
                        # cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
                        # cv2.moveWindow(title, 0, 0)
                        # cv2.resizeWindow(title, 1440, 270)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        if frame_cnt < 1 or check_frame_list[frame_cnt] - 1 == check_frame_list[frame_cnt - 1]:
                            self.feed_to_rife(now_frame_key, img0, img0, n=0,
                                              is_end=is_end)
                        elif self.ARGS.is_scdet_mix:
                            self.feed_to_rife(now_frame_key, img0, img1, n=now_frame_key - last_frame_key - 1,
                                              add_scene=True,
                                              is_end=is_end)
                        else:
                            self.feed_to_rife(now_frame_key, img0, before_img, n=now_frame_key - last_frame_key - 2,
                                              add_scene=True,
                                              is_end=is_end)
                    else:
                        self.scene_detection.update_scene_status(now_frame_key, "normal")
                        self.feed_to_rife(now_frame_key, img0, img1, n=now_frame_key - last_frame_key - 1,
                                          is_end=is_end)
                    last_frame_key = now_frame_key
                    img0 = img1
                img1 = input_frame_data[check_frame_list[-1]]  # write last frame since it's not in loop
                self.feed_to_rife(now_frame_key, img1, img1, n=0, is_end=is_end)
                self.task_info.update({"now_frame": check_frame_list[-1]})

        pass
        self.rife_task_queue.put(None)  # bad way to end
        """Wait for Rife and Render Thread to finish"""

    def rife_run_any_fps(self):
        """
        Go through all procedures to produce interpolation result
        :return:
        """

        self.logger.info("Activate Any FPS Mode")

        """Get Start Info"""
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.logger.info("Resuming Video Frames...")
        self.frame_reader = self.generate_frame_reader(start_frame)

        """Get Frames to interpolate"""
        videogen = self.frame_reader.nextFrame()
        img1 = self.crop_read_img(Tools.gen_next(videogen))
        now_frame = start_frame
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}")

        is_end = False

        """Update Interp Mode Info"""
        if self.ARGS.remove_dup_mode == 1:  # 单一模式
            self.ARGS.remove_dup_threshold = self.ARGS.remove_dup_threshold if self.ARGS.remove_dup_threshold > 0.01 else 0.01
        else:  # 0， 不去除重复帧
            self.ARGS.remove_dup_threshold = 0.001

        self.rife_work_event.set()
        """Start Process"""
        run_time = time.time()
        while True:
            if is_end or self.main_error:
                break

            if not self.render_thread.is_alive():
                self.logger.critical("Render Thread Dead Unexpectedly")
                break

            if self.ARGS.multi_task_rest and self.ARGS.multi_task_rest_interval and \
                    time.time() - run_time > self.ARGS.multi_task_rest_interval * 3600:
                self.logger.info(
                    f"\n\n INFO - Exceed Run Interval {self.ARGS.multi_task_rest_interval} hour. Time to Rest for 5 minutes!")
                time.sleep(600)
                run_time = time.time()

            img0 = img1
            img1 = self.crop_read_img(Tools.gen_next(videogen))

            now_frame += 1

            if img1 is None:
                is_end = True
                self.feed_to_rife(now_frame, img0, img0, is_end=is_end)
                break

            diff = Tools.get_norm_img_diff(img0, img1)
            skip = 0  # 用于记录跳过的帧数

            """Find Scene"""
            if self.scene_detection.check_scene(img0, img1):
                self.feed_to_rife(now_frame, img0, img1, n=0,
                                  is_end=is_end)  # add img0 only, for there's no gap between img0 and img1
                self.scene_detection.update_scene_status(now_frame, "scene")
                continue
            else:
                if diff < self.ARGS.remove_dup_threshold:
                    before_img = img1
                    is_scene = False
                    while diff < self.ARGS.remove_dup_threshold:
                        skip += 1
                        self.scene_detection.update_scene_status(now_frame, "dup")

                        img1 = self.crop_read_img(Tools.gen_next(videogen))

                        if img1 is None:
                            img1 = before_img
                            is_end = True
                            break

                        diff = Tools.get_norm_img_diff(img0, img1)

                        is_scene = self.scene_detection.check_scene(img0, img1)  # update scene stack
                        if is_scene:
                            break
                        if skip == self.dup_skip_limit * self.target_fps // self.input_fps:
                            """超过重复帧计数限额，直接跳出"""
                            break

                    # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                    if is_scene:
                        if self.ARGS.is_scdet_mix:
                            self.feed_to_rife(now_frame, img0, img1, n=skip, add_scene=True,
                                              is_end=is_end)
                        else:
                            self.feed_to_rife(now_frame, img0, before_img, n=skip - 1, add_scene=True,
                                              is_end=is_end)
                            """
                            0 (1 2 3) 4[scene] => 0 (1 2) 3 4[scene] 括号内为RIFE应该生成的帧
                            """
                        self.scene_detection.update_scene_status(now_frame, "scene")

                    elif skip != 0:  # skip >= 1
                        assert skip >= 1
                        """Not Scene"""
                        self.feed_to_rife(now_frame, img0, img1, n=skip, is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame, "normal")
                    now_frame += skip + 1
                else:
                    """normal frames"""
                    self.feed_to_rife(now_frame, img0, img1, n=0, is_end=is_end)  # 当前模式下非重复帧间没有空隙，仅输入img0
                    self.scene_detection.update_scene_status(now_frame, "normal")
                self.task_info.update({"now_frame": now_frame})
            pass

        self.rife_task_queue.put(None)  # bad way to end

    def run(self):
        if self.ARGS.concat_only:
            self.concat_all()
        elif self.ARGS.extract_only:
            self.extract_only()
            pass
        elif self.ARGS.render_only:
            self.render_only()
            pass
        else:
            def update_progress():
                nonlocal previous_cnt
                scene_status = self.scene_detection.get_scene_status()

                render_status = self.task_info  # render status quo
                """(chunk_cnt, start_frame, end_frame, frame_cnt)"""

                pbar.set_description(
                    f"Process at Chunk {render_status['chunk_cnt']:0>3d}")
                pbar.set_postfix({"R": f"{render_status['render']}", "C": f"{now_frame}",
                                  "S": f"{scene_status['recent_scene']}",
                                  "SC": f"{self.scene_detection.scdet_cnt}", "TAT": f"{task_acquire_time:.2f}s",
                                  "PT": f"{process_time:.2f}s", "QL": f"{self.rife_task_queue.qsize()}"})
                pbar.update(now_frame - previous_cnt)
                previous_cnt = now_frame
                pass

            """Load RIFE Model"""
            if self.ARGS.use_ncnn:
                self.ARGS.rife_model_name = os.path.basename(self.ARGS.rife_model)
                from Utils import inference_A as inference
            else:
                try:
                    from Utils import inference
                except Exception:
                    self.logger.warning("Import Torch Failed, use NCNN-RIFE instead")
                    traceback.print_exc()
                    self.ARGS.use_ncnn = True
                    self.ARGS.rife_model_name = "rife-v2"
                    from Utils import inference_A as inference

            """Update RIFE Core"""
            self.rife_core = inference.RifeInterpolation(self.ARGS)
            self.rife_core.initiate_rife(self.ARGS)

            if not self.ARGS.use_ncnn:
                self.nvidia_vram_test()

            """Get RIFE Task Thread"""
            if self.ARGS.remove_dup_mode in [0, 1]:
                self.rife_thread = threading.Thread(target=self.rife_run_any_fps, name="[ARGS] RifeTaskThread", )
            else:  # 1, 2 => 去重一拍二或一拍三
                self.rife_thread = threading.Thread(target=self.rife_run, name="[ARGS] RifeTaskThread", )
            self.rife_thread.start()

            chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0

            """Get Renderer"""
            self.render_thread = threading.Thread(target=self.render, name="[ARGS] RenderThread",
                                                  args=(chunk_cnt, start_frame,))
            self.render_thread.start()

            previous_cnt = start_frame
            now_frame = start_frame
            PURE_SCENE_THRESHOLD = 30

            self.rife_work_event.wait()
            pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
            pbar.update(n=start_frame)
            pbar.unpause()
            task_acquire_time = time.time()
            process_time = time.time()
            while True:
                task = self.rife_task_queue.get()
                task_acquire_time = time.time() - task_acquire_time
                if task is None:
                    self.feed_to_render([None], is_end=True)
                    break
                """
                task = {"now_frame", "img0", "img1", "n", "exp","scale", "is_end", "is_scene", "add_scene"}
                """
                # now_frame = task["now_frame"]
                img0 = task["img0"]
                img1 = task["img1"]
                n = task["n"]
                exp = task["exp"]
                scale = task["scale"]
                is_end = task["is_end"]
                add_scene = task["add_scene"]

                debug = False

                if img1 is None:
                    self.feed_to_render([None], is_end=True)
                    break

                if self.ARGS.use_sr and self.ARGS.use_sr_mode == 0:
                    """先超后补"""
                    img0, img1 = self.sr_module.svfi_process(img0), self.sr_module.svfi_process(img1)
                frames_list = [img0]
                if self.ARGS.is_scdet_mix and add_scene:
                    mix_list = Tools.get_mixed_scenes(img0, img1, n + 1)
                    frames_list.extend(mix_list)
                else:
                    if n > 0:
                        if n > PURE_SCENE_THRESHOLD and Tools.check_pure_img(img0):
                            """It's Pure Img Sequence, Copy img0"""
                            for i in range(n):
                                frames_list.append(img0)
                        else:
                            interp_list = self.rife_core.generate_n_interp(img0, img1, n=n, scale=scale, debug=debug)
                            frames_list.extend(interp_list)
                    if add_scene:
                        frames_list.append(img1)

                if self.ARGS.use_sr and self.ARGS.use_sr_mode == 1:
                    """先补后超"""
                    for i in range(len(frames_list)):
                        frames_list[i] = self.sr_module.svfi_process(frames_list[i])
                feed_list = list()
                for i in frames_list:
                    feed_list.append([now_frame, i])
                    now_frame += 1

                self.feed_to_render(feed_list, is_end=is_end)
                process_time = time.time() - process_time
                update_progress()
                process_time = time.time()
                task_acquire_time = time.time()
                if is_end:
                    break

            process_time = 0  # rife's work is done
            task_acquire_time = 0  # task acquire is impossible
            while (self.render_thread is not None and self.render_thread.is_alive()) or \
                    (self.rife_thread is not None and self.rife_thread.is_alive()):
                """等待渲染线程结束"""
                update_progress()
                time.sleep(0.1)

            pbar.update(abs(self.all_frames_cnt - now_frame))
            pbar.close()

            self.logger.info(f"Scedet Status Quo: {self.scene_detection.get_scene_status()}")

            """Check Finished Safely"""
            if self.main_error is not None:
                raise self.main_error

            """Concat the chunks"""
            if not self.ARGS.is_no_concat and not self.ARGS.is_img_output:
                self.concat_all()

        if os.path.exists(self.ARGS.config):
            self.logger.info("Successfully Remove Config File")
            os.remove(self.ARGS.config)
        self.logger.info(f"Program finished at {datetime.datetime.now()}")
        pass

    def extract_only(self):
        chunk_cnt, start_frame = self.check_chunk()
        videogen = self.generate_frame_reader(start_frame)

        img1 = self.crop_read_img(Tools.gen_next(videogen))
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}")

        renderer = ImgSeqIO(folder=self.output, is_read=False,
                            start_frame=self.ARGS.interp_start, logger=self.logger,
                            output_ext=self.ARGS.output_ext, )
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
        pbar.update(n=start_frame)
        img_cnt = 0
        while img1 is not None:
            renderer.writeFrame(img1)
            pbar.update(n=1)
            img_cnt += 1
            pbar.set_description(
                f"Process at Extracting Img {img_cnt}")
            img1 = self.crop_read_img(Tools.gen_next(videogen))

        renderer.close()

    def render_only(self):
        chunk_cnt, start_frame = self.check_chunk()
        videogen = self.generate_frame_reader(start_frame).nextFrame()

        img1 = self.crop_read_img(Tools.gen_next(videogen))
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}")

        render_path = os.path.join(self.output, Tools.get_filename(self.input) + f"_SVFI_Render{self.output_ext}")
        renderer = self.generate_frame_renderer(render_path)
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
        pbar.update(n=start_frame)
        img_cnt = 0

        while img1 is not None:
            if self.ARGS.use_sr:
                img1 = self.sr_module.svfi_process(img1)
            renderer.writeFrame(img1)
            pbar.update(n=1)
            img_cnt += 1
            pbar.set_description(
                f"Process at Rendering {img_cnt}")
            img1 = self.crop_read_img(Tools.gen_next(videogen))

        renderer.close()

    def concat_all(self):
        """
        Concat all the chunks
        :return:
        """

        os.chdir(self.project_dir)
        concat_path = os.path.join(self.project_dir, "concat.ini")
        self.logger.info("Final Round Finished, Start Concating")
        concat_list = list()

        for f in os.listdir(self.project_dir):
            chunk_regex = rf"chunk-[\d+].*?\{self.output_ext}"
            if re.match(chunk_regex, f):
                concat_list.append(os.path.join(self.project_dir, f))
            else:
                self.logger.debug(f"concat escape {f}")

        concat_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))  # sort as start-frame

        if os.path.exists(concat_path):
            os.remove(concat_path)

        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")

        """Check Input file ext"""
        output_ext = os.path.splitext(self.input)[-1]
        if output_ext not in SupportFormat.vid_outputs:
            output_ext = self.output_ext
        if "ProRes" in self.ARGS.render_encoder:
            output_ext = ".mov"

        concat_filepath = f"{os.path.join(self.output, Tools.get_filename(self.input))}"
        concat_filepath += f"_{int(self.target_fps)}fps"  # 输出帧率
        if self.ARGS.is_render_slow_motion:  # 慢动作
            concat_filepath += f"_[SLM_{self.ARGS.render_slow_motion_fps}fps]"
        if self.ARGS.use_deinterlace:
            concat_filepath += f"_[DI]"
        if self.ARGS.use_fast_denoise:
            concat_filepath += f"_[DN]"
        if self.ARGS.use_rife_auto_scale:
            concat_filepath += f"_[SA]"
        else:
            concat_filepath += f"_[S-{self.ARGS.rife_scale}]"  # 全局光流尺度
        if self.ARGS.use_ncnn:
            concat_filepath += "_[NCNN]"
        concat_filepath += f"_[{os.path.basename(self.ARGS.rife_model_name)}]"  # 添加模型信息
        if self.ARGS.use_rife_fp16:
            concat_filepath += "_[FP16]"
        if self.ARGS.is_rife_reverse:
            concat_filepath += "_[RR]"
        if self.ARGS.use_rife_forward_ensemble:
            concat_filepath += "_[RFE]"
        if self.ARGS.use_rife_tta_mode:
            concat_filepath += "_[TTA]"
        if self.ARGS.use_sr:  # 使用超分
            concat_filepath += f"_[SR-{self.ARGS.use_sr_algo}-{self.ARGS.use_sr_model}]"
        if self.ARGS.remove_dup_mode:  # 去重模式
            concat_filepath += f"_[RD-{self.ARGS.remove_dup_mode}]"
        concat_filepath += output_ext  # 添加后缀名

        if self.ARGS.is_save_audio and not self.ARGS.is_img_input:
            audio_path = self.input
            map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy '
            if self.ARGS.input_start_point is not None or self.ARGS.input_end_point is not None:
                time_fmt = "%H:%M:%S"
                start_point = datetime.datetime.strptime(self.ARGS.input_start_point, time_fmt)
                end_point = datetime.datetime.strptime(self.ARGS.input_end_point, time_fmt)
                if end_point > start_point:
                    self.logger.info(
                        f"Update Concat Audio Range: in {self.ARGS.input_start_point} -> out {self.ARGS.input_end_point}")
                    map_audio = f'-ss {self.ARGS.input_start_point} -to {self.ARGS.input_end_point} -i "{audio_path}" -map 0:v:0 -map 1:a? -c:a aac -ab 640k '
                else:
                    self.logger.warning(
                        f"Input Time Section change to origianl course")

        else:
            map_audio = ""

        ffmpeg_command = f'{self.ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy ' \
                         f'{Tools.fillQuotation(concat_filepath)} -metadata title="Made By SVFI {self.ARGS.version}" -y'
        self.logger.debug(f"Concat command: {ffmpeg_command}")
        sp = Tools.popen(ffmpeg_command)
        sp.wait()
        if self.ARGS.is_output_only and os.path.exists(concat_filepath):
            if not os.path.getsize(concat_filepath):
                self.logger.error(f"Concat Error, {output_ext}, empty output")
                raise FileExistsError("Concat Error, empty output, Check Output Extension!!!")
            self.check_chunk(del_chunk=True)

    def concat_check(self, concat_list, concat_filepath):
        """
        Check if concat output is valid
        :param concat_filepath:
        :param concat_list:
        :return:
        """
        original_concat_size = 0
        for f in concat_list:
            original_concat_size += os.path.getsize(f)
        output_concat_size = os.path.getsize(concat_filepath)
        if output_concat_size < original_concat_size * 0.9:
            return False
        return True


interpworkflow = InterpWorkFlow(ARGS)
interpworkflow.run()
sys.exit(0)
