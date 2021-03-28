import datetime
import json
import logging
import math
import os
import re
import traceback

import cv2
import torch
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor, QIcon
from pprint import pprint, pformat
from Utils import RIFE_GUI
from Utils.utils import Utils, EncodePresetAssemply
import sys
import subprocess as sp
import shlex
import time

MAC = True
try:
    from PyQt5.QtGui import qt_mac_set_native_menubar
except ImportError:
    MAC = False

Utils = Utils()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
ddname = os.path.dirname(abspath)
appDataPath = os.path.join(dname, "RIFE_GUI.ini")
appData = QSettings(appDataPath, QSettings.IniFormat)
appData.setIniCodec("UTF-8")

logger = Utils.get_logger("GUI", dname)
ols_potential = os.path.join(dname, "one_line_shot_args.exe")
appData.setValue("OneLineShotPath", ols_potential)
appData.setValue("ffmpeg", dname)
appData.setValue("model", os.path.join(ddname, "train_log"))
if not os.path.exists(ols_potential):
    appData.setValue("OneLineShotPath",
                     r"D:\60-fps-Project\arXiv2020-RIFE-main\RIFE_GUI\one_line_shot_args_v6.2.3.py")
    appData.setValue("ffmpeg", "ffmpeg")
    appData.setValue("model", r"D:\60-fps-Project\arXiv2020-RIFE-main\RIFE_GUI\Utils\train_log")
    logger.info("Change to Debug Path")


class RIFE_Run_Other_Threads(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, command, parent=None):
        super(RIFE_Run_Other_Threads, self).__init__(parent)
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        self.command = command

    def run(self):
        logger.info(f"[CMD Thread]: Start execute {self.command}")
        os.system(self.command)
        pass

    pass


class MyLineWidget(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            url = e.mimeData().urls()[0]
            self.setText(url.toLocalFile())
        else:
            e.ignore()


class MyListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            self.clear()
            for url in e.mimeData().urls():
                self.addItem(url.toLocalFile())
        else:
            e.ignore()

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            self.clear()
            for url in e.mimeData().urls():
                self.addItem(url.toLocalFile())
        else:
            e.ignore()

    def get_items(self):
        widgetres = []
        # 获取listwidget中条目数
        count = self.count()
        # 遍历listwidget中的内容
        for i in range(count):
            widgetres.append(self.item(i).text())
        return widgetres


class MyTextWidget(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, event):
        try:
            if event.mimeData().hasUrls:
                url = event.mimeData().urls()[0]
                # url_list = list()
                # url_list.append(self.toPlainText().strip(";"))
                # for url in event.mimeData().urls():
                #     url_list.append(f"{url.toLocalFile()}")
                # text = ""
                # for url in url_list:
                #     text += f"{url};"
                # text = text.strip(";")
                self.setText(f"{url.toLocalFile()}")
            else:
                event.ignore()
        except Exception as e:
            print(e)


class RIFE_Run_Thread(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, parent=None, concat_only=False):
        super(RIFE_Run_Thread, self).__init__(parent)
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        self.concat_only = concat_only
        self.command = ""
        self.current_proc = None
        self.kill = False
        self.all_cnt = 0
        self.silent = False
        self.tqdm_re = re.compile("Process at .*?\]")

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def build_command(self, input_file):
        if os.path.splitext(appData.value("OneLineShotPath"))[-1] == ".exe":
            self.command = appData.value("OneLineShotPath") + " "
        else:
            self.command = f'python {appData.value("OneLineShotPath")} '
        if not len(input_file) or not os.path.exists(input_file):
            self.command = ""
            return ""
        if appData.value("fps", type=float) <= 0 or appData.value("target_fps", type=float) <= 0:
            return ""

        self.command += f'--input {self.fillQuotation(input_file)} '
        if os.path.isfile(appData.value("output")):
            logger.info("[GUI]: OutputPath with FileName detected")
            output_path = appData.value("output")
            appData.setValue("output", os.path.basename(output_path))
        self.command += f'--output {self.fillQuotation(appData.value("output"))} '
        self.command += f'--config {self.fillQuotation(appDataPath)} '
        if self.concat_only:
            self.command += f"--concat-only "

        self.command = self.command.replace("\\", "/")
        return self.command

    def update_status(self, current_step, finished=False, notice="", sp_status="", returncode=-1):
        """
        update sub process status
        :return:
        """
        emit_json = {"cnt": self.all_cnt, "current": current_step, "finished": finished,
                     "notice": notice, "subprocess": sp_status, "returncode": returncode}
        emit_json = json.dumps(emit_json)
        self.run_signal.emit(emit_json)

    def run(self):
        logger.info("[GUI]: Start")

        file_list = appData.value("InputFileName", "").split(";")

        command_list = list()
        for f in file_list:
            command = self.build_command(f)
            if not len(command):
                continue
            command_list.append((f, command))

        current_step = 0
        self.all_cnt = len(command_list)
        if self.all_cnt > 1:
            """MultiTask"""
            appData.setValue("output_only", True)
        if not self.all_cnt:
            logger.info("[GUI]: Task List Empty, Please Check Your Settings! (input fps for example)")
            self.update_status(current_step, True, "\nTask List is Empty!\n")
            return
        interval_time = time.time()
        try:
            for f in command_list:
                logger.info(f"[GUI]: Designed Command:\n{f}")
                # if appData.value("debug", type=bool):
                #     logger.info(f"DEBUG: {f[1]}")
                #     continue
                proc_args = shlex.split(f[1])
                self.current_proc = sp.Popen(args=proc_args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding='gbk',
                                             universal_newlines=True)
                flush_lines = ""
                while self.current_proc.poll() is None:
                    if self.kill:
                        self.current_proc.terminate()
                        self.update_status(current_step, False, notice=f"\n\nWARNING, 补帧已被强制结束", returncode=-1)
                        break
                    line = self.current_proc.stdout.readline()
                    self.current_proc.stdout.flush()
                    flush_lines += line.strip("").strip("\r").replace("[A", "")
                    if "error" in flush_lines.lower():
                        print(flush_lines)
                    if len(flush_lines) and time.time() - interval_time > 0.1:
                        flush_lines = "\n".join(list(set(list(flush_lines.splitlines()))))
                        interval_time = time.time()
                        self.update_status(current_step, False, sp_status=f"{flush_lines}")
                        flush_lines = ""

                current_step += 1
                self.update_status(current_step, False, f"\nINFO - {datetime.datetime.now()} {f[0]} 完成\n\n")
                appData.setValue("chunk", 1)
                appData.setValue("interp_start", 0)
                appData.setValue("interp_cnt", 1)
                # Multi tasks, remove last settings

        except Exception:
            logger.error(traceback.format_exc())

        self.update_status(current_step, True, returncode=self.current_proc.returncode)
        logger.info("[GUI]: Tasks Finished")
        pass

    def kill_proc_exec(self):
        self.kill = True
        logger.info("Kill Process Command Fired")

    pass


class RIFE_GUI_BACKEND(QDialog, RIFE_GUI.Ui_RIFEDialog):
    kill_proc = pyqtSignal(int)
    notfound = pyqtSignal(int)

    def __init__(self, parent=None):
        super(RIFE_GUI_BACKEND, self).__init__()
        self.setupUi(self)
        self.thread = None
        self.init_before_settings()
        self.Exp = int(math.log(float(appData.value("exp", "2")), 2))

        if appData.value("ffmpeg") != "ffmpeg":
            self.ffmpeg = os.path.join(appData.value("ffmpeg"), "ffmpeg.exe")
        else:
            self.ffmpeg = appData.value("ffmpeg")

        if os.path.exists(appDataPath):
            logger.info("[GUI]: Previous Settings, Found, Loaded")

        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
        self.check_gpu = False
        self.silent = False
        self.tqdm_re = re.compile(".*?Process at .*?\]")
        self.current_failed = False
        self.formatted_option_check = []

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def sendWarning(self, title, string, msg_type=1):
        """

        :param title:
        :param string:
        :param msg_type: 1 warning 2 info
        :return:
        """
        if self.silent:
            return
        QMessageBox.setWindowIcon(self, QIcon('ico.ico'))
        if msg_type == 1:
            reply = QMessageBox.warning(self,
                                        f"{title}",
                                        f"{string}",
                                        QMessageBox.Yes)
        elif msg_type == 2:
            reply = QMessageBox.information(self,
                                            f"{title}",
                                            f"{string}",
                                            QMessageBox.Yes)
        else:
            return
        return reply

    def select_file(self, filename, folder=False, _filter=None, multi=False):
        if folder:
            directory = QFileDialog.getExistingDirectory(None, caption="选取文件夹")
            return directory
        if multi:
            files = QFileDialog.getOpenFileNames(None, caption=f"选择{filename}", filter=_filter)
            return files[0]
        directory = QFileDialog.getOpenFileName(None, caption=f"选择{filename}", filter=_filter)
        return directory[0]

    def quick_concat(self):
        input_v = self.ConcatInputV.text()
        input_a = self.ConcatInputA.text()
        output_v = self.OutputConcat.text()
        self.load_current_settings()
        if not input_v or not input_a or not output_v:
            self.sendWarning("Parameters unfilled", "请填写输入或输出视频路径！")
            return

        ffmpeg_command = f"""
            {self.ffmpeg} -i {self.fillQuotation(input_a)} -i {self.fillQuotation(input_v)} 
            -map 1:v:0 -map 0:a:0 -c:v copy -c:a copy -shortest {self.fillQuotation(output_v)} -y
        """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"[GUI] concat {ffmpeg_command}")
        os.system(ffmpeg_command)
        self.sendWarning("音视频合并操作完成！", f"请查收", msg_type=2)

    def quick_gif(self):
        input_v = self.GifInput.text()
        output_v = self.GifOutput.text()
        self.load_current_settings()
        if not input_v or not output_v:
            self.sendWarning("Parameters unfilled", "请填写输入或输出视频路径！")
            return
        palette_path = os.path.join(os.path.dirname(input_v), "palette.png")
        ffmpeg_command = f"""
                    {self.ffmpeg} -hide_banner -i {self.fillQuotation(input_v)} -vf "palettegen=stats_mode=diff" -y {palette_path}
                """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"ffmpeg_command for create palette: {ffmpeg_command}")
        os.system(ffmpeg_command)
        if not appData.value("target_fps"):
            appData.setValue("target_fps", 48)
            logger.info("Not find output GIF fps, Auto set GIF output fps to 48 as it's smooth enough")
        ffmpeg_command = f"""
                           {self.ffmpeg} -hide_banner -i {self.fillQuotation(input_v)} -i {palette_path} -r {appData.value("target_fps")} -lavfi "fps={appData.value("target_fps")},scale=960:-1[x];[x][1:v]paletteuse=dither=floyd_steinberg" {self.fillQuotation(output_v)} -y
                        """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"[GUI] create gif: {ffmpeg_command}")
        os.system(ffmpeg_command)
        self.sendWarning("GIF操作完成！", f'GIF帧率:{appData.value("target_fps")}', 2)

    def set_start_info(self, sf, scf, sc):
        """

        :return:
        """
        self.StartFrame.setText(str(sf))
        self.StartCntFrame.setText(str(scf))
        self.StartChunk.setText(str(sc))
        return

    def auto_set_fps(self, sample_file):
        currentExp = self.ExpSelecter.currentText()[1:]
        try:
            input_stream = cv2.VideoCapture(sample_file)
            input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            self.InputFPS.setText(f"{input_fps:.5f}")
            self.OutputFPS.setText(f"{float(input_fps) * float(currentExp):.5f}")
        except Exception:
            logger.error(traceback.format_exc())

    def auto_set(self):
        chunk_list = list()
        output_dir = self.OutputFolder.toPlainText()
        input_files = self.load_input_files()
        if not len(input_files):
            self.sendWarning("Select Item first", "请先输入文件")
            self.set_start_info(0, 1, 1)
            return
        input_file = input_files[0]
        if not len(output_dir):
            logger.info("OutputFolder path is empty, pls enter it first")
            self.sendWarning("Parameters unfilled", "请移步基础设置填写输出文件夹！")
            return
        if os.path.isfile(output_dir):
            output_dir = os.path.dirname(output_dir)
            self.OutputFolder.setText(output_dir)

        if not os.path.exists(output_dir) or not os.path.exists(input_file):
            logger.info("Not Exists OutputFolder")
            self.sendWarning("Output Folder Not Found", "输入文件或输出文件夹不存在！请确认输入")
            return

        for f in os.listdir(output_dir):
            if re.match("chunk-[\d+].*?\.(mp4|mov)", f):
                chunk_list.append(os.path.join(output_dir, f))
        if not len(chunk_list):
            self.set_start_info(0, 1, 1)
            logger.info("AutoSet find None")
            return
        logger.info("Found Previous Chunks")
        chunk_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))
        last_chunk = chunk_list[-1]
        os.remove(last_chunk)
        chunk_list.pop(-1)
        if not len(chunk_list):
            self.set_start_info(0, 1, 1)
            logger.info("AutoSet Remove and found none")
            return
        last_chunk = chunk_list[-1]
        match_result = re.findall("chunk-(\d+)-(\d+)-(\d+)\.(mp4|mov)", last_chunk)[0]

        chunk = int(match_result[0])
        first_frame = int(match_result[1])
        last_frame = int(match_result[2])
        first_interp_cnt = (last_frame + 1) * (2 ** self.Exp) + 1
        self.set_start_info(last_frame + 1, first_interp_cnt, chunk + 1)
        logger.info("AutoSet Ready")

        pass

    def get_filename(self, path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    def get_gpu_info(self):
        json_output = os.path.join(dname, "NVIDIA_info.json")
        infos = {}
        for i in range(torch.cuda.device_count()):
            card = torch.cuda.get_device_properties(i)
            info = f"{card.name}, {card.total_memory / 1024 ** 3:.1f} GB"
            infos[f"{i}"] = info
        with open(json_output, "w", encoding="UTF-8") as w:
            json.dump(infos, w)
        logger.info(f"NVIDIA data: {infos}")
        return infos

    def get_model_info(self):
        model_dir = appData.value("model")
        if not os.path.exists(model_dir):
            logger.info(f"Not find Module dir at {model_dir}")
            self.sendWarning("Model Dir Not Found", "未找到补帧模型路径，请检查！")
            return
        model_list = list()
        for m in os.listdir(model_dir):
            if not os.path.isfile(os.path.join(model_dir, m)):
                model_list.append(m)
        model_list.reverse()
        self.ModuleSelector.clear()
        for mod in model_list:
            self.ModuleSelector.addItem(f"{mod}")

    def init_before_settings(self):
        input_list = appData.value("InputFileName", "").split(";")
        for i in input_list:
            if len(i):
                self.InputFileName.addItem(i)
        self.OutputFolder.setText(appData.value("output"))
        self.InputFPS.setText(appData.value("fps", "0"))
        self.OutputFPS.setText(appData.value("target_fps"))
        self.CRFSelector.setValue(appData.value("crf", 16, type=int))
        self.ExpSelecter.setCurrentText("x" + str(2 ** int(appData.value("exp", "1"))))
        self.BitrateSelector.setValue(appData.value("bitrate", 90, type=int))
        self.PresetSelector.setCurrentText(appData.value("preset", "fast[软编, 硬编]"))
        self.HwaccelChecker.setChecked(appData.value("hwaccel", False, type=bool))
        self.CloseScedetChecker.setChecked(appData.value("no_scdet", False, type=bool))
        self.ScdetSelector.setValue(appData.value("scdet_threshold", 30, type=int))
        self.DupRmChecker.setChecked(appData.value("remove_dup", False, type=bool))
        self.DupFramesTSelector.setValue(appData.value("dup_threshold", 1.00, type=float))
        # self.HDRChecker.setChecked(appData.value("HDRChecker", False, type=bool))
        self.CropSettings.setText(appData.value("crop"))
        self.ResizeSettings.setText(appData.value("resize"))
        self.FFmpegCustomer.setText(appData.value("ffmpeg_customized", ""))

        self.SaveAudioChecker.setChecked(appData.value("save_audio", True, type=bool))
        self.QuickExtractChecker.setChecked(appData.value("quick_extract", True, type=bool))
        self.ImgOutputChecker.setChecked(appData.value("img_output", False, type=bool))
        self.ImgInputChecker.setChecked(appData.value("img_input", False, type=bool))
        self.NoConcatChecker.setChecked(appData.value("no_concat", False, type=bool))
        self.OutputOnlyChecker.setChecked(appData.value("output_only", False, type=bool))

        self.FP16Checker.setChecked(appData.value("fp16", False, type=bool))
        self.ReverseChecker.setChecked(appData.value("reverse", False, type=bool))
        self.UseCUDAButton.setChecked(appData.value("UseCUDAButton", False, type=bool))
        self.UseCRF.setChecked(appData.value("UseCRF", True, type=bool))
        self.UseTargetBitrate.setChecked(appData.value("UseTargetBitrate", False, type=bool))
        self.InterpScaleSelector.setCurrentText(appData.value("scale", "1.00"))
        self.EncoderSelector.setCurrentText(appData.value("encoder", "HEVC"))
        self.j_settings.setText(appData.value("j_settings", "2:4:4"))
        self.slowmotion.setChecked(appData.value("slow_motion", False, type=bool))

        pos = appData.value("pos", QVariant(QPoint(1920, 1080)))
        size = appData.value("size", QVariant(QSize(400, 400)))
        self.resize(size)
        self.move(pos)

    def load_input_files(self):
        widgetres = []
        # 获取listwidget中条目数
        count = self.InputFileName.count()
        # 遍历listwidget中的内容
        for i in range(count):
            widgetres.append(self.InputFileName.item(i).text())
        return widgetres

    def load_current_settings(self):
        input_file_names = ""
        for i in self.load_input_files():
            if len(i):
                input_file_names += f"{i};"
        appData.setValue("InputFileName", input_file_names)
        appData.setValue("output", self.OutputFolder.toPlainText())
        appData.setValue("fps", self.InputFPS.text())
        appData.setValue("target_fps", self.OutputFPS.text())
        appData.setValue("crf", self.CRFSelector.value())
        appData.setValue("exp", int(math.log(int(self.ExpSelecter.currentText()[1:]), 2)))
        appData.setValue("bitrate", self.BitrateSelector.value())
        appData.setValue("preset", self.PresetSelector.currentText())
        appData.setValue("encoder", self.EncoderSelector.currentText())
        appData.setValue("hwaccel", self.HwaccelChecker.isChecked())
        appData.setValue("no_scdet", self.CloseScedetChecker.isChecked())
        appData.setValue("scdet_threshold", self.ScdetSelector.value())
        appData.setValue("remove_dup", self.DupRmChecker.isChecked())
        appData.setValue("dup_threshold", self.DupFramesTSelector.value())
        # appData.setValue("HDRChecker", self.HDRChecker.isChecked())
        appData.setValue("crop", self.CropSettings.text())
        appData.setValue("resize", self.ResizeSettings.text())

        appData.setValue("save_audio", self.SaveAudioChecker.isChecked())
        appData.setValue("quick_extract", self.QuickExtractChecker.isChecked())
        appData.setValue("img_output", self.ImgOutputChecker.isChecked())
        appData.setValue("img_input", self.ImgInputChecker.isChecked())
        appData.setValue("no_concat", self.NoConcatChecker.isChecked())
        appData.setValue("output_only", self.OutputOnlyChecker.isChecked())
        appData.setValue("fp16", self.FP16Checker.isChecked())
        appData.setValue("reverse", self.ReverseChecker.isChecked())
        appData.setValue("ncnn", self.UseNCNNButton.isChecked())
        appData.setValue("use_cpu", self.UseCPUButton.isChecked())
        appData.setValue("UseMultiCUDAButton", self.UseMultiCUDAButton.isChecked())
        appData.setValue("UseCUDAButton", self.UseCUDAButton.isChecked())
        appData.setValue("UseCRF", self.UseCRF.isChecked())
        appData.setValue("UseTargetBitrate", self.UseTargetBitrate.isChecked())

        appData.setValue("scale", self.InterpScaleSelector.currentText())
        appData.setValue("encoder", self.EncoderSelector.currentText())
        appData.setValue("pix_fmt", self.PixFmtSelector.currentText())

        appData.setValue("chunk", self.StartChunk.text() if len(self.StartChunk.text()) else 1)
        appData.setValue("interp_start", self.StartFrame.text() if len(self.StartFrame.text()) else 0)
        appData.setValue("interp_cnt", self.StartCntFrame.text() if len(self.StartCntFrame.text()) else 1)
        appData.setValue("render_gap", 1000)
        # TODO Selected GPU
        appData.setValue("SelectedModel", os.path.join(appData.value("model"), self.ModuleSelector.currentText()))
        appData.setValue("use_specific_gpu", self.DiscreteCardSelector.currentIndex())
        appData.setValue("pos", QVariant(self.pos()))
        appData.setValue("size", QVariant(self.size()))
        appData.setValue("ffmpeg_customized", self.FFmpegCustomer.text())
        appData.setValue("debug", self.DebugChecker.isChecked())
        appData.setValue("j_settings", self.j_settings.text().replace("：", ":"))
        appData.setValue("slow_motion", self.slowmotion.isChecked())
        if appData.value("slow_motion", False, type=bool):
            appData.setValue("save_audio", False)
            self.SaveAudioChecker.setChecked(False)

        logger.info("[Main]: Download all settings")
        status_check = "[当前导出设置预览]\n\n"
        for key in appData.allKeys():
            status_check += f"{key} => {appData.value(key)}\n"
        if not len(self.OptionCheck.toPlainText()):
            self.OptionCheck.setText(status_check)
        self.OptionCheck.isReadOnly = True
        appData.sync()
        pass

    def update_rife_process(self, json_data):
        """
        Communicate with RIFE Thread
        :return:
        """
        data = json.loads(json_data)
        now_text = self.OptionCheck.toPlainText()
        self.progressBar.setMaximum(int(data["cnt"]))
        self.progressBar.setValue(int(data["current"]))
        if len(data.get("notice", "")):
            now_text += data["notice"] + "\n"

        if len(data.get("subprocess", "")):
            # now_text = re.sub("\n+", "\n", now_text)
            if self.tqdm_re.match(data["subprocess"]) is not None or "Process at" in data["subprocess"]:
                data["subprocess"] = data["subprocess"].splitlines()[-1]
                now_text = self.tqdm_re.sub("", now_text)
                # now_text = re.sub(".*?Process at .*?\n", "", now_text)
            t = data["subprocess"]
            now_text += t
        self.OptionCheck.setText(now_text)

        new_text = []
        option_html = re.sub(r"font-weight:.*?;", "", self.OptionCheck.toHtml())
        option_html = re.sub(r"color:.*?;", "", option_html)

        for t in option_html.splitlines():
            t = re.sub(r">(.*?Process at.*?)</p>",
                       lambda x: f'><span style=" font-weight:600;">%s</span></p>' % x.group(1), t, flags=re.I)
            t = re.sub(r">(.*?INFO.*?)</p>",
                       lambda x: f'><span style=" font-weight:600; color:#0000ff;">%s</span></p>' % x.group(1), t,
                       flags=re.I)
            t = re.sub(r">(.*?ERROR.*?)</p>",
                       lambda x: f'><span style=" font-weight:600; color:#ff0000;">%s</span></p>' % x.group(1), t,
                       flags=re.I)
            t = re.sub(r">(.*?Critical.*?)</p>",
                       lambda x: f'><span style=" font-weight:600; color:#ff0000;">%s</span></p>' % x.group(1), t,
                       flags=re.I)
            t = re.sub(r">(.*?fail.*?)</p>",
                       lambda x: f'><span style=" font-weight:600; color:#ff0000;">%s</span></p>' % x.group(1), t,
                       flags=re.I)
            t = re.sub(r">(.*?WARN.*?)</p>",
                       lambda x: f'><span style=" font-weight:600; color:#ffaa00;">%s</span></p>' % x.group(1), t,
                       flags=re.I)
            t = re.sub(r">(.*?Duration.*?)</p>",
                       lambda x: f'><span style=" font-weight:600; color:#550000;">%s</span></p>' % x.group(1), t,
                       flags=re.I)
            t = re.sub(r"><span .*?><span .*?>(.*?Program finished.*?)</span></span></p>",
                       lambda x: f'><span style=" font-weight:600; color:#55aa00;">%s</span></p>' % x.group(1), t,
                       flags=re.I)

            new_text.append(t)
        now_text = "\n".join(new_text)

        if data["finished"]:
            """Error Handle"""
            if "CUDA out of memory" in now_text and not self.current_failed:
                self.sendWarning("CUDA Failed", "你的显存不够啦！快去'高级设置'把补帧精度调低/降低视频分辨率/使用半精度模式~", )
                self.current_failed = True
            if "error" in data.get("subprocess", "").lower() and not self.current_failed:
                self.sendWarning("Something Went Wrong", f"程序运行出现错误！\n{data.get('subprocess')}\n联系开发人员解决", )
                self.current_failed = True

            returncode = data["returncode"]
            self.sendWarning("补帧任务完成", f"共 {data['cnt']} 个补帧任务\n"
                                       f"{'成功！' if returncode == 0 else f'失败, 返回码：{returncode}'}\n"
                                       f"请尝试前往高级设置恢复补帧进度", 2)
            self.ProcessStart.setEnabled(True)
            self.ConcatAllButton.setEnabled(True)
        self.OptionCheck.setText(now_text)
        self.OptionCheck.moveCursor(QTextCursor.End)

    def on_InputFileName_currentItemChanged(self):
        if self.InputFileName.currentItem() is None:
            return
        text = self.InputFileName.currentItem().text().strip('"')
        self.InputFileName.disconnect()
        if text == "":
            return
        """empty text"""
        if self.ImgInputChecker.isChecked():
            "ignore img input"
            return
        input_filename = text.strip(";").split(";")[0]
        self.auto_set_fps(input_filename)

        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
        return

    @pyqtSlot(bool)
    def on_InputButton_clicked(self):
        try:
            self.InputFileName.disconnect()
        except TypeError:
            pass
        self.InputFileName.clear()
        if self.ImgInputChecker.isChecked():
            input_directory = self.select_file("要补帧的图片序列文件夹", folder=True)
            self.InputFileName.addItem(input_directory)
            self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
            return
        input_files = self.select_file('要补帧的视频', multi=True)
        for f in input_files:
            self.InputFileName.addItem(f)
        if len(input_files):
            self.OutputFolder.setText(os.path.dirname(input_files[0]))
            sample_file = input_files[0]
            self.auto_set_fps(sample_file)
        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)

    @pyqtSlot(bool)
    def on_OutputButton_clicked(self):
        folder = self.select_file('要输出项目的文件夹', folder=True)
        if os.path.isfile(folder):
            folder = os.path.dirname(folder)
        self.OutputFolder.setText(folder)

    @pyqtSlot(bool)
    def on_ImgInputChecker_clicked(self):
        if self.ImgInputChecker.isChecked():
            self.SaveAudioChecker.setChecked(False)

    @pyqtSlot(bool)
    def on_SaveAudioChecker_clicked(self):
        if self.ImgInputChecker.isChecked():
            self.SaveAudioChecker.setChecked(False)
            self.sendWarning("ImgInput Detected", "图片输入模式下不可带音频！")

    @pyqtSlot(bool)
    def on_AutoSet_clicked(self):
        self.auto_set()

    @pyqtSlot(bool)
    def on_ConcatButton_clicked(self):
        if not self.ConcatInputV.text():
            input_filename = self.select_file('请输入要进行音视频合并的视频文件')
            self.ConcatInputV.setText(input_filename)
            self.ConcatInputA.setText(input_filename)
            self.OutputConcat.setText(
                os.path.join(os.path.dirname(input_filename), f"{self.get_filename(input_filename)}_concat.mp4"))
            return
        self.quick_concat()
        # self.auto_set()
        pass

    @pyqtSlot(bool)
    def on_GifButton_clicked(self):
        if not self.GifInput.text():
            input_filename = self.select_file('请输入要制作成gif的视频文件')
            self.GifInput.setText(input_filename)
            self.GifOutput.setText(
                os.path.join(os.path.dirname(input_filename), f"{self.get_filename(input_filename)}.gif"))
            return
        self.quick_gif()
        pass

    @pyqtSlot(bool)
    def on_HwaccelChecker_clicked(self):
        logger.info("Switch To HWACCEL Mode: %s" % self.HwaccelChecker.isChecked())
        self.on_EncoderSelector_currentTextChanged("")

    @pyqtSlot(str)
    def on_EncoderSelector_currentTextChanged(self, currentEncoder):
        self.PresetSelector.clear()
        self.PixFmtSelector.clear()
        currentEncoder = self.EncoderSelector.currentText()
        presets = []
        pixfmts = []
        if currentEncoder == "HEVC":
            if not self.HwaccelChecker.isChecked():
                # x265
                presets = EncodePresetAssemply.preset["HEVC"]["x265"]
                pixfmts = EncodePresetAssemply.pixfmt["HEVC"]["x265"]
            else:
                # hevc_nvenc
                presets = EncodePresetAssemply.preset["HEVC"]["NVENC"]
                pixfmts = EncodePresetAssemply.pixfmt["HEVC"]["NVENC"]
        elif currentEncoder == "H264":
            if not self.HwaccelChecker.isChecked():
                # x265
                presets = EncodePresetAssemply.preset["H264"]["x264"]
                pixfmts = EncodePresetAssemply.pixfmt["H264"]["x264"]
            else:
                # hevc_nvenc
                presets = EncodePresetAssemply.preset["H264"]["NVENC"]
                pixfmts = EncodePresetAssemply.pixfmt["H264"]["NVENC"]
        elif currentEncoder == "ProRes":
            presets = EncodePresetAssemply.preset["ProRes"]
            pixfmts = EncodePresetAssemply.pixfmt["ProRes"]

        for preset in presets:
            self.PresetSelector.addItem(preset)
        for pixfmt in pixfmts:
            self.PixFmtSelector.addItem(pixfmt)

    @pyqtSlot(str)
    def on_ExpSelecter_currentTextChanged(self, currentExp):
        input_files = self.load_input_files()
        if not len(input_files):
            return
        input_filename = input_files[0]
        self.auto_set_fps(input_filename)

    @pyqtSlot(str)
    def on_InputFPS_textChanged(self, text):
        currentExp = self.ExpSelecter.currentText()[1:]
        try:
            self.OutputFPS.setText(f"{float(text) * float(currentExp):.5f}")
        except ValueError:
            self.sendWarning("Pls Enter Valid InputFPS", "请输入正常的视频帧率")

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, tabIndex):
        if tabIndex in [2, 3]:
            """Step 3"""
            if tabIndex == 2:
                self.progressBar.setValue(0)
            logger.info("[Main]: Start Loading Settings")
            self.load_current_settings()
            # self.sendWarning("再三强调！！！", "一定要把命令行窗口拉长！不然看不到补帧进度！")

        if tabIndex in [1] and not self.check_gpu:
            gpu_info = self.get_gpu_info()
            self.on_EncoderSelector_currentTextChanged("")
            self.get_model_info()
            if not len(gpu_info):
                self.sendWarning("No NVIDIA Card Found", "未找到任何N卡，请检查驱动")
                return
            self.DiscreteCardSelector.clear()
            for gpu in gpu_info:
                self.DiscreteCardSelector.addItem(f"{gpu}: {gpu_info[gpu]}")
            self.check_gpu = True
            pass

    @pyqtSlot(bool)
    def on_ProcessStart_clicked(self):
        self.ProcessStart.setEnabled(False)
        self.progressBar.setValue(0)
        RIFE_thread = RIFE_Run_Thread()
        RIFE_thread.run_signal.connect(self.update_rife_process)
        RIFE_thread.start()
        self.thread = RIFE_thread
        self.OptionCheck.setText("[补帧操作启动，请移步命令行查看进度详情]\n显示“Program finished”则任务完成\n"
                                 "如果遇到任何问题，请将命令行（黑色界面）和软件运行界面的Step1、Step2、Step3截图并联系开发人员解决，"
                                 "群号在首页说明\n\n\n\n\n")
        self.current_failed = False

    @pyqtSlot(bool)
    def on_AllInOne_clicked(self):
        """
        Alas
        :return:
        """
        video = self.load_input_files()
        output_dir = self.OutputFolder.toPlainText()
        if not len(video) or not len(output_dir):
            self.sendWarning("Empty Input", "请先拖入要补帧的文件并确认输出文件夹")
            return
        self.silent = True
        if not self.ImgInputChecker.isChecked():
            self.on_ExpSelecter_currentTextChanged(self.ExpSelecter.currentText())
        self.auto_set()
        self.on_tabWidget_currentChanged(1)
        self.on_tabWidget_currentChanged(2)
        self.on_EncoderSelector_currentTextChanged("")
        self.on_ProcessStart_clicked()
        self.tabWidget.setCurrentIndex(2)
        self.silent = False

    @pyqtSlot(bool)
    def on_ConcatAllButton_clicked(self):
        """

        :return:
        """
        self.ConcatAllButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(2)
        self.progressBar.setValue(0)
        RIFE_thread = RIFE_Run_Thread(concat_only=True)
        RIFE_thread.run_signal.connect(self.update_rife_process)
        RIFE_thread.start()
        self.thread = RIFE_thread
        self.OptionCheck.setText("[仅合并操作启动，请移步命令行查看进度详情]\n显示“Program finished”则任务完成\n"
                                 "如果遇到任何问题，请将命令行（黑色界面）和软件运行界面的Step1、Step2、Step3截图并联系开发人员解决，"
                                 "群号在首页说明\n\n\n\n\n")

    @pyqtSlot(bool)
    def on_KillProcButton_clicked(self):
        """
        :return:
        """
        if self.thread is not None:
            self.thread.kill_proc_exec()

    @pyqtSlot(bool)
    def on_CloseButton_clicked(self):
        self.load_current_settings()
        sys.exit()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        form = RIFE_GUI_BACKEND()
        form.show()
        app.exec_()
    except Exception:
        logger.critical(traceback.format_exc())
        sys.exit()
