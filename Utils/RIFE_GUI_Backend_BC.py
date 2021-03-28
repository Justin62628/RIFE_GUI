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
from pprint import pprint, pformat
from Utils import RIFE_GUI
from Utils.utils import Utils, ImgSeqIO, VideoInfo, CommandResult
import sys
import subprocess as sp
import shlex

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
appData.setValue("FFmpeg", dname)
appData.setValue("Model", os.path.join(ddname, "train_log"))
if not os.path.exists(ols_potential):
    appData.setValue("OneLineShotPath",
                     r"D:\60-fps-Project\arXiv2020-RIFE-main\RIFE_GUI\one_line_shot_args_v6.2alpha.py")
    appData.setValue("FFmpeg", "ffmpeg")
    appData.setValue("Model", r"D:\60-fps-Project\arXiv2020-RIFE-main\RIFE_GUI\Utils\train_log")
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
    kill_proc = pyqtSignal(int)
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

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def build_command(self, input_file):
        if os.path.splitext(appData.value("OneLineShotPath"))[-1] == ".exe":
            self.command = appData.value("OneLineShotPath") + " "
        else:
            self.command = f'python {appData.value("OneLineShotPath")}     '
        if not len(input_file) or not os.path.exists(input_file):
            self.command = ""
            return ""
        if appData.value("InputFPS", type=float) <=0 or  appData.value("OutputFPS", type=float)<=0 :
            return ""

        self.command += f'--input {self.fillQuotation(input_file)} '
        if os.path.isfile(appData.value("OutputFolder")):
            logger.info("[GUI]: OutputPath with FileName detected")
            output_path = appData.value("OutputFolder")
            appData.setValue("OutputFolder", os.path.basename(output_path))
        self.command += f'--output {self.fillQuotation(appData.value("OutputFolder"))} '
        self.command += f'--fps {appData.value("InputFPS")} '
        if appData.value("OutputFPS"):
            self.command += f'--target-fps {appData.value("OutputFPS")} '
        self.command += f'--ratio {int(math.sqrt(int(appData.value("ExpSelecter")[1])))} '
        if appData.value("UseCRF", True):
            self.command += f'--crf {appData.value("CRFSelector")} '
        elif appData.value("UseTargetBitrate", False) and appData.value("BitrateSelector"):
            self.command += f'--bitrate {appData.value("BitrateSelector")}M '
        self.command += f'--scale {appData.value("InterpScaleSelector")} '
        self.command += f'--model {self.fillQuotation(appData.value("SelectedModel"))} '
        if appData.value("StartFrame"):
            self.command += f'--interp-start {appData.value("StartFrame")} '
        if appData.value("StartCntFrame"):
            self.command += f'--interp-cnt {appData.value("StartCntFrame")} '
        if appData.value("StartChunk"):
            self.command += f'--chunk {appData.value("StartChunk")} '
        if appData.value("CropSettings") not in ["", "0"]:
            self.command += f'--crop {appData.value("CropSettings")} '
        if appData.value("ResizeSettings"):
            self.command += f'--resize {appData.value("ResizeSettings")} '
        if appData.value("PresetSelector"):
            self.command += f'--preset {appData.value("PresetSelector").split("[")[0]} '  # fast[...
        if float(appData.value("CloseScedetChecker")):
            self.command += f'--scdet-threshold {appData.value("ScdetSelector")} '

        if appData.value("ReverseChecker", type=bool):
            self.command += f'--reverse '
        if appData.value("CloseScedetChecker", type=bool):
            self.command += f'--no-scdet '
        if appData.value("NoConcatChecker", type=bool):
            self.command += f'--no-concat '
        # if appData.value("HDRChecker", type=bool):
        #     self.command += f'--HDR '
        if appData.value("HwaccelChecker", type=bool):
            self.command += f'--hwaccel '
        if appData.value("QuickExtractChecker", type=bool):
            self.command += f'--quick-extract '
        if appData.value("DupRmChecker", type=bool):
            self.command += f'--remove-dup '
            if appData.value("DupFramesTSelector", type=bool):
                self.command += f'--dup-threshold {appData.value("DupFramesTSelector")} '

        if appData.value("FP16Checker", type=bool):
            self.command += f'--fp16 '
        if appData.value("DebugChecker", type=bool):
            self.command += f'--debug '
        if appData.value("UseNCNNButton", type=bool):
            self.command += f'--ncnn '
        if appData.value("UseCPUButton", type=bool):
            self.command += f'--cpu '
        if appData.value("SaveAudioChecker", type=bool):
            self.command += f'--audio '

        if appData.value("ImgInputChecker", type=bool):
            self.command += f'--img-input '
        if appData.value("ImgOutputChecker", type=bool):
            self.command += f'--img-output '
        if appData.value("OutputOnlyChecker", type=bool):
            self.command += f'--output-only '
        if appData.value("DiscreteCardSelector") not in ["-1", -1]:
            self.command += f'--use-gpu {appData.value("DiscreteCardSelector")} '
        if len(appData.value("EncoderSelector")):
            self.command += f'--encoder {appData.value("EncoderSelector")} '
        if self.concat_only:
            self.command += f'--concat-only '
        self.command = self.command.replace("\\", "/")
        return self.command

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
        all_cnt = len(command_list)
        if all_cnt > 1:
            """MultiTask"""
            appData.setValue("OutputOnlyChecker", True)
        if not all_cnt:
            logger.info("[GUI]: Task List Empty")
            emit_json = {"input": None, "cnt": all_cnt, "current": current_step, "finished": True}
            emit_json = json.dumps(emit_json)
            self.run_signal.emit(emit_json)
            return

        try:
            for f in command_list:
                logger.info(f"[GUI]: Designed Command:\n{f}")
                if appData.value("DebugChecker", type=bool):
                    logger.info(f"DEBUG: {f[1]}")
                    continue
                proc_args = shlex.split(f[1])
                self.current_proc = sp.Popen(args=proc_args, encoding="utf-8", universal_newlines=True)
                self.current_proc.communicate(input=None)
                # os.system(f[1])
                current_step += 1
                emit_json = {"input": f[0], "cnt": all_cnt, "current": current_step, "finished": False}
                emit_json = json.dumps(emit_json)
                self.run_signal.emit(emit_json)
        except Exception:
            logger.error(traceback.format_exc())

        emit_json = {"input": None, "cnt": all_cnt, "current": current_step, "finished": True}
        emit_json = json.dumps(emit_json)
        self.run_signal.emit(emit_json)
        logger.info("[GUI]: Tasks Finished")
        pass

    def kill_proc_exec(self, kill_com):
        if self.current_proc is not None:
            self.current_proc.kill()
            logger.info("Current Process Killed")

    pass


class RIFE_GUI_BACKEND(QDialog, RIFE_GUI.Ui_RIFEDialog):
    kill_proc = pyqtSignal(int)
    notfound = pyqtSignal(int)

    def __init__(self, parent=None):
        super(RIFE_GUI_BACKEND, self).__init__()
        self.setupUi(self)
        self.thread = None
        self.init_before_settings()
        self.Exp = int(math.sqrt(int(appData.value("Exp", "x4")[1])))

        if appData.value("FFmpeg") != "ffmpeg":
            self.ffmpeg = os.path.join(appData.value("FFmpeg"), "ffmpeg.exe")
        else:
            self.ffmpeg = appData.value("FFmpeg")

        if os.path.exists(appDataPath):
            logger.info("[GUI]: Previous Settings, Found, Loaded")

        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)

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

    def select_file(self, filename, folder=False, _filter=None):
        if folder:
            directory = QFileDialog.getExistingDirectory(None, caption="选取文件夹")
            return directory
        directory = QFileDialog.getOpenFileName(None, caption=f"选择{filename}", filter=_filter)
        return directory[0]

    def quick_concat(self):
        input_v = self.ConcatInputV.text()
        input_a = self.ConcatInputA.text()
        output_v = self.OutputConcat.text()
        self.load_current_settings()
        if not input_v or not input_a or not output_v:
            self.sendWarning("Parameters unfilled","请填写输入或输出视频路径！")
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
            self.sendWarning("Parameters unfilled","请填写输入或输出视频路径！")
            return
        palette_path = os.path.join(os.path.dirname(input_v), "palette.png")
        ffmpeg_command = f"""
                    {self.ffmpeg} -hide_banner -i {self.fillQuotation(input_v)} -vf "palettegen=stats_mode=diff" -y {palette_path}
                """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"ffmpeg_command for create palette: {ffmpeg_command}")
        os.system(ffmpeg_command)
        if not appData.value("OutputFPS"):
            appData.setValue("OutputFPS", 48)
            logger.info("Not find output GIF fps, Auto set GIF output fps to 48 as it's smooth enough")
        ffmpeg_command = f"""
                           {self.ffmpeg} -hide_banner -i {self.fillQuotation(input_v)} -i {palette_path} -r {appData.value("OutputFPS")} -lavfi "fps={appData.value("OutputFPS")},scale=960:-1[x];[x][1:v]paletteuse=dither=floyd_steinberg" {self.fillQuotation(output_v)} -y
                        """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"[GUI] create gif: {ffmpeg_command}")
        os.system(ffmpeg_command)
        self.sendWarning("GIF操作完成！",f'GIF帧率:{appData.value("OutputFPS")}', 2)

    def set_start_info(self, sf, scf, sc):
        """

        :return:
        """
        self.StartFrame.setText(str(sf))
        self.StartCntFrame.setText(str(scf))
        self.StartChunk.setText(str(sc))
        return


    def auto_set(self):
        chunk_list = list()
        output_dir = self.OutputFolder.toPlainText()
        now_text = self.InputFileName.currentItem()
        if now_text is None:
            self.sendWarning("Select Item first","请先在Step1点击要补帧的条目以恢复进度")
            return
        input_file = now_text.text()
        if not len(output_dir):
            logger.info("OutputFolder path is empty, pls enter it first")
            self.sendWarning("Parameters unfilled","请移步Stage1填写输出文件夹！")
            return
        if os.path.isfile(output_dir):
            output_dir = os.path.dirname(output_dir)
            self.OutputFolder.setText(output_dir)
            # TODO multi folders support

        if not os.path.exists(output_dir) or not os.path.exists(input_file):
            logger.info("Not Exists OutputFolder")
            self.sendWarning("Output Folder Not Found","输入文件或输出文件夹不存在！请确认输入")
            return

        for f in os.listdir(output_dir):
            if re.match("chunk-[\d+].*?\.mp4", f):
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
            self.set_start_info(0,1,1)
            logger.info("AutoSet Remove and found none")
            return
        last_chunk = chunk_list[-1]
        match_result = re.findall("chunk-(\d+)-(\d+)-(\d+)\.mp4", last_chunk)[0]

        chunk = int(match_result[0])
        first_frame = int(match_result[1])
        last_frame = int(match_result[2])
        first_interp_cnt = (last_frame + 1) * (2 ** self.Exp) + 1
        self.set_start_info(last_frame+1,first_interp_cnt,chunk+1)
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
        model_dir = appData.value("Model")
        if not os.path.exists(model_dir):
            logger.info(f"Not find Module dir at {model_dir}")
            self.sendWarning("Model Dir Not Found","未找到补帧模型路径，请检查！")
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
        self.OutputFolder.setText(appData.value("OutputFolder"))
        self.InputFPS.setText(appData.value("InputFPS", "0"))
        self.OutputFPS.setText(appData.value("OutputFPS"))
        self.CRFSelector.setValue(appData.value("CRFSelector", 16, type=int))
        self.ExpSelecter.setCurrentText(appData.value("ExpSelecter", "x2"))
        self.BitrateSelector.setValue(appData.value("BitrateSelector", 90, type=int))
        self.PresetSelector.setCurrentText(appData.value("PresetSelector", "fast[软编, 硬编]"))
        self.HwaccelChecker.setChecked(appData.value("HwaccelChecker", False, type=bool))
        self.CloseScedetChecker.setChecked(appData.value("CloseScedetChecker", False, type=bool))
        self.ScdetSelector.setValue(appData.value("ScdetSelector", 30, type=int))
        self.DupRmChecker.setChecked(appData.value("DupRmChecker", False, type=bool))
        self.DupFramesTSelector.setValue(appData.value("DupFramesTSelector", 1.00, type=float))
        # self.HDRChecker.setChecked(appData.value("HDRChecker", False, type=bool))
        self.CropSettings.setText(appData.value("CropSettings"))
        self.ResizeSettings.setText(appData.value("ResizeSettings"))

        self.SaveAudioChecker.setChecked(appData.value("SaveAudioChecker", True, type=bool))
        self.QuickExtractChecker.setChecked(appData.value("QuickExtractChecker", True, type=bool))
        self.ImgOutputChecker.setChecked(appData.value("ImgOutputChecker", False, type=bool))
        self.ImgInputChecker.setChecked(appData.value("ImgInputChecker", False, type=bool))
        self.NoConcatChecker.setChecked(appData.value("NoConcatChecker", False, type=bool))
        self.OutputOnlyChecker.setChecked(appData.value("OutputOnlyChecker", False, type=bool))

        self.FP16Checker.setChecked(appData.value("FP16Checker", False, type=bool))
        self.ReverseChecker.setChecked(appData.value("ReverseChecker", False, type=bool))
        self.UseCUDAButton.setChecked(appData.value("UseCUDAButton", False, type=bool))
        self.UseCRF.setChecked(appData.value("UseCRF", True, type=bool))
        self.UseTargetBitrate.setChecked(appData.value("UseTargetBitrate", False, type=bool))
        self.InterpScaleSelector.setCurrentText(appData.value("InterpScaleSelector", "1.00"))
        self.EncoderSelector.setCurrentText(appData.value("EncoderSelector", "HEVC"))

        pos = appData.value("pos", QVariant(QPoint(1920, 1080)))
        size = appData.value("size", QVariant(QSize(400, 400)))
        self.resize(size)
        self.move(pos)

    def load_current_settings(self):
        input_file_names = ""
        for i in self.InputFileName.get_items():
            if len(i):
                input_file_names += f"{i};"
        appData.setValue("InputFileName", input_file_names)
        appData.setValue("OutputFolder", self.OutputFolder.toPlainText())
        appData.setValue("InputFPS", self.InputFPS.text())
        appData.setValue("OutputFPS", self.OutputFPS.text())
        appData.setValue("CRFSelector", self.CRFSelector.value())
        appData.setValue("ExpSelecter", self.ExpSelecter.currentText())
        appData.setValue("BitrateSelector", self.BitrateSelector.value())
        appData.setValue("PresetSelector", self.PresetSelector.currentText())
        appData.setValue("EncoderSelector", self.EncoderSelector.currentText())
        appData.setValue("HwaccelChecker", self.HwaccelChecker.isChecked())
        appData.setValue("CloseScedetChecker", self.CloseScedetChecker.isChecked())
        appData.setValue("ScdetSelector", self.ScdetSelector.value())
        appData.setValue("DupRmChecker", self.DupRmChecker.isChecked())
        appData.setValue("DupFramesTSelector", self.DupFramesTSelector.value())
        # appData.setValue("HDRChecker", self.HDRChecker.isChecked())
        appData.setValue("CropSettings", self.CropSettings.text())
        appData.setValue("ResizeSettings", self.ResizeSettings.text())

        appData.setValue("SaveAudioChecker", self.SaveAudioChecker.isChecked())
        appData.setValue("QuickExtractChecker", self.QuickExtractChecker.isChecked())
        appData.setValue("ImgOutputChecker", self.ImgOutputChecker.isChecked())
        appData.setValue("ImgInputChecker", self.ImgInputChecker.isChecked())
        appData.setValue("NoConcatChecker", self.NoConcatChecker.isChecked())
        appData.setValue("OutputOnlyChecker", self.OutputOnlyChecker.isChecked())
        appData.setValue("FP16Checker", self.FP16Checker.isChecked())
        appData.setValue("ReverseChecker", self.ReverseChecker.isChecked())
        appData.setValue("UseCUDAButton", self.UseCUDAButton.isChecked())
        appData.setValue("UseNCNNButton", self.UseNCNNButton.isChecked())
        appData.setValue("UseCPUButton", self.UseCPUButton.isChecked())
        appData.setValue("UseMultiCUDAButton", self.UseMultiCUDAButton.isChecked())
        appData.setValue("UseCRF", self.UseCRF.isChecked())
        appData.setValue("UseTargetBitrate", self.UseTargetBitrate.isChecked())

        appData.setValue("InterpScaleSelector", self.InterpScaleSelector.currentText())
        appData.setValue("EncoderSelector", self.EncoderSelector.currentText())

        appData.setValue("StartChunk", self.StartChunk.text())
        appData.setValue("StartFrame", self.StartFrame.text())
        appData.setValue("StartCntFrame", self.StartCntFrame.text())

        appData.setValue("SelectedModel", os.path.join(appData.value("Model"), self.ModuleSelector.currentText()))
        appData.setValue("DiscreteCardSelector", self.DiscreteCardSelector.currentIndex())
        appData.setValue("pos", QVariant(self.pos()))
        appData.setValue("size", QVariant(self.size()))

        appData.setValue("DebugChecker", self.DebugChecker.isChecked())

        logger.info("[Main]: Download all settings")
        status_check = "[当前导出设置预览]\n\n"
        for key in appData.allKeys():
            status_check += f"{key} => {appData.value(key)}\n"
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
        """        emit_json = {"input": None, "cnt": len(file_list), "current": current_step, "finished": True}"""
        now_text = self.OptionCheck.toPlainText()
        if data["input"] is not None:
            now_text += f'{datetime.datetime.now()} {data["input"]} 完成\n'
            # logger.info(f"Update RIFE Process: \n {pformat(data)}")
            self.progressBar.setMaximum(int(data["cnt"]))
            self.progressBar.setValue(int(data["current"]))
            self.OptionCheck.setText(now_text)
        if data["finished"]:
            self.sendWarning("补帧任务完成",f"共{data['cnt']}个补帧任务完成",2)
            now_text += "[当前补帧任务结束]"
            self.ProcessStart.setEnabled(True)
            self.ConcatAllButton.setEnabled(True)

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
        try:
            input_stream = cv2.VideoCapture(input_filename)
            input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            self.InputFPS.setText(f"{input_fps:.5f}")
            exp = int(math.sqrt(int(self.ExpSelecter.currentText()[1])))
            output_fps = input_fps * (2**exp)
            self.OutputFPS.setText(f"{output_fps:.5f}")
        except Exception:
            logger.error(traceback.format_exc())
        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
        return

    @pyqtSlot(bool)
    def on_OutputFolder_clicked(self):
        if self.OutputFolder.toPlainText() != "":
            """empty text"""
            return
        output_folder = self.select_file('输出项目文件夹', folder=True)
        self.OutputFolder.setText(output_folder)

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
        # self.auto_set()
        pass

    @pyqtSlot(str)
    def on_ExpSelecter_currentTextChanged(self, currentExp):
        if self.InputFileName.currentItem() is None:
            return
        input_filename = self.InputFileName.currentItem().text().strip(";")
        input_fps = self.InputFPS.text()
        if input_filename == "":
            return
        if input_fps:
            selected_exp = currentExp[1:]
            self.OutputFPS.setText(f"{float(input_fps) * float(selected_exp):.5f}")
            return
        try:
            input_stream = cv2.VideoCapture(input_filename)
            input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            self.InputFPS.setText(f"{input_fps:.5f}")
        except Exception:
            logger.error(traceback.format_exc())

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, tabIndex):
        if tabIndex in [2, 3]:
            """Step 3"""
            if tabIndex == 2:
                self.progressBar.setValue(0)
            logger.info("[Main]: Start Loading Settings")
            self.load_current_settings()
            self.sendWarning("千叮咛万嘱咐","一定要记得把命令行窗口拉长！不然看不到进度！")

        if tabIndex in [1]:
            gpu_info = self.get_gpu_info()
            self.get_model_info()
            if not len(gpu_info):
                self.sendWarning( "No NVIDIA Card Found", "未找到任何N卡，请检查驱动")
                return
            self.DiscreteCardSelector.clear()
            for gpu in gpu_info:
                self.DiscreteCardSelector.addItem(f"{gpu}: {gpu_info[gpu]}")
            pass

    @pyqtSlot(bool)
    def on_ProcessStart_clicked(self):
        self.ProcessStart.setEnabled(False)
        self.progressBar.setValue(0)
        RIFE_thread = RIFE_Run_Thread()
        try:
            self.kill_proc.disconnect()
        except Exception:
            pass
        self.kill_proc.connect(RIFE_thread.kill_proc)
        RIFE_thread.run_signal.connect(self.update_rife_process)
        RIFE_thread.start()
        self.thread = RIFE_thread
        self.OptionCheck.setText("[补帧操作启动，请移步命令行查看进度详情]\n显示“Program finished”则任务完成\n"
                                 "如果遇到任何问题，请将命令行（黑色界面）和软件运行界面的Step1、Step2、Step3截图并联系开发人员解决，"
                                 "群号在首页说明\n\n\n\n\n")

    @pyqtSlot(bool)
    def on_AllInOne_clicked(self):
        """
        Alas
        :return:
        """
        video = self.InputFileName.get_items()
        output_dir = self.OutputFolder.toPlainText()
        if not len(video) or not len(output_dir):
            self.sendWarning("Empty Input", "请先拖入要补帧的文件并确认输出文件夹")
            return
        self.on_ExpSelecter_currentTextChanged(self.ExpSelecter.currentText())
        self.auto_set()
        self.on_tabWidget_currentChanged(1)
        self.on_tabWidget_currentChanged(2)
        self.on_ProcessStart_clicked()


    @pyqtSlot(bool)
    def on_ConcatAllButton_clicked(self):
        """

        :return:
        """
        self.ConcatAllButton.setEnabled(False)
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
        print("try to kill")
        try:
            self.kill_proc.emit(233)
        except Exception:
            logger.error(traceback.format_exc())


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
