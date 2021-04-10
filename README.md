# SVFI 2 (Original based on RIFE GUI)
A GUI for RIFE interpolation of long-time movies or video materials

## How to use SVFI 使用简明教程
built by PyQt5, easy to launch. Support CPU, NVIDIA(CUDA) and AMD([NCNN](https://github.com/Tencent/ncnn) based on [RIFE-NCNN](https://github.com/nihui/rife-ncnn-vulkan))
- Extract release package to an empty folder, ASCII path only 将release压缩包解压到仅有纯英文、无空格路径的文件夹

![Step1](./static/step1.png)
- launch SVFI 双击启动SVFI（你也可以自己在Package文件夹里找到SVFI.2.0.1.exe，双击启动）

![Step2](./static/step2.png)
- Fill in related parameters, follow the step from 1 to 2 to 3 输入相关参数完成准备工作

![Step3](./static/step3.png)

![Step4](./static/step4.png)
- Press One-Line-Shot button and wait for output, which is easy 按“开始补帧”按钮，等待成品输出。如果遇到输出信息堆叠，请横向拉伸窗口

![Step5](./static/step5.png)

## CLI tool  命令行工具
If you anticipate some more customizable outputs, please use CLI tool(one line shot args.exe) to generate your own footage
### 1. Command Shortcuts: 命令帮助 
```
python3 one_line_shot_args.py -h
```
*For Latest Update: 最新公测版本 2.0.1：*
```
usage: #### RIFE CLI tool/补帧分步设置命令行工具 by Jeanna #### [-h] -i INPUT -o OUTPUT
                                                     -c CONFIG [--concat-only]

Interpolation for sequences of images

optional arguments:
  -h, --help            show this help message and exit

Basic Settings, Necessary:
  -i INPUT, --input INPUT
                        原视频/图片序列文件夹路径
  -o OUTPUT, --output OUTPUT
                        成品输出的路径，注意默认在项目文件夹
  -c CONFIG, --config CONFIG
                        配置文件路径
  --concat-only         只执行合并已有区块操作

```
The Config File should be like this:
```buildoutcfg
[General]
OneLineShotPath=...Your One Line Shot Args.exe Path
ffmpeg=...Your FFmpeg.exe Path
model=...Your Model(train_log) Path organized in single folders
img_input=false  # is input image sequence
ncnn=false  # use ncnn for AMD support
InputFileName=  # irrevelant
output=...Your output folder
fps=0  # your input fps
target_fps=  # your output fps
crf=16  # constant rate factor
exp=1  # interpolation
bitrate=90  # 90M
preset=slow  # ffmpeg preset
encoder=H264/AVC
hwaccel=false  # Use Hardware Acceleration in rendering
no_scdet=false  # no scene detection
scdet_threshold=12
any_fps=false  # output in any fps rate 
remove_dup=false  # remove duplicated frames?
dup_threshold=1
crop=  # ffmpeg crop parameters, 0:9 means a black par with 9 pixels on each side of heights of the input video
resize=  # ffmpeg resize parameters
save_audio=true  # output with audio
quick_extract=true  # quick method applied to extract frames
img_output=false  # is output img sequence
no_concat=false  # don't concat interpolated chunks
output_only=true  # leave no traces after interpolation
fp16=false  # use fp16 for low VRAM
reverse=false  # use reversed interpolation
UseCRF=true
UseTargetBitrate=false
scale=1.00  # interpolation precision rate
pix_fmt=yuv420p
output_ext=mp4
chunk=1
interp_start=0
render_gap=1000  # frames cnt of each output chunk
SelectedModel=...Your selected model, e.g. ...\\train_log\\RIFE2.3
use_specific_gpu=0
pos=@Point(3 33)
size=@Size(1174 833)
ffmpeg_customized=  # your customed ffmpeg parameters while rendering output, "-crf 9 -qp ..."
debug=false
j_settings=2:4:4  # NCNN RIFE interpolation parameters
slow_motion=false  # output in slow motion in target?
```
### 2. Example 操作样例
```
one_line_shot_args.exe -i <input_video> --output <output_video> -c <Your Config Path>
```

### 3. Output
- output `chunk(.mp4)` is named after following rules
- 输出的视频块数按照以下规则命名
```
chunk-<chunk count>-<start frame count>-<end frame count>.mp4
```
e.g. chunk-001-00000001-00001466.mp4
## Other tools in this repository
measure.py, handy scripts for measuring two images(for inferenced imgs)
## Reference & Acknowledgement
- Video interpolation method(for CUDA): [RIFE](https://github.com/hzwer/arXiv2020-RIFE)
- [RIFE-NCNN](https://github.com/nihui/rife-ncnn-vulkan) 
- This project is already merged into [Squirrel-Video-Frame-Interpolation](https://github.com/YiWeiHuang-stack/Squirrel-RIFE)
## Patron for fun, thanks!!
![Help](./static/help.png)