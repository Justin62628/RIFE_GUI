# RIFE_GUI
Python Scripts (CLI tool with GUI) for RIFE batch process (Mainly for interpolation of long movies or video materials with cuts)

## *Usage*:
### *Preparation*
#### 1. Clone this Repository
#### 2. Clone [RIFE](https://github.com/hzwer/arXiv2020-RIFE)
#### 3. Follow RIFE's instructions and download assigned datasets
## *Run*
### 1. CLI tool
#### 1. Run command to obtain CLI guidance: 
```
python3 one_line_shot_args.py -h
```
*For Latest Update:*
```
usage: #### RIFE Step by Step CLI tool/补帧分步设置命令行工具 from Jeanna ####
       [-h] -i INPUT -o OUTPUT [--rife RIFE] [--ffmpeg FFMPEG] [--fps FPS]
       [--target-fps TARGET_FPS] -r {1,2,3} [--chunk CHUNK]
       [--interp-start INTERP_START] [--interp-cnt INTERP_CNT] [--hwaccel]
       [--model MODEL] [--UHD] [--debug] [--pause] [--quick-extract]
       [--extract-only] [--rife-only] [--render-only] [--accurate] [--reverse]
       [--scdet SCDET] [--scdet-threshold SCDET_THRESHOLD]
       [--UHD-crop UHDCROP] [--HD-crop HDCROP] [--resize RESIZE] [-b BITRATE]
       [--preset PRESET] [--crf CRF]

Interpolation for sequences of images

optional arguments:
  -h, --help            show this help message and exit

Basic Settings, Necessary:
  -i INPUT, --input INPUT
                        原视频路径, 补帧项目将在视频所在文件夹建立
  -o OUTPUT, --output OUTPUT
                        成品输出的路径，注意默认在项目文件夹
  --rife RIFE           inference_img_only.py的路径
  --ffmpeg FFMPEG       ffmpeg三件套所在文件夹
  --fps FPS             原视频的帧率, 默认0(自动识别)
  --target-fps TARGET_FPS
                        目标视频帧率, 默认0(fps * 2 ** exp)

Step by Step Settings:
  -r {1,2,3}, --ratio {1,2,3}
                        补帧系数, 2的几次方，23.976->95.904，填2
  --chunk CHUNK         新增视频的序号(auto)
  --interp-start INTERP_START
                        用于补帧的原视频的帧序列起始帧数，默认：0
  --interp-cnt INTERP_CNT
                        成品帧序列起始帧数
  --hwaccel             支持硬件加速编码(想搞快点就用上)
  --model MODEL         Select RIFE Model, default v2

Preference Settings:
  --UHD                 支持UHD补帧
  --debug               debug
  --pause               pause, 在各步暂停确认
  --quick-extract       快速抽帧
  --extract-only        只进行帧序列提取操作
  --rife-only           只进行补帧操作
  --render-only         只进行渲染操作
  --accurate            精确补帧
  --reverse             反向光流

Output Settings:
  --scdet SCDET         转场识别灵敏度，越小越准确，人工介入也会越多
  --scdet-threshold SCDET_THRESHOLD
                        转场间隔阈值判定，要求相邻转场间隔大于该阈值
  --UHD-crop UHDCROP    UHD裁切参数，默认开启，填0不裁，默认：3840:1608:0:276
  --HD-crop HDCROP      QHD裁切参数，默认：1920:804:0:138
  --resize RESIZE       ffmpeg -s 缩放参数，默认不开启（为空）
  -b BITRATE, --bitrate BITRATE
                        成品目标(最高)码率
  --preset PRESET       压制预设，medium以下可用于收藏。硬件加速推荐hq
  --crf CRF             恒定质量控制，12以下可作为收藏，16能看，默认：9
```
#### 2. Example
```
python3 one_line_shot_args.py -i <input_video> --rife <path of inference_img_only.py> \-r 2 --output <output_video>  --fps 24000/1001 --preset hq --hwaccel --crf 6 --UHD  --start 1 --chunk 1 --ffmpeg ffmpeg
```
*which means follow operation:*
1. Input a 23.976 fps video with UHD contents (formats specifically assigned to **-color_range tv -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc**)
2. Interpolate the footage 4 times with exp(ratio)=2 from 23.976 fps to 96 fps(framerates of output chunks are locked to integral for stable PERFORMANCE)
3. Assign encode presets of ffmpeg as "hq"(for encoder **hevc_nvenc**, note that HDR metadata will not be written in this case)
4. Assign **crf** for ffmpeg as 6
5. start the interpolation process from **frame 1, chunk 1** to the end
#### 3. Output
*output chunk(.mp4) is named after following rules*
```
chunk-<chunk count>-<start frame count>-<end frame count>.mp4
```
e.g. chunk-001-00000001-00001466.mp4
### 2. GUI tool
built by PyQt5, easy to launch
### 3. Other tools
measure.py, handy scripts for measuring two images(for inferenced imgs)
### 4. Notes:
English usages on fly
### 5. Reference & Acknowledgement
Video interpolation method: [RIFE](https://github.com/hzwer/arXiv2020-RIFE)

RIFE GUI Assembly: [Squirrel-Video-Frame-Interpolation](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation/stargazers)
