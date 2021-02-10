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
usage:
#### RIFE 补帧分步设置命令行 from Jeanna ####   [-h] -i INPUT -o OUTPUT --rife
                                               RIFE --ffmpeg FFMPEG --fps FPS
                                               -r {1,2,3} --start START
                                               --chunk CHUNK [--end END]
                                               [--round ROUND]
                                               [--interp_start INTERP_START]
                                               [--interp_cnt INTERP_CNT]
                                               [--hwaccel] [--model MODEL]
                                               [--UHD] [--debug] [--pause]
                                               [--rifeonly] [--renderonly]
                                               [--accurate] [--reverse]
                                               [-s SCENE] [--UHDcrop UHDCROP]
                                               [--HDcrop HDCROP] [-b BITRATE]
                                               [--preset PRESET] [--crf CRF]

Interpolation for sequences of images => 基础设置：以下各项必须填写

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        原视频路径，记得打双引号。补帧项目将在视频所在文件夹建立
  -o OUTPUT, --output OUTPUT
                        成品输出的路径，注意默认在项目文件夹
  --rife RIFE           inference_img_only.py的路径
  --ffmpeg FFMPEG       ffmpeg.exe 所在路径
  --fps FPS             原视频的帧率, 请使用分数，23.976 = 24000/1001 => 补帧分步设置
  -r {1,2,3}, --ratio {1,2,3}
                        补帧系数, 2的几次方，23.976->96，填2
  --start START         起始补帧的原视频帧数
  --chunk CHUNK         新增视频的序号
  --end END             结束补帧的原视频帧数
  --round ROUND         要处理的转场数(一个转场将素材补成一个视频)
  --interp_start INTERP_START
                        起始补帧的帧序列帧数，默认：1
  --interp_cnt INTERP_CNT
                        补帧帧序列起始帧数
  --hwaccel             支持硬件加速编码（想搞快点就可以用这个） => 个性化分步设置：
  --model MODEL         Select RIFE Model, default v2
  --UHD                 支持UHD补帧
  --debug               debug
  --pause               pause, 删除项目缓存前提供确认操作
  --rifeonly            只进行补帧操作，其余手动，要求文件夹目录齐全
  --renderonly          只进行渲染操作，其余手动，要求文件夹目录齐全
  --accurate            精确补帧
  --reverse             反向光流 => 个性化画质/成品质量控制：
  -s SCENE, --scene SCENE
                        转场识别灵敏度，越小越准确，人工介入也会越多
  --UHDcrop UHDCROP     UHD裁切参数，默认开启，填0不裁，默认：3840:1608:0:276
  --HDcrop HDCROP       QHD裁切参数，默认：1920:804:0:138
  -b BITRATE, --bitrate BITRATE
                        成品目标（最高）码率
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
