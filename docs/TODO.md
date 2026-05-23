bash: warning: setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)


$ ssh stt-server export PATH="$HOME/.local/bin:$PATH" && cd ~/stt-remote && uv pip list
Package                Version
---------------------- ---------
annotated-doc          0.0.4
annotated-types        0.7.0
certifi                2026.5.20
charset-normalizer     3.4.7
click                  8.4.1
cuda-bindings          13.2.0
cuda-pathfinder        1.5.4
cuda-toolkit           13.0.2
filelock               3.29.0
fsspec                 2026.4.0
idna                   3.16
jinja2                 3.1.6
llvmlite               0.47.0
markdown-it-py         4.2.0
markupsafe             3.0.3
mdurl                  0.1.2
more-itertools         11.1.0
mpmath                 1.3.0
networkx               3.6.1
numba                  0.65.1
numpy                  2.4.6
nvidia-cublas          13.1.1.3
nvidia-cuda-cupti      13.0.85
nvidia-cuda-nvrtc      13.0.88
nvidia-cuda-runtime    13.0.96
nvidia-cudnn-cu13      9.20.0.48
nvidia-cufft           12.0.0.61
nvidia-cufile          1.15.1.6
nvidia-curand          10.4.0.35
nvidia-cusolver        12.0.4.66
nvidia-cusparse        12.6.3.3
nvidia-cusparselt-cu13 0.8.1
nvidia-nccl-cu13       2.29.7
nvidia-nvjitlink       13.0.88
nvidia-nvshmem-cu13    3.4.5
nvidia-nvtx            13.0.85
openai-whisper         20250625
pydantic               2.13.4
pydantic-core          2.46.4
pydantic-settings      2.14.1
pygments               2.20.0
python-dotenv          1.2.2
regex                  2026.5.9
requests               2.34.2
rich                   15.0.0
ruff                   0.15.14
setuptools             81.0.0
shellingham            1.5.4
sympy                  1.14.0
tiktoken               0.13.0
torch                  2.12.0
torchaudio             2.11.0
tqdm                   4.67.3
triton                 3.7.0
typer                  0.25.1
typing-extensions      4.15.0
typing-inspection      0.4.2
urllib3                2.7.0




需要 apt update -y && apt install ffmpeg -y

> nvidia-cuda-toolkit 这个好像会版本对不上

TLDR
- 选了gpu型以后，页面会出现一个提示，让你选对应的镜像，不要选普通镜像，不然到时候 驱动安装会有问题，要选对应的预装镜像
  - 像我选的 A10/6 这种提示grid镜像 选择即可

阿里的面板做得好怪啊，找不到预装，只有先找系统 https://market.aliyun.com/detail/cmjj00066847.html，找有cuda预装的系统
- 查价页面 https://www.aliyun.com/price/detail 也好垃圾，不能排序
- ai云助理也是傻逼，回答不了
- 再搜 sgn7i（表示共享gpu? 在它gpu选择里看到的)
  - vgn ， sgn. vgn = CPU独享，sgn = CPU共享，两者GPU均基于vGPU切分且显存独享。
  - 6i是T4, 7i是A10
- 还是不行，这种 grid的似乎只能装他的市场的名字里有grid的系统，有个ubuntu 22.04 grid, 进去反正有 nvidia-smi就对了

识别失败，不要拷贝srt

先nvidia-smi 看右上角cuda版本，比如11.4 然后去https://pytorch.org/get-started/previous-versions/ 找11相关的，比如这个118表示11.8
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



- export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

- NVIDIA A10*1/6
  - 正在加载模型 'small'...
  - 模型加载到设备：cuda
  - 计算设备：GPU: NVIDIA A10-4Q
  - 已处理：音频 56m55s，耗时 10m8s