FROM tensorflow/tensorflow:1.10.1-gpu-py3

RUN alias lst="ls -lrth"; alias nv="watch -n 1 nvidia-smi"; alias python="python3"
RUN apt-get update -y
RUN apt-get install -y htop curl vim python3-tk git

RUN pip install --upgrade pip
RUN pip install torch==0.4.1 torchvision setproctitle line_profiler setGPU waitGPU psutil robustml
RUN pip install seaborn tensorpack
RUN pip install -e git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans

RUN cd /
ENTRYPOINT bash
