FROM python:3.8

RUN apt-get update &&\
    apt-get upgrade &&\
    apt-get install htop 

RUN /usr/local/bin/python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    git clone https://gitee.com/cheakf/robo-master2022_visual.git &&\
    cd robo-master2022_visual &&\
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN cd nms_files &&\
    python setup.py build_ext --inplace &&\
    cd .. &&\
    /bin/bash start.sh
