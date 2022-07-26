#!/bin/bash
#初始化虚拟环境，激活openvino
source /home/cheakf/openvino_venv/bin/activate
cd /home/cheakf/RoboMaster2022_visual_program
#编译生成nms模块
cd nms_files
python3 setup.py build_ext --inplace
cd ..
#循环挂起程序，保证存活
while true
do
    python3 main.py 
    #在程序非正常死亡后保证程序完全杀死
    kill -9 $(ps x | grep "main.py" | grep 'python' | grep -v grep | awk '{print $1}')
done
