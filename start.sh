#!/bin/bash
#初始化虚拟环境，激活openvino
source /home/cheakf/openvino_venv/bin/activate
cd /home/cheakf/RoboMaster2022_visual_program
#循环挂起程序，保证存活
while true
do
    python3 main.py 
done