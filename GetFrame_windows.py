import sys
import numpy as np
import json
import cv2
import os
from utils import log



class GetFrame:
    def __init__(self, source_path, mode):
        self.mode = mode
        if source_path == 'HIVISION':
            log.print_info('source: HIVISION')
            self.video_debug_set = 0
        elif source_path.isdigit():
            log.print_info('source: USB carmera')
            self.video_debug_set = 1
            self.video_debug_path = int(source_path)
        elif os.path.isfile(source_path) and (source_path[-4:] == '.avi' or source_path[-3:] == '.mp4'):
            log.print_info('source: video')
            self.video_debug_set = 1
            self.video_debug_path = source_path
        elif os.path.isdir(source_path):
            log.print_info('source: photo')
            self.video_debug_set = 2
            self.count = 0
        else:
            log.print_error('unknow source')
            self.video_debug_set = 0
        self.StartCamera()
        self.SetCamera()

    def restart_camera(self, mode):
        #重置摄像头，并更新参数
        self.mode = mode
        self.EndCamera()
        self.StartCamera()
        self.SetCamera()

    def StartCamera(self):
        if self.video_debug_set == 1:
            #从视频获取信息时需要初始化虚拟opencv相机
            self.cap = cv2.VideoCapture(self.video_debug_path)
        elif self.video_debug_set == 2:
            #从路径获取图片时需要获取图片文件列表
            self.files = os.listdir(self.video_debug_path)

    def SetCamera(self):
        if self.video_debug_set == 1:
            #尝试读取一次相机，保证相机正常运行
            while 1:
                ret,_ = self.cap.read()
                if ret == 1:
                    break 
                else:
                    log.print_error("open VideoCapture fail! ret[%x]" % ret)
                    sys.exit()
        elif self.video_debug_set == 2:
            self.file_list = []
            for file in self.files:
                if file.endswith('jpg'):
                    self.file_list.append(file)
            if len(self.file_list) == 0:
                log.print_error("There are no pictures under the path")
                sys.exit()


    def GetOneFrame(self):
        #获取单帧图像，根据debug参数来确定图像获取来源
        if self.video_debug_set == 1:
            _ , frame = self.cap.read()
            return frame
        elif self.video_debug_set == 2:
            file = self.file_list[self.count]
            frame = cv2.imread(self.video_debug_path+'/'+file)
            self.count += 1
            return frame

    def EndCamera(self):
        if self.video_debug_set == 1:
            self.cap.release()
        elif self.video_debug_set == 2:
            self.count = None