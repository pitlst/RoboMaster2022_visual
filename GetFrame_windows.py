import cv2
import json

class GetFrame:
    def __init__(self, mode):
        self.mode = mode
        self.read_json()
        self.StartCamera()
        self.SetCamera()

    def restart_camera(self, mode):
        #重置摄像头，并更新参数
        self.mode = mode
        self.EndCamera()
        self.read_json()
        self.StartCamera()
        self.SetCamera()
    
    def read_json(self):
        with open('./debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.video_debug_set = load_dict["Debug"]["video_debug_set"]
            self.video_debug_path = load_dict["Debug"]["video_debug_path"]
    

    def StartCamera(self):
        if self.video_debug_set:
            #从视频获取信息时需要初始化虚拟opencv相机
            self.cap = cv2.VideoCapture(self.video_debug_path)

    def SetCamera(self):
        if self.video_debug_set:
            while 1:
                ret,_ = self.cap.read()
                if ret == 1:
                    break 


    def GetOneFrame(self):
        #获取单帧图像，根据debug参数来确定图像获取来源
        if self.video_debug_set:
            _ , frame = self.cap.read()
            return frame

    def EndCamera(self):
        if self.video_debug_set:
            self.cap.release()