'''

               _____                       _____                       ____
              /\    \                     /\    \                     /\    \ 
             /::\    \                   /::\    \                   /::\    \  
             \:::\    \                 /::::\    \                 /::::\    \ 
              \:::\    \               /::::::\    \               /::::::\    \ 
               \:::\    \             /:::/\:::\    \             /:::/\:::\    \ 
                \:::\    \           /:::/  \:::\    \           /:::/__\:::\    \ 
                /::::\    \         /:::/    \:::\    \         /::::\   \:::\    \  
               /::::::\    \       /:::/      \:::\    \       /::::::\   \:::\    \ 
              /:::/\:::\    \     /:::/        \:::\    \     /:::/\:::\   \:::\    \ 
             /:::/  \:::\____\   /:::/          \:::\____\   /:::/__\:::\   \:::\____\ 
            /:::/   /\::/    /   \:::\          /:::/    /   \:::\   \:::\   \::/    /
           /:::/   /  \/____/     \:::\        /:::/    /     \:::\   \:::\   \/____/
          /:::/   /                \:::\      /:::/    /       \:::\   \:::\    \     
         /:::/   /                  \:::\    /:::/    /         \:::\   \:::\____\   
        /:::/   /                    \:::\  /:::/    /           \:::\   \::/    /   
       /:::/   /                      \:::\/:::/    /             \:::\   \/____/     
      /:::/   /                        \::::::/    /               \:::\    \         
     /:::/   /                          \::::/    /                 \:::\____\       
     \::/   /                            \::/    /                   \::/    /       
      \/___/                              \/____/                     \/____/ 

TOE实验室算法组---打符/自瞄程序
@作者：苏盲，cheakf
'''

import threading
import cv2
import time
import json
import numpy as np
import queue
import argparse
import sys
import time
import os
from Communication import MySerial
from Aimbot import GetArmor
from EnergyMac import GetEnergyMac
from logger import log
#根据平台调用不同的图像获取类，注意，win平台无法调取海康相机
if sys.platform.startswith('win'):
    from GetFrame_windows import GetFrame
else:
    from GetFrame import GetFrame



class Main:
    def __init__(self,command_input):
        #获取和处理debug参数
        debug_list = self.__get_debug(command_input)
        #串口初始化
        self.MySerial_class = MySerial(debug_list[0])
        log.print_info('serial init done')
        #多线程中跨线程调用资源使用资源锁，这里为初始化资源锁，缓冲区为10次，一般不会出现溢出的问题
        #如果出现溢出问题首先排查其他程序执行耗时问题，最后增大缓冲区
        self.frame = queue.Queue(10)
        self.reset_label = queue.Queue(5)
        self.break_all = queue.Queue(1)
        #获取第一次串口输入，方便确定红蓝方和模式信息
        temp = self.MySerial_class.get_msg_first()
        self.color_init = temp[0]
        self.mode_init = temp[1]
        log.print_info('mode '+str(self.mode_init))
        log.print_info('color '+str(self.color_init))
        #初始化图像获取类，主要是对海康威视相机的操作,需要数据来源和模式信息
        self.GetFrame_class = GetFrame(debug_list[1],self.mode_init)
        log.print_info('hikvision init done')
        #初始化传统视觉自瞄类，需要debug模式标志位，数据输入标志位，颜色和模式信息
        self.GetArmor_class = GetArmor(self.debug,debug_list[4],self.color_init,self.mode_init)
        log.print_info('aimbot init done')
        #初始化深度学习打符类，需要颜色信息
        self.GetEnergyMac_class = GetEnergyMac(self.debug,self.video_debug_set,self.color_init)
        log.print_info('energy init done')
        #根据标志位开始录像
        if debug_list[4]:
            self.video_debug = True
            if self.mode_init == 0 or self.mode_init == 3:
                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 10,(debug_list[5][0],debug_list[5][1]))
            elif self.mode_init == 1 or self.mode_init == 2 or self.mode_init == 4 or self.mode_init == 5:
                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 8,(debug_list[5][2],debug_list[5][3]))
            log.print_info('video writer init done')
        else:
            self.video_debug = False
        log.print_info('all module init done')
 

    def grab_image(self):
        #图像获取线程
        t0 = time.time()
        t1 = time.time()
        fps = 0
        new = [-1,-1]
        old = [self.color_init,self.mode_init]
        temp_num = 0
        while 1:
            #按键退出
            if self.break_all.full():
                break
            #每隔一秒输出一次帧率
            if time.time()-t0 > 1:
                log.print_info('fps:',fps)
                fps = 0
                t0 = time.time()
            #获取单帧图像
            frame = self.GetFrame_class.GetOneFrame()
            self.frame.put((frame,time.time()))
            fps += 1
            if self.video_debug:
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
                temp_num += 1
                if temp_num > 20:
                    temp_num = 0
                    self.video_writer.write(frame)
            #在最开始的时候获取颜色信息重启摄像头，在之后用于检查模式信息
            if time.time()-t1 > 0.05:
                upper_computer_data = list(self.MySerial_class.get_msg())
                if upper_computer_data[0] != -1 and upper_computer_data[1] != -1:
                    new = upper_computer_data
                    log.print_info(new)
                    #模式与颜色原先不同时进行重置,并通过消息队列通知其他线程
                    if new != old:
                        cv2.destroyAllWindows()
                        if new[1] == 0 or new[1] == 3:
                            self.GetArmor_class = GetArmor(new[0],new[1])
                        if old[0] == new[0] and (((old[1] == 0 or old[1] == 3) and (new[1] == 0 or new[1] == 3)) or ((old[1] == 1 or old[1] == 2 or old[1] == 4 or old[1] == 5) and (new[1] == 1 or new[1] == 2 or new[1] == 4 or new[1] == 5))):
                            pass
                        else:
                            if (new[1] == 0 or new[1] == 3) and self.label_list[4]:
                                self.video_writer.release()
                                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 10,(1440,864))
                            elif (new[1] == 1 or new[1] == 2 or new[1] == 4 or new[1] == 5) and self.label_list[4]:
                                self.video_writer.release()
                                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 8,(768,768))
                            self.GetFrame_class.restart_camera(new[1])
                        self.reset_label.put(new)
                        old = new
                        log.print_info('reset all done')
                t1 = time.time()


    def post_process(self):
        #图像处理线程
        t0 = time.time()
        old = [self.color_init,self.mode_init]
        msg_temp = [-1,-1,-1]
        interpolation_label = 0
        new = -1
        if self.label_list[0]:
            cv2.namedWindow('colorTest')
            cv2.namedWindow('LBTest')
            cv2.namedWindow('armorTest')
            self.GetArmor_class.TrackerBar_create()
        if (self.label_list[0] and (self.mode_init == 0 or self.mode_init == 3) ) or (self.label_list[3] and (self.mode_init == 1 or self.mode_init == 2 or self.mode_init == 4 or self.mode_init == 5)):
            time_sleep = self.label_list[1]
        else:
            time_sleep = -1
        while 1:
            #每秒10次更新mode参数
            if time.time()-t0 > 0.01 or old[1] == -1:
                try:
                    new = self.reset_label.get_nowait()
                except:
                    new = old
                if (old[0] != new[0] or old[1] != new[1])and (new != [-1,-1]):
                    #如果有debug，重新创建窗口
                    if old[0] == new[0] and ((old[1] == 1 and new[1] == 2) or (old[1] == 2 and new[1] == 1) or (old[1] == 4 and new[1] == 5) or (old[1] == 5 and new[1] == 4)):
                        pass
                    else:
                        if self.label_list[0]:
                            cv2.namedWindow('colorTest')
                            cv2.namedWindow('LBTest')
                            cv2.namedWindow('armorTest')
                            self.GetArmor_class.TrackerBar_create()
                            if (self.label_list[0] and (self.mode_init == 0 or self.mode_init == 3)) or (self.label_list[3] and (self.mode_init == 1 or self.mode_init == 2 or self.mode_init == 4 or self.mode_init == 5)):
                                time_sleep = self.label_list[1]
                    old = new
                t0 = time.time()
            #把一帧图像从缓冲区取出
            frame, f_time = self.frame.get()
            #根据模式信息来却动图像处理过程，确定是打符还是自瞄
            if type(frame) is np.ndarray:
                interpolation_label = 0
                if old[1] == 0 or old[1] == 3:
                    interpolation_label = 1
                    msg_temp = list(self.GetArmor_class.GetArmor(frame))
                elif old[1] == 1 or old[1] == 2 or old[1] == 4 or old[1] == 5:
                    interpolation_label = 2
                    msg_temp = list(self.GetEnergyMac_class.GetHitPointDL(frame,f_time,old[1]))
            #将处理得到的信息发送
            if interpolation_label == 1 or interpolation_label == 2:
                self.MySerial_class.send_message(msg_temp)
            #根据debug参数来确定是否延时
            if time_sleep != -1:
                k = cv2.waitKey(time_sleep)
                if k == 27:
                    cv2.destroyAllWindows()
                    #一定要注意摄像头关闭取流
                    self.GetFrame_class.EndCamera()
                    if self.label_list[4]:
                        self.video_writer.release()
                    #向其他线程广播消息
                    self.break_all.put(0)
                    log.print_info('return')
                    break


    def __get_debug(self,command_input):
        #处理命令行输入
        serial_debug = command_input.serial
        self.debug = int(command_input.debug)
        source_path = command_input.source
        #从json获取debug参数
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            timeout = load_dict["Debug"]["timeout"]
            kalmanfilter_enable = load_dict["Debug"]["kalmanfilter_enable"]
            video_writer_debug = load_dict["Debug"]["video_writer"]
            if video_writer_debug:
                with open('./json/common.json','r',encoding = 'utf-8') as load_f:
                    load_dict = json.load(load_f,strict=False)
                    Aimbot_width = load_dict["Aimbot"]["width"]
                    Aimbot_height = load_dict["Aimbot"]["height"]
                    Energy_mac_width = load_dict["Energy_mac"]["width"]
                    Energy_mac_height = load_dict["Energy_mac"]["height"]
                    video_writer_list = [Aimbot_width,Aimbot_height,Energy_mac_width,Energy_mac_height]
            else:
                video_writer_list = []
        #判断数据来源
        if source_path == 'HIVISION':
            self.video_debug_set = 0
        elif source_path.isdigit():
            self.video_debug_set = 1
        elif os.path.isfile(source_path) and (source_path[-4:] == '.avi' or source_path[-3:] == '.mp4'):
            self.video_debug_set = 1
        elif os.path.isdir(source_path):
            self.video_debug_set = 2
        else:
            self.video_debug_set = 0

        return serial_debug, source_path, timeout, kalmanfilter_enable, video_writer_debug, video_writer_list


#程序从这里开始
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TOE实验室视觉步兵程序调试帮助')
    parser.add_argument('--debug','-d', action='store_true', help='是否开启滑动条调参/debug模式,不输入默认不开启')
    parser.add_argument('--serial','-c', action='store_true', help='是否开启串口,不输入默认开启，输入后串口传入值固定从debug.json读取')
    parser.add_argument('--source','-s', type=str, default='HIVISION', help='输入识别数据的来源，可以是视频的路径或者图片文件夹的路径，不输入该选项默认海康摄像头，输入该选项默认usb摄像头')
    input = parser.parse_args()
    #日志系统初始化，必须在最前
    log.set_level(input.debug)
    # 程序初始化
    infantry = Main(input)
    grab_image_thread = threading.Thread(target=infantry.grab_image)
    post_process_thread = threading.Thread(target=infantry.post_process)
    #启动线程
    post_process_thread.start()
    grab_image_thread.start()