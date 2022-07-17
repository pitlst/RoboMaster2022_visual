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
import signal
import ctypes
import traceback
from Communication import MySerial
from Aimbot import GetArmor
from logger import log, video_writer
#根据平台调用不同的图像获取类，注意，win平台无法调取海康相机
if sys.platform.startswith('win'):
    log.print_info('run computer is wim and can not open HIVISION')
    from GetFrame_windows import GetFrame
else:
    log.print_info('run computer is linux')
    from GetFrame import GetFrame
#该变量为打符是否初始化的标志位
energy_label = True
#根据环境选择打符是否开启
try:
    from EnergyMac import GetEnergyMac
    from Eenergy_predicted import AnglePredicted
except ImportError:
    log.print_info('can not find request module, energy find can not init')
    energy_label = False
except Exception as e: 
    traceback.print_exc()
    raise ValueError("成功退出程序 导入包错误")



#该变量为退出线程的标志位
break_label = False
#该变量为重置模式的标志位
reset_label = None

class Main:
    def __init__(self,command_input):
        #获取和处理debug参数
        self.__get_debug(command_input)
        #串口初始化
        self.MySerial_class = MySerial(self.serial_debug)
        log.print_info('serial init done')
        #多线程中跨线程调用资源使用资源锁，这里为初始化资源锁，缓冲区为10次，一般不会出现溢出的问题
        #如果出现溢出问题首先排查其他程序执行耗时问题，最后增大缓冲区
        self.frame = queue.Queue(10)
        #获取第一次串口输入，方便确定红蓝方和模式信息
        temp = self.MySerial_class.get_msg_first()
        self.color_init = temp[0]
        self.mode_init = temp[1]
        log.print_info('mode '+str(self.mode_init))
        log.print_info('color '+str(self.color_init))
        #初始化图像获取类，主要是对海康威视相机的操作,需要数据来源和模式信息
        self.GetFrame_class = GetFrame(self.source_path,self.mode_init)
        log.print_info('hikvision init done')
        #初始化传统视觉自瞄类，需要debug模式标志位，数据输入标志位，颜色和模式信息
        self.GetArmor_class = GetArmor(self.debug,self.video_debug_set,self.color_init,self.mode_init)
        log.print_info('aimbot init done')
        if energy_label:
            #初始化深度学习识别类，需要debug模式标志位，数据输入标志位,颜色信息
            self.GetEnergyMac_class = GetEnergyMac(self.debug,self.video_debug_set,self.color_init)
            #初始化深度学习预测类，需要模式信息
            self.AnglePredicted_class = AnglePredicted(self.mode_init)
            log.print_info('energy init done')
        #根据标志位开始录像
        if self.video_writer_debug:
            video_writer.set_mode(self.mode_init)
            log.print_info('video writer init done')
        log.print_info('all module init done')
 

    def grab_image(self):
        #图像获取线程
        global reset_label
        t0 = time.time()
        t1 = time.time()
        fps = 0
        new = [-1,-1]
        old = [self.color_init,self.mode_init]
        reset_label = self.mode_init
        temp_num = 0
        temp_num_max = int(self.video_interval_fps)
        while 1:
            #获取单帧图像
            frame = self.GetFrame_class.GetOneFrame()
            fram_time= time.time()
            self.frame.put((frame,fram_time))
            fps += 1
            #录像
            if self.video_writer_debug:
                temp_num += 1
                if temp_num > temp_num_max:
                    temp_num = 0
                    frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
                    video_writer.write(frame,fram_time)
            #根据帧率判断线程是否处于正常状态，不正常引发报错退出
            if fps > 1000:
                raise ValueError("error：图像获取线程帧数异常")
            #每隔一秒输出一次帧率
            if time.time()-t0 > 1:
                log.print_info('fps:'+str(fps))
                fps = 0
                t0 = time.time()
            #在最开始的时候获取颜色信息重启摄像头，在之后用于检查模式信息
            if time.time()-t1 > 0.05:
                t1 = time.time()
                new = list(self.MySerial_class.get_msg())
                #模式与颜色原先不同时进行重置,并通过全局变量通知其他线程
                if -1 not in new and new != old:
                    #如果切换颜色
                    if new[0] != old[0]:
                        self.GetArmor_class.reinit(new[0],new[1])
                        if energy_label:
                            self.GetEnergyMac_class.reinit(new[0])
                    #如果切换模式
                    elif new[1] != old[1]:
                        video_writer.set_mode(new[1])
                        reset_label = new[1]
                        if new[1] in [0,3]:
                            self.GetArmor_class.reinit(new[0],new[1])
                        elif new[1] in [1,2,4,5] and energy_label:
                            self.AnglePredicted_class.reinit(new[1])
                        #自瞄打符切换时重启相机
                        if (old[1] in [0,3] and new[1] in [1,2,4,5]) or (new[1] in [0,3] and old[1] in [1,2,4,5]) and energy_label:                            
                            self.GetFrame_class.restart_camera(new[1])
                    old = new
                    log.print_info('reset all done')
                


    def post_process(self):
        #图像处理线程
        msg_temp = [-1,-1,-1]
        while 1:
            #把一帧图像从缓冲区取出
            frame, f_time = self.frame.get()
            #根据模式信息来却动图像处理过程，确定是打符还是自瞄
            if type(frame) is np.ndarray:
                if reset_label in [0,3]:
                    msg_temp = list(self.GetArmor_class.GetArmor(frame))
                elif reset_label in [1,2,4,5] and energy_label:
                    msg_temp = self.GetEnergyMac_class.GetHitPointDL(frame)
                    msg_temp = list(self.AnglePredicted_class.NormalHit(msg_temp,f_time))
            #串口发送消息
            self.MySerial_class.send_message(msg_temp)

    
    def debug_show(self):
        #debug显示线程
        label = int(self.debug)
        timeout = self.timeout
        if label:
            cv2.namedWindow('colorTest')
            cv2.namedWindow('energyTest')
            cv2.namedWindow('armorTest')
            self.GetArmor_class.TrackerBar_create()
            if energy_label:
                self.GetEnergyMac_class.TrackerBar_create()
        while label:
            self.GetArmor_class.updata_argument()
            frame_set1 = self.GetArmor_class.get_debug_frame()
            for i,frame in enumerate(frame_set1):
                cv2.imshow('aimbot_'+str(i),frame)
            if energy_label:
                self.GetEnergyMac_class.updata_argument()
                frame_set2 = self.GetEnergyMac_class.get_debug_frame()
                msg_set = self.AnglePredicted_class.get_debug_msg()
                for i,frame in enumerate(frame_set2):
                    if i == 3:
                        cv2.circle(frame,(int(msg_set[0]),int(msg_set[1])),5,(0,255,255),-1)
                    cv2.imshow('energy_'+str(i),frame)
            k = cv2.waitKey(timeout)
            if k == 27:
                cv2.destroyAllWindows()
                self.__break_thread()
                log.print_info('debug_show return')
                sys.exit()


    def __get_debug(self,command_input):
        #处理命令行输入
        self.serial_debug = command_input.serial
        self.debug = int(command_input.debug)
        self.source_path = command_input.source
        #从json获取debug参数
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.timeout = load_dict["Debug"]["timeout"]
            self.video_writer_debug = load_dict["Debug"]["video_writer"]
            self.video_interval_fps = load_dict["Debug"]["video_interval_fps"]
        #判断数据来源
        if self.source_path == 'HIVISION':
            self.video_debug_set = 0
        elif self.source_path.isdigit():
            self.video_debug_set = 1
        elif os.path.isfile(self.source_path) and (self.source_path[-4:] == '.avi' or self.source_path[-3:] == '.mp4'):
            self.video_debug_set = 1
        elif os.path.isdir(self.source_path):
            self.video_debug_set = 2
        else:
            self.video_debug_set = 0
    
    def __break_thread(self):
        #debug下退出线程执行函数
        global break_label
        self.GetFrame_class.EndCamera()
        video_writer.release()
        break_label = True
        

#定义线程退出回调函数
def quit(signum, frame):
    global break_label
    log.print_info('get return sigh')
    break_label = True
    stop_thread(post_process_thread)
    stop_thread(grab_image_thread)
    stop_thread(debug_show_thread)

#通过手动引起异常终止线程
def stop_thread(th):
    th_id = th.ident
    th_id = ctypes.c_long(th_id)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(th_id, ctypes.py_object(SystemExit))
    if res == 0:
        raise ValueError("成功退出程序 序号："+str(th.ident))
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(th_id, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

#定义一个看门狗，程序非正常运行直接杀死线程，shell脚本会重新运行程序
def watch_dog():
    while True:
        if post_process_thread.is_alive() and grab_image_thread.is_alive() and not break_label:
            time.sleep(5)
            log.print_info('watchdog alive')
        else:
            log.print_error('watchdog died')
            stop_thread(post_process_thread)
            stop_thread(grab_image_thread)
            stop_thread(debug_show_thread)


#程序从这里开始
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TOE实验室视觉步兵程序调试帮助')
    parser.add_argument('--debug','-d', action='store_true', help='是否开启滑动条调参/debug模式,不输入默认不开启')
    parser.add_argument('--breakpoint','-b', action='store_true', help='是否开启断点模式,不输入默认不开启')
    parser.add_argument('--serial','-c', action='store_true', help='是否开启串口,不输入默认开启，输入后串口传入值固定从debug.json读取')
    parser.add_argument('--source','-s', type=str, default='HIVISION', help='输入识别数据的来源，可以是视频的路径或者图片文件夹的路径，不输入该选项默认海康摄像头，输入该选项默认usb摄像头')
    input = parser.parse_args()
    #日志设置输出等级，必须在最前
    log.set_level(input.debug)
    #若开启断点调试，导入库并初始化
    if input.breakpoint:
        import ipdb;ipdb.set_trace()
    #开始截获命令行的终止信号
    signal.signal(signal.SIGINT, quit)
    # 程序初始化
    infantry = Main(input)
    grab_image_thread = threading.Thread(target=infantry.grab_image)
    post_process_thread = threading.Thread(target=infantry.post_process)
    debug_show_thread = threading.Thread(target=infantry.debug_show)
    #设置各个线程为后台线程，保证一次性杀死
    post_process_thread.daemon = True
    grab_image_thread.daemon = True
    debug_show_thread.daemon = True
    #启动线程
    post_process_thread.start()
    grab_image_thread.start()
    debug_show_thread.start()
    #运行看门狗
    watch_dog()

