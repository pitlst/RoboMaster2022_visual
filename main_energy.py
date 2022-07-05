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
from GetFrame_windows import GetFrame
from Communication import MySerial
from Aimbot_old import GetArmor
from EnergyMac import GetEnergyMac


class Main:
    def __init__(self):
        #串口初始化
        self.MySerial_class = MySerial()
        print('serial init done')
        self.singal = threading.Event()
        #多线程中跨线程调用资源使用资源锁，这里为初始化资源锁，缓冲区为10次，一般不会出现溢出的问题
        #如果出现溢出问题首先排查其他程序执行耗时问题，最后增大缓冲区
        self.frame = queue.Queue(10)
        self.reset_label = queue.Queue(5)
        self.break_all = queue.Queue(1)
        #该部分为debug显示用参数，根据debug参数创建和初始化
        self.get_debug()
        self.msg_temp = [-1,-1,-1]
        #将以后用得上的debug信息添加进类的变量
        temp = self.MySerial_class.get_msg_first()
        self.color_init = temp[0]
        self.mode_init = temp[1]
        print('mode:',self.mode_init)
        print('color',self.color_init)
        #初始化图像获取类，主要是对海康威视相机的操作
        self.GetFrame_class = GetFrame(self.mode_init)
        print('hikvision init done')
        #初始化自瞄类和卡尔曼滤波类，主要是对传统视觉的自瞄
        self.GetArmor_class = GetArmor(self.color_init,self.mode_init)
        print('aimbot init done')
        #初始化打符类，主要是对深度学习打符的操作
        self.GetEnergyMac_class = GetEnergyMac()
        print('energy init done')
        #根据标志位开始录像
        if self.debug[4]:
            if self.mode_init == 0 or self.mode_init == 3:
                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 10,(1440,864))
            elif self.mode_init == 1 or self.mode_init == 2 or self.mode_init == 4 or self.mode_init == 5:
                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 8,(768,768))
            print('video writer init done')
        print('all module init done')
 

    def grab_image(self):
        #图像获取线程
        t0 = time.time()
        t1 = time.time()
        fps = 0
        new = [-1,-1]
        old = [self.color_init,self.mode_init]
        video_debug = self.debug[4]
        temp_num = 0
        while 1:
            #按键退出
            if self.break_all.full():
                break
            if time.time()-t0 > 1:
                #每隔一秒输出一次
                print('fps:',fps)
                fps = 0
                t0 = time.time()
            #获取单帧图像
            frame = self.GetFrame_class.GetOneFrame()
            self.frame.put((frame,time.time()))
            fps += 1
            if video_debug:
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
                    print(new)
                    #模式与颜色原先不同时进行重置,并通过消息队列通知其他线程
                    if new != old:
                        cv2.destroyAllWindows()
                        if new[1] == 0 or new[1] == 3:
                            self.GetArmor_class = GetArmor(new[0],new[1])
                        if old[0] == new[0] and (((old[1] == 0 or old[1] == 3) and (new[1] == 0 or new[1] == 3)) or ((old[1] == 1 or old[1] == 2 or old[1] == 4 or old[1] == 5) and (new[1] == 1 or new[1] == 2 or new[1] == 4 or new[1] == 5))):
                            pass
                        else:
                            if (new[1] == 0 or new[1] == 3) and self.debug[4]:
                                self.video_writer.release()
                                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 10,(1440,864))
                            elif (new[1] == 1 or new[1] == 2 or new[1] == 4 or new[1] == 5) and self.debug[4]:
                                self.video_writer.release()
                                self.video_writer = cv2.VideoWriter(str(time.time())+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 8,(768,768))
                            self.GetFrame_class.restart_camera(new[1])
                        self.reset_label.put(new)
                        old = new
                        print('reset all done')
                t1 = time.time()


    def post_process(self):
        #图像处理线程
        t0 = time.time()
        old = [self.color_init,self.mode_init]
        msg_temp = [-1,-1,-1]
        interpolation_label = 0
        new = -1
        if self.debug[0]:
            cv2.namedWindow('colorTest')
            cv2.namedWindow('LBTest')
            cv2.namedWindow('armorTest')
            self.GetArmor_class.TrackerBar_create()
        if (self.debug[0] and (self.mode_init == 0 or self.mode_init == 3) ) or (self.debug[3] and (self.mode_init == 1 or self.mode_init == 2 or self.mode_init == 4 or self.mode_init == 5)):
            time_sleep = self.debug[1]
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
                        if self.debug[0]:
                            cv2.namedWindow('colorTest')
                            cv2.namedWindow('LBTest')
                            cv2.namedWindow('armorTest')
                            self.GetArmor_class.TrackerBar_create()
                            if (self.debug[0] and (self.mode_init == 0 or self.mode_init == 3)) or (self.debug[3] and (self.mode_init == 1 or self.mode_init == 2 or self.mode_init == 4 or self.mode_init == 5)):
                                time_sleep = self.debug[1]
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
                    if self.debug[4]:
                        self.video_writer.release()
                    #向其他线程广播消息
                    self.break_all.put(0)
                    print('return')
                    break


    def get_debug(self):
        #获取debug参数
        with open('./debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            show_debug = load_dict["Debug"]["show_debug"]
            timeout = load_dict["Debug"]["timeout"]
            kalmanfilter_enable = load_dict["Debug"]["kalmanfilter_enable"]
            Energy_debug = load_dict["Debug"]["Energy_debug"]
            video_writer_debug = load_dict["Debug"]["video_writer"]
        if show_debug == 0 and Energy_debug == 0:
            timeout = -1
        self.debug = [show_debug,timeout,kalmanfilter_enable,Energy_debug,video_writer_debug]


#程序从这里开始
if __name__ == '__main__':
    infantry = Main()
    grab_image_thread = threading.Thread(target=infantry.grab_image)
    post_process_thread = threading.Thread(target=infantry.post_process)
    #图像处理线程初始化时间比较高
    post_process_thread.start()
    grab_image_thread.start()