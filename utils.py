import cv2
import time
import json
import os
import logging
import numpy as np

#用于c++同学复健
endl = '\n'

class cpp:
    def __lshift__(self,other):
        '''
        重载左移运算符,用于c++转python的适应期(笑
        示例写法：
        cout << 111 <<< endl
        输出：
        111
        '''
        print(other,end='')
        return self

#用于比赛

#定义一个装饰器，用于保证函数调用次数
def count(func):
    num = 0  # 初始化次数
    result = 0 #初始化结果
    def call_fun(*args, **kwargs):
        nonlocal num # 声明num 变当前作用域局部变量为最临近外层（非全局）作用域变量。
        nonlocal result
        num += 1 # 每次调用次数加1
        if num < 1000:
            #只有在最开始的1000次调用，函数才会真正执行并更改结果
            result = func(*args, **kwargs)#原函数
        return result

    return call_fun

class MyLogging:
    def __init__(self):
        #日志类初始化
        time_tuple = list(time.localtime(time.time()))
        #创建日志文件夹
        if os.path.exists('./log/') == False:
            os.mkdir('./log/') 
        #如果类中已经存在文件名，证明已初始化过，不再二次生成
        if hasattr(self,'value') == False:
            self.log_name = '-'
            self.log_name = './log/'+self.log_name.join([str(i) for i in time_tuple])
        self.logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", '%Y-%m-%d %H:%M:%S')
        #命令行日志输出记录
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        #文件日志输出记录
        fhlr = logging.FileHandler(self.log_name)
        fhlr.setFormatter(formatter)
        self.logger.handlers = []
        self.logger.addHandler(fhlr)
        self.logger.addHandler(chlr)

    def set_level(self,level):
        #设定日志输出等级
        if level:
            self.logger.setLevel('DEBUG')
            logging.debug('this is debug')
        else:
            self.logger.setLevel('INFO')
            logging.info('this is info')

    #以下为静态方法，对应不同等级的输入输出
    @staticmethod
    def print_info(content):
        logging.info(content)

    @staticmethod
    def print_debug(content):
        logging.debug(content)

    @staticmethod
    def print_warning(content):
        logging.warning(content)

    @staticmethod
    def print_error(content):
        logging.error(content)

    @staticmethod
    def print_critical(content):
        logging.critical(content)



class MyVideoWriter:
    def __init__(self):
        #相机类初始化
        self.__read_json()
        self.mode = 1
        time_tuple = list(time.localtime(time.time()))
        video_name = '-'
        video_name_aimbot = './log/'+'aimbot_'+video_name.join([str(i) for i in time_tuple])+'.avi'
        video_name_energy = './log/'+'energy_'+video_name.join([str(i) for i in time_tuple])+'.avi'
        file_name_aimbot = './log/'+'aimbot_'+video_name.join([str(i) for i in time_tuple])+'.txt'
        file_name_energy = './log/'+'energy_'+video_name.join([str(i) for i in time_tuple])+'.txt'
        self.video_writer_aimbot = cv2.VideoWriter(video_name_aimbot, cv2.VideoWriter_fourcc(*'XVID'), int(self.video_fps),(int(self.Aimbot_width),int(self.Aimbot_height)))
        self.video_writer_energy = cv2.VideoWriter(video_name_energy, cv2.VideoWriter_fourcc(*'XVID'), int(self.video_fps),(int(self.Energy_mac_width),int(self.Energy_mac_height)))
        self.file_aimbot = open(file_name_aimbot,'w',encoding="utf-8")
        self.file_energy = open(file_name_energy,'w',encoding="utf-8")
    
    def __read_json(self):
        with open('./json/common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.Aimbot_width = load_dict["Aimbot"]["width"]
            self.Aimbot_height = load_dict["Aimbot"]["height"]
            self.Energy_mac_width = load_dict["Energy_mac"]["width"]
            self.Energy_mac_height = load_dict["Energy_mac"]["height"]
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.video_fps = int(load_dict["Debug"]["video_fps"])

    def write(self,frame,time):
        if isinstance(frame,np.ndarray):
            if frame.shape[-1] != 3:
                frame = frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
            if self.mode:
                self.video_writer_energy.write(frame)
                self.file_energy.write(str(time)+'\n')
            else:
                self.video_writer_aimbot.write(frame)
                self.file_aimbot.write(str(time)+'\n')

    
    def set_mode(self,mode):
        if mode in [1,2,4,5]:
            self.mode = 1
        else:
            self.mode = 0
    
    def release(self):
        self.video_writer_aimbot.release()
        self.video_writer_aimbot.release()



#调用本文件时只进行一次初始化，引用时只使用已经实例化的对象
if __name__ == 'utils':
    log = MyLogging()
    video_writer = MyVideoWriter()
    cout = cpp()
    para_json = json_dict()

