import cv2
import time
import json
import os
import logging
import inspect
import collections
import functools
import numpy as np


# 通用函数参数检查装饰器，需要配合函数注解表达式（Function Annotations）使用
# 注意本装饰器不能在log类中和他的继承使用
def para_check(func):
    msg = 'Argument {argument} must be {expected!r},but got {got!r},value {value!r}'
    # 获取函数定义的参数
    sig = inspect.signature(func)
    parameters = sig.parameters  # 参数有序字典
    arg_keys = tuple(parameters.keys())  # 参数名称
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        CheckItem = collections.namedtuple('CheckItem', ('anno', 'arg_name', 'value'))
        check_list = []
        #*args 传入的参数以及对应的函数参数注解
        for i, value in enumerate(args):
            arg_name = arg_keys[i]
            anno = parameters[arg_name].annotation
            #类本身的传参在函数中不做检查
            if arg_name != 'self':
                check_list.append(CheckItem(anno, arg_name, value))
        #**kwargs 传入的参数以及对应的函数参数注解
        for arg_name, value in kwargs.items():
            anno = parameters[arg_name].annotation
            check_list.append(CheckItem(anno, arg_name, value))
        #检查类型并生成错误信息
        label = True
        for item in check_list:
            if not isinstance(item.value, item.anno):
                error = msg.format(expected=item.anno, argument=item.arg_name,
                                   got=type(item.value), value=item.value)
                log.print_info(error)
                label = False
        if label:
            #参数正常执行函数
            return func(*args, **kwargs)

    return wrapper


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

    @para_check
    def write(self,frame:np.ndarray,time:float)-> None:
        if frame.shape[-1] != 3:
            frame = frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
        if self.mode:
            self.video_writer_energy.write(frame)
            self.file_energy.write(str(time)+'\n')
        else:
            self.video_writer_aimbot.write(frame)
            self.file_aimbot.write(str(time)+'\n')
    
    @para_check
    def set_mode(self,mode:int)-> None:
        if mode in [1,2,4,5]:
            self.mode = 1
        else:
            self.mode = 0
    
    def release(self):
        self.video_writer_aimbot.release()
        self.video_writer_aimbot.release()

#调用本文件时只进行一次初始化，引用时只使用已经实例化的对象
if __name__ == 'logger':
    log = MyLogging()
    video_writer = MyVideoWriter()

