import logging
import time

class MyLogging:
    def __init__(self):
        #日志类初始化
        time_tuple = list(time.localtime(time.time()))
        #如果类中已经存在文件名，证明已初始化过，不再二次生成
        if hasattr(self,'value') == False:
            self.log_name = '-'
            self.log_name = './log/'+self.log_name.join([str(i) for i in time_tuple])+'.txt'
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

#调用logger模块时只进行一次初始化，引用时只使用已经实例化的对象
if __name__ == 'logger':
    log = MyLogging()