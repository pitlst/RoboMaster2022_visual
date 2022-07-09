import serial
import serial.tools.list_ports
import json
import time
from logger import log
from struct import *


class MySerial:

    def __init__(self,serial_debug):
        '''串口通信初始化
        @para com: 端口
        @para bps: 波特率
        @para timeout: 最大读取超时
        @attention
            函数原型默认内容8位，无奇偶校验，1位终止位，此处保持默认
        '''
        #读取json文件中的参数
        self.read_communication()
        self.com_debug = int(serial_debug)
        if self.com_debug == 0:
            while 1:
                try:
                    self.my_engine = serial.Serial(self.com, self.bps, timeout = self.timeout)
                    break
                except:
                    self.reset_serial()
        #这是crc校验用的，需要与电控统一
        self.crc8=[
            0x00,0x31,0x62,0x53,0xc4,0xf5,0xa6,0x97,0xb9,0x88,0xdb,0xea,0x7d,0x4c,0x1f,0x2e,
            0x43,0x72,0x21,0x10,0x87,0xb6,0xe5,0xd4,0xfa,0xcb,0x98,0xa9,0x3e,0x0f,0x5c,0x6d,
            0x86,0xb7,0xe4,0xd5,0x42,0x73,0x20,0x11,0x3f,0x0e,0x5d,0x6c,0xfb,0xca,0x99,0xa8,
            0xc5,0xf4,0xa7,0x96,0x01,0x30,0x63,0x52,0x7c,0x4d,0x1e,0x2f,0xb8,0x89,0xda,0xeb,
            0x3d,0x0c,0x5f,0x6e,0xf9,0xc8,0x9b,0xaa,0x84,0xb5,0xe6,0xd7,0x40,0x71,0x22,0x13,
            0x7e,0x4f,0x1c,0x2d,0xba,0x8b,0xd8,0xe9,0xc7,0xf6,0xa5,0x94,0x03,0x32,0x61,0x50,
            0xbb,0x8a,0xd9,0xe8,0x7f,0x4e,0x1d,0x2c,0x02,0x33,0x60,0x51,0xc6,0xf7,0xa4,0x95,
            0xf8,0xc9,0x9a,0xab,0x3c,0x0d,0x5e,0x6f,0x41,0x70,0x23,0x12,0x85,0xb4,0xe7,0xd6,
            0x7a,0x4b,0x18,0x29,0xbe,0x8f,0xdc,0xed,0xc3,0xf2,0xa1,0x90,0x07,0x36,0x65,0x54,
            0x39,0x08,0x5b,0x6a,0xfd,0xcc,0x9f,0xae,0x80,0xb1,0xe2,0xd3,0x44,0x75,0x26,0x17,
            0xfc,0xcd,0x9e,0xaf,0x38,0x09,0x5a,0x6b,0x45,0x74,0x27,0x16,0x81,0xb0,0xe3,0xd2,
            0xbf,0x8e,0xdd,0xec,0x7b,0x4a,0x19,0x28,0x06,0x37,0x64,0x55,0xc2,0xf3,0xa0,0x91,
            0x47,0x76,0x25,0x14,0x83,0xb2,0xe1,0xd0,0xfe,0xcf,0x9c,0xad,0x3a,0x0b,0x58,0x69,
            0x04,0x35,0x66,0x57,0xc0,0xf1,0xa2,0x93,0xbd,0x8c,0xdf,0xee,0x79,0x48,0x1b,0x2a,
            0xc1,0xf0,0xa3,0x92,0x05,0x34,0x67,0x56,0x78,0x49,0x1a,0x2b,0xbc,0x8d,0xde,0xef,
            0x82,0xb3,0xe0,0xd1,0x46,0x77,0x24,0x15,0x3b,0x0a,0x59,0x68,0xff,0xce,0x9d,0xac
        ]

    def read_communication(self):
        #读取串口参数
        with open('./json/common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.com = load_dict["UARTSet"]["com"]
            self.bps = load_dict["UARTSet"]["bps"]
            self.timeout = load_dict["UARTSet"]["timeout"]
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.com_debug_color = load_dict["Debug"]["com_debug_color"]
            self.com_debug_mode = load_dict["Debug"]["com_debug_mode"]
        
    def send_message(self, m):
        '''发送数据
        @para message(bytes)
        '''
        x = float(m[0])
        y = float(m[1])
        z = float(m[2])
        #将数据打包
        m = self.pack_data(x,y,z)
        #print(int(x),int(y),int(z))
        #数据串口发送
        if self.com_debug == 0:
            while 1:
                try:
                    self.my_engine.write(m)
                    break
                except:
                    self.reset_serial()


    def get_msg_first(self):
        '''接收第一次数据
        @return blue/red
        @return mode(打福/自瞄)
        '''
        t0 = time.time()
        while 1:
            t1 = time.time()
            color, mode = self.get_msg()
            if color != -1 and mode != -1:
                break
            elif t1 - t0 > 5:
                color, mode = self.com_debug_color, self.com_debug_mode
                break
        return color, mode
    
    def get_msg(self):
        '''接收数据
        @return blue/red
        @return mode(打福/自瞄)
        
        读取采用一次读取全部，取最新使用
        请务必反复测试以下两项内容
        1.不要让串口阻塞太久，缓冲区堆集太多数据
        2.不要让读取空数据的次数太多，以防影响代码运行效率
        以上两点通过合理修改嵌入式发送数据频率实现，不要过快或者过慢

        '''
        #初始化接收缓冲区
        data = []
        #判断是否debug
        if self.com_debug == 0:
            while 1:
                try:
                    data = self.my_engine.readlines()
                    break
                except:
                    self.reset_serial()
            #判断是否接受到信息
            if len(data):
                data = data[-1]
                if data[0] == 166:
                    # 根据机器人id来判断颜色
                    # 10打头大于100的为蓝色
                    # 小于100的一位int数为红色
                    # 自己为蓝方就去识别红方，自己为红方就识别蓝方
                    if data[1] > 100:
                        my_color = 1
                    elif data[1] > 0 and data[1] < 10:
                        my_color = 0
                    else:
                        log.print_error('error: unknow color num')
                        my_color = -1
                    #判断模式为打符还是自瞄，自瞄为0，打小符为1，打大符为2，打哨兵为3
                    if data[2] > 0 or data[2] < 10:
                        my_mode = data[2]
                    else:
                        my_mode = -1
                    return my_color, my_mode
                else:
                    return -1,-1
            #未初始化成功直接返回-1
            else:
                return -1, -1
        else:
            #开启串口debug模式后，不再接受电控的消息，固定传值
            return self.com_debug_color, self.com_debug_mode

    def close_engine(self):
        #关闭串口
        if self.com_debug == 0:
            self.my_engine.close()

    def reset_serial(self):
        #在串口失效后重启串口
        log.print_warning('reset_serial...')
        i = 0
        while 1:
            #重置串口号
            temp = self.com[:-1] + str(i)
            i = i + 1
            if i > 10:
                i = 0 
            try:
                #尝试重新初始化串口
                self.my_engine = serial.Serial(temp, self.bps, timeout = self.timeout)
                log.print_info('reset_serial success')
                break
            except:
                continue


#以下部分暂时未使用，视电控需要随时更改

    def get_CRC(self, data, data_length):
        #生成crc检验码，方法可自行学习
        #注意crc8需要与电控相同
        crc = 0
        l = data_length - 1
        a = 0
        while l > 0:
            crc = self.crc8[crc ^ data[a]]
            a += 1
            l -= 1
        return crc

    def pack_data(self, x, y, z):
        #打包需要发送的数据，这部分需要与电控协商传输数据的形式
        # flag = True
        # if z == -1:
        #     flag = False
        message = pack('fff', x, y, z)
        # crc = get_CRC(message, 14)
        # message = pack('fff?H', x, y, z, flag, crc)
        # message = pack('fff',x,y,z)
        #print(message)
        return message

#crc校验方式和结果应当与嵌入式相同，需要协商与实验。    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    

