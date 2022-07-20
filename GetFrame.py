import sys
import numpy as np
import json
import cv2
import os
from logger import log
from ctypes import *
sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *


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
        self.read_json()

        self.getframe_label = True

        #读取校验，isinstance还有范围这些，先不写了，想写的话自行补上
        #这里定义的常值是一些不会经常更改的相机参数
        #具体定义查看在MVS的客户端中，或者翻到SetCamera方法
        self.DeviceLinkThroughputLimitMode = 0

        self.RegionSelector = 0
        self.ReverseX = 0
        self.PixelFormat = 0x01080009
        self.TestPattern = 0
        self.FrameSpecInfo = False

        self.AcquisitionMode = 2
        self.AcquisitionBurstFrameCount = 1
        self.AcquisitionFrameRate = 999.0
        self.AcquisitionFrameRateEnable = 1
        self.TriggerMode = 0
        self.TriggerCacheEnable = 0
        self.ExposureMode = 0
        self.ExposureAuto = 0
        self.HDREnable = 0

        self.Gain = 0.0
        self.GainAuto = 0
        self.BlackLevelEnable = 1
        self.BlackLevel = 100
        self.BalanceWhiteAuto = 2
        self.AutoFunctionAOIUsageIntensity = 0
        self.NUCEnable = 1

        self.LineInverter = 0
        self.StrobeEnable = 0

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
        with open('./json/common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            #自瞄所需要的相机参数
            aimbot_width = load_dict["Aimbot"]["width"]
            aimbot_height = load_dict["Aimbot"]["height"]
            aimbot_exposure_time = load_dict["Aimbot"]["exposure_time"]
            aimbot_offsetX = load_dict["Aimbot"]["offsetX"]
            aimbot_offsetY = load_dict["Aimbot"]["offsetY"]
            #打符所需要的相机参数
            em_width = load_dict["Energy_mac"]["width"]
            em_height = load_dict["Energy_mac"]["height"]
            em_exposure_time = load_dict["Energy_mac"]["exposure_time"]
            em_offsetX = load_dict["Energy_mac"]["offsetX"]
            em_offsetY = load_dict["Energy_mac"]["offsetY"] 
        #根据模式参数选择需要的参数
        if self.mode == 0 or self.mode == 3:
            self.width = aimbot_width
            self.height = aimbot_height
            self.exposure_time = aimbot_exposure_time
            self.offsetX = aimbot_offsetX
            self.offsetY = aimbot_offsetY
        elif self.mode == 1 or self.mode == 2 or self.mode == 4 or self.mode == 5:
            self.width = em_width
            self.height = em_height
            self.exposure_time = em_exposure_time
            self.offsetX = em_offsetX
            self.offsetY = em_offsetY
        else:
            log.print_error('unknow mode')
            self.width = 200
            self.height = 200
            self.exposure_time = 2000
            self.offsetX = 0
            self.offsetY = 0


    def StartCamera(self):
        if self.video_debug_set == 1:
            #从视频获取信息时需要初始化虚拟opencv相机
            self.cap = cv2.VideoCapture(self.video_debug_path)
        elif self.video_debug_set == 2:
            #从路径获取图片时需要获取图片文件列表
            self.files = os.listdir(self.video_debug_path)
        else:
            SDKVersion = MvCamera.MV_CC_GetSDKVersion()    #获取sdk版本号
            log.print_info("SDKVersion[0x%x]" % SDKVersion)

            deviceList = MV_CC_DEVICE_INFO_LIST() 
            self.deviceList = deviceList
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

            # ch:枚举设备 | en:Enum device
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0:
                log.print_error("enum devices fail! ret[0x%x]" % ret)
                sys.exit()

            if deviceList.nDeviceNum == 0:
                log.print_error("find no device!")
                sys.exit()

            log.print_info("Find %d devices!" % deviceList.nDeviceNum)

            for i in range(0, deviceList.nDeviceNum):
                mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                log.print_info("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                strModeName = strModeName + chr(per)
                log.print_info("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                log.print_info("user serial number: %s" % strSerialNumber)

            #取我们唯一的设备号
            nConnectionNum = "0"
            if int(nConnectionNum) >= deviceList.nDeviceNum:
                log.print_error("intput error!")
                sys.exit()
            cam = MvCamera() 
            self.cam = cam

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
        else:
            # ch:创建相机实例 | en:Creat Camera Object
            #取我们唯一的设备号
            nConnectionNum = "0"

            # ch:选择设备并创建句柄| en:Select device and create handle
            stDeviceList = cast(self.deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                log.print_error("create handle fail! ret[0x%x]" % ret)
                sys.exit()

            # 以下为常用且经常更改的信息
            # ch:打开设备 | en:Open device
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                log.print_error("open device fail! ret[0x%x]" % ret)
                if ret == 0x80000006:
                    log.print_info('try reboot camera...')
                    self.restart_camera(self.mode)
                sys.exit()
            # ch:设置触发模式为off | en:Set trigger mode as off
            # 这个实际意义为让相机自行处理图像的采集，不通过外部信号来控制
            # 该项目在图像硬同步时需开启
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                log.print_error("set trigger mode fail! ret[0x%x]" % ret)
                sys.exit()
            # ch：设置曝光时间，图像的长宽,和所取图像的偏移 | en:Set values
            # 注意，这里对offset的值应当提前归零，防止出现长度溢出问题
            ret = self.cam.MV_CC_SetIntValue("OffsetX", 0)
            if ret != 0:
                log.print_error("set offsetX fail! ret[0x%x]" % ret)
                sys.exit()
            ret = self.cam.MV_CC_SetIntValue("OffsetY", 0)
            if ret != 0:
                log.print_error("set offsetY fail! ret[0x%x]" % ret)
                sys.exit()
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
            if ret != 0:
                log.print_error("set ExposureTime fail! ret[0x%x]" % ret)
                sys.exit()
            ret = self.cam.MV_CC_SetIntValue("Width", self.width)
            if ret != 0:
                log.print_error("set Width fail! ret[0x%x]" % ret)
                sys.exit()
            ret = self.cam.MV_CC_SetIntValue("Height", self.height)
            if ret != 0:
                log.print_error("set Height fail! ret[0x%x]" % ret)
                sys.exit()
            ret = self.cam.MV_CC_SetIntValue("OffsetX", self.offsetX)
            if ret != 0:
                log.print_error("set offsetX fail! ret[0x%x]" % ret)
                sys.exit()
            ret = self.cam.MV_CC_SetIntValue("OffsetY", self.offsetY)
            if ret != 0:
                log.print_error("set offsetY fail! ret[0x%x]" % ret)
                sys.exit()

            #以下为不常用且如非必要不要更改的信息
            '''
            本相机不支持无损传输压缩
            对于测试模式的注释与讲解在程序末尾
            对于IInteger类型节点值：MV_CC_SetIntValue()
            对于IEnumeration类型节点值：MV_CC_SetEnumValue()
            对于IFloat类型节点值：MV_CC_SetFloatValue()
            对于IBoolean类型节点值：MV_CC_SetBoolValue()
            对于IString类型节点值：MV_CC_SetStringValue()
            对于ICommand类型节点值：MV_CC_SetCommandValue()
            对于ret报错信息：

                正确码：MV_OK
                    MV_OK 	                    0 	        成功，无错误
                通用错误码定义:范围0x80000000-0x800000FF
                    MV_E_HANDLE 	            0x80000000 	错误或无效的句柄
                    MV_E_SUPPORT            	0x80000001 	不支持的功能
                    MV_E_BUFOVER 	            0x80000002 	缓存已满
                    MV_E_CALLORDER          	0x80000003 	函数调用顺序错误
                    MV_E_PARAMETER 	            0x80000004 	错误的参数
                    MV_E_RESOURCE 	            0x80000006 	资源申请失败
                    MV_E_NODATA 	            0x80000007 	无数据
                    MV_E_PRECONDITION 	        0x80000008 	前置条件有误，或运行环境已发生变化
                    MV_E_VERSION 	            0x80000009 	版本不匹配
                    MV_E_NOENOUGH_BUF 	        0x8000000A 	传入的内存空间不足
                    MV_E_ABNORMAL_IMAGE 	    0x8000000B 	异常图像，可能是丢包导致图像不完整
                    MV_E_LOAD_LIBRARY 	        0x8000000C 	动态导入DLL失败
                    MV_E_NOOUTBUF 	            0x8000000D 	没有可输出的缓存
                    MV_E_UNKNOW 	            0x800000FF 	未知的错误
                GenICam系列错误:范围0x80000100-0x800001FF
                    MV_E_GC_GENERIC 	        0x80000100 	通用错误
                    MV_E_GC_ARGUMENT 	        0x80000101 	参数非法
                    MV_E_GC_RANGE 	            0x80000102 	值超出范围
                    MV_E_GC_PROPERTY 	        0x80000103 	属性
                    MV_E_GC_RUNTIME 	        0x80000104 	运行环境有问题
                    MV_E_GC_LOGICAL 	        0x80000105 	逻辑错误
                    MV_E_GC_ACCESS 	            0x80000106 	节点访问条件有误
                    MV_E_GC_TIMEOUT 	        0x80000107 	超时
                    MV_E_GC_DYNAMICCAST 	    0x80000108 	转换异常
                    MV_E_GC_UNKNOW 	            0x800001FF 	GenICam未知错误

            '''
            #DeviceControl部分
            #设置设备连接超时模式为关闭
            '''
            ret = self.cam.MV_CC_SetEnumValue("DeviceLinkThroughputLimitMode",self.DeviceLinkThroughputLimitMode)
            if ret != 0:
                log.print_info("set DeviceLinkThroughputLimitMode fail! ret[0x%x]" % ret)
                sys.exit()
            '''
            #ImageFormatControl部分
            #选择ROI区域为存档0,这是默认存档，我们只有这一个。
            ret = self.cam.MV_CC_SetEnumValue("RegionSelector",self.RegionSelector)
            if ret != 0:
                log.print_error("set RegionSelector fail! ret[0x%x]" % ret)
                sys.exit()
            '''
            #该区域设置的起始点(0,0)为左上角，横向为x,纵向为y，且向右和向下为正方向
            #设置相机选取区域的宽度
            ret = self.cam.MV_CC_SetIntValue("Width", self.width)
            if ret != 0:
                log.print_info("set Width fail! ret[0x%x]" % ret)
                sys.exit()
            #设置相机选取区域的高度
            ret = self.cam.MV_CC_SetIntValue("Height", self.height)
            if ret != 0:
                log.print_info("set Height fail! ret[0x%x]" % ret)
                sys.exit()
            #设置相机选取区域的横向偏移
            ret = self.cam.MV_CC_SetIntValue("OffsetX", self.offsetX)
            if ret != 0:
                log.print_info("set offsetX fail! ret[0x%x]" % ret)
                sys.exit()
            #设置相机选取区域的纵向偏移
            ret = self.cam.MV_CC_SetIntValue("OffsetY", self.offsetY)
            if ret != 0:
                log.print_info("set offsetY fail! ret[0x%x]" % ret)
                sys.exit()
            '''
            #设置相机传输画面是否需要水平翻转，该反转是在AOI修改之后应用的
            #该翻转通过相机本身完成，不是由电脑完成
            ret = self.cam.MV_CC_SetBoolValue("ReverseX",self.ReverseX)
            if ret != 0:
                log.print_error("set ReverseX fail! ret[0x%x]" % ret)
                sys.exit()
            '''
            设置相机传输的图像的格式
            packed为数据填充方式，空余部分填充下一帧数据，其余的空余部分填充0
            关于拜耳格式数据差值方法参考官方文档，我们暂时选取BayerRG8格式(2022.2.16)
            以下是格式对应的参数对照表
            Mono8格式：0x01080001                       单色黑白8bit色深
            Mono10格式：0x01100003                      单色黑白10bit色深，以16bit格式填充
            Mono12格式：0x01100005                      单色黑白12bit色深，以16bit格式填充
            RGB8Packed格式：0x02180014                  彩色三通道8bit*3=24bit色深，
            YUV422_YUYV_Packed格式：0x02100032          从byer转rgb再转yuv，16bit色深，排列格式为YUYV
            YUV422Packed格式：0x0210001F                从byer转rgb再转yuv，16bit色深，排列格式为UYVY
            BayerRG8格式：0x01080009                    彩色相机原始数据格式，由Bayer12下采样得到，每像素8bit色深
            BayerRG10格式：0x0110000d                   彩色相机原始数据格式，由Bayer12下采样得到，每像素10bit色深
            BayerRG10Packed格式：0x010C0027             彩色相机原始数据格式，由Bayer12下采样得到，每像素10bit色深
            BayerRG12格式：0x01100011                   彩色相机原始数据格式，每像素12bit色深
            BayerRG12Packed格式：0x010C002B             彩色相机原始数据格式，每像素12bit色深
            '''
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat",self.PixelFormat)
            if ret != 0:
                log.print_error("set PixelFormat fail! ret[0x%x]" % ret)
                sys.exit()
            #设置关闭测试模式
            ret = self.cam.MV_CC_SetEnumValue("TestPattern",self.TestPattern)
            if ret != 0:
                log.print_error("set TestPattern fail! ret[0x%x]" % ret)
                sys.exit()
            #关闭帧水印
            ret = self.cam.MV_CC_SetBoolValue("FrameSpecInfo",self.FrameSpecInfo)
            if ret != 0:
                log.print_error("set FrameSpecInfo fail! ret[0x%x]" % ret)
                sys.exit()
            
            #AcquisitionControl部分
            #设置设备的采集模式
            ret = self.cam.MV_CC_SetEnumValue("AcquisitionMode",self.AcquisitionMode)
            if ret != 0:
                log.print_error("set AcquisitionMode fail! ret[0x%x]" % ret)
                sys.exit()
            #设置每次触发获得帧数为1
            #该项目在触发方式为外部触发时启用，用于设定一次触发拍摄几张图片
            #ret = self.cam.MV_CC_SetIntValue("AcquisitionBurstFrameCount",self.AcquisitionBurstFrameCount)
            #if ret != 0:
            #    log.print_info("set AcquisitionBurstFrameCount fail! ret[0x%x]" % ret)
            #    sys.exit()
            #设置获取帧率最大值为999
            ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate",self.AcquisitionFrameRate)
            if ret != 0:
                log.print_error("set AcquisitionFrameRate fail! ret[0x%x]" % ret)
                sys.exit()
            #启用对摄像机帧速率的手动控制
            ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable",self.AcquisitionFrameRateEnable)
            if ret != 0:
                log.print_error("set AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
                sys.exit()
            #关闭触发器
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode",self.TriggerMode)
            if ret != 0:
                log.print_error("set TriggerMode fail! ret[0x%x]" % ret)
                sys.exit()
            #指定是否启用触发器缓存
            #该项目设置外部触发时启用，用于临时保存未来得及处理的后续触发，不开启则优先处理新触发
            #ret = self.cam.MV_CC_SetBoolValue("TriggerCacheEnable",self.TriggerCacheEnable)
            #if ret != 0:
            #    log.print_info("set TriggerCacheEnable fail! ret[0x%x]" % ret)
            #    sys.exit()
            #设置曝光或者快门的运行方式
            ret = self.cam.MV_CC_SetEnumValue("ExposureMode",self.ExposureMode)
            if ret != 0:
                log.print_error("set ExposureMode fail! ret[0x%x]" % ret)
                sys.exit()
            '''        
            #设置曝光时间
            ret = cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
            if ret != 0:
                log.print_info("set ExposureTime fail! ret[0x%x]" % ret)
                sys.exit()
            '''
            #设置关闭自动曝光
            ret = self.cam.MV_CC_SetEnumValue("ExposureAuto",self.ExposureAuto)
            if ret != 0:
                log.print_error("set ExposureAuto fail! ret[0x%x]" % ret)
                sys.exit()
            #关闭HDR轮询，我们不需要高动态范围
            ret = self.cam.MV_CC_SetBoolValue("HDREnable",self.HDREnable)
            if ret != 0:
                log.print_error("set HDREnable fail! ret[0x%x]" % ret)
                sys.exit()

            #AnalogControl部分
            #增益gain设置为0
            #对于增益，可以简单理解为将像素值固定乘上增益，来做到提高亮部降低暗部的效果
            #本质上增益就是信号放大器，但是同理噪声也同样放大，所以不启用增益，设置为0
            ret = self.cam.MV_CC_SetFloatValue("Gain",self.Gain)
            if ret != 0:
                log.print_error("set Gain fail! ret[0x%x]" % ret)
                sys.exit()    
            #关闭自动增益控制
            ret = self.cam.MV_CC_SetEnumValue("GainAuto",self.GainAuto)
            if ret != 0:
                log.print_error("set GainAuto fail! ret[0x%x]" % ret)
                sys.exit()
            '''         
            开启黑电平调整
            黑电平本质上就是把所有图像信号增加一个固定的值，由于相机在读取电平信号时对于极小值不敏感
            所以就人为增加一个值来让暗部的细节得到更多的保留，当然会损失一部分最亮的细节
            但是我们比赛时更需要暗部细节，而且丢失亮部细节的问题可以通过调低曝光时间来解决
            还有一部分原因是相机的电路本身会存在暗电流
            暗电流主要产生在CMOS芯片通过光电二极管将光信号转化成模拟信号的过程中
            光电二极管受到温度，电压稳定性等因素的干扰
            导致全黑状态下输出的电平并不一定稳定为0
            而信号的不稳定会导致部分图像的偏色
            人为将全黑状态的数值固定钳制在黑电平这个值
            很大程度上是能保证信号的稳定性，从而保证全图图像效果表现一致。
            '''
            ret = self.cam.MV_CC_SetBoolValue("BlackLevelEnable",self.BlackLevelEnable)
            if ret != 0:
                log.print_error("set BlackLevelEnable fail! ret[0x%x]" % ret)
                sys.exit()
            #设置黑电平调整参数
            ret = self.cam.MV_CC_SetIntValue("BlackLevel", self.BlackLevel)
            if ret != 0:
                log.print_error("set BlackLevel fail! ret[0x%x]" % ret)
                sys.exit()
            #关闭连续自动白平衡，在平均亮度突变时会在暗光情况下将橙红色更改为青绿色，无法真的自动白平衡
            #这部分设置为根据当前场景，运行一段时间自动白平衡后停止
            #比赛前应当在赛场进行白平衡校准，并更改为手动模式
            #校准方式在文件末
            ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", self.BalanceWhiteAuto)
            if ret != 0:
                log.print_error("set BalanceWhiteAuto fail! ret[0x%x]" % ret)
                sys.exit()
            #关闭自动AOI
            ret = self.cam.MV_CC_SetBoolValue("AutoFunctionAOIUsageIntensity",self.AutoFunctionAOIUsageIntensity)
            if ret != 0:
                log.print_error("set AutoFunctionAOIUsageIntensity fail! ret[0x%x]" % ret)
                sys.exit()
            '''
            #启用非均匀性校正，用于屏蔽相机自身对相机图像产生的影响，本机只有明度矫正
            ret = self.cam.MV_CC_SetBoolValue("NUCEnable",self.NUCEnable)
            if ret != 0:
                log.print_info("set NUCEnable fail! ret[0x%x]" % ret)
                sys.exit()
            '''
            #DigitalIOControl部分
            #关闭所选输入或输出线的信号反转
            '''
            ret = self.cam.MV_CC_SetBoolValue("LineInverter",self.LineInverter)
            if ret != 0:
                log.print_info("set LineInverter fail! ret[0x%x]" % ret)
                sys.exit()
            
            #禁止选通信号输出到选定的行，这功能我们用不上
            ret = self.cam.MV_CC_SetBoolValue("StrobeEnable",self.StrobeEnable)
            if ret != 0:
                log.print_info("set StrobeEnable fail! ret[0x%x]" % ret)
                sys.exit()  
            '''
            #剩下的几项参数即用不上，又都对相机成像没有影响，硬件debug部分也不需要
            #所以就不用设置了


            # ch:获取数据包大小 | en:Get payload size
            stParam =  MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                log.print_error("get payload size fail! ret[0x%x]" % ret)
                sys.exit()
            nPayloadSize = stParam.nCurValue

            # ch:开始取流 | en:Start grab image

            ret = self.cam.MV_CC_StartGrabbing()

            if ret != 0:
                log.print_error("start grabbing fail! ret[0x%x]" % ret)
                self.EndCamera()
                sys.exit()

            self.data_buf = (c_ubyte * nPayloadSize)()
            pData,nDataSize = self.data_buf, nPayloadSize
            self.pData = pData
            self.nDataSize = nDataSize


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
        else:
            #如果上一次没有成功取流，重启相机
            if not self.getframe_label: 
                self.restart_camera(self.mode)
            pData = self.pData
            stFrameInfo = MV_FRAME_OUT_INFO_EX()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
            temp = np.array(pData)
            ret = self.cam.MV_CC_GetOneFrameTimeout(byref(pData), self.nDataSize, stFrameInfo, 1000)
            if ret != 0:
                log.print_error("get frame fail! ret[0x%x]" % ret)
                self.getframe_label = False
            else:
                self.getframe_label = True
            temp = temp.reshape((int(self.height),int(self.width)))
            return temp

    def EndCamera(self):
        if self.video_debug_set == 1:
            self.cap.release()
        elif self.video_debug_set == 2:
            self.count = None
        else:
            # ch:停止取流 | en:Stop grab image
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                log.print_error("stop grabbing fail! ret[0x%x]" % ret)
                del self.data_buf
                sys.exit()

            # ch:关闭设备 | Close device
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                log.print_error("close deivce fail! ret[0x%x]" % ret)
                del self.data_buf
                sys.exit()

            # ch:销毁句柄 | Destroy handle
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                log.print_error("destroy handle fail! ret[0x%x]" % ret)
                del self.data_buf
                sys.exit()

'''
测试模式是用于检测相机是否有异常，方便排查问题
注意，以下程序未运行，仅作为例子参考

测试模式     本相机不支持显示测试图像
off:0                    关闭
ColorBar:9               垂直彩条
MonoBar:11               黑白竖条
HorizontalColorBar:12    水平彩条
Checkboard:13            棋盘格
ObliqueMonoBar:14        斜向渐变灰度条
GradualMonoBar:15        渐变灰度条纹

ret = self.cam.MV_CC_SetEnumValue("TestPattern",self.TestPattern)
if ret != 0:
    log.print_info("set TestPattern fail! ret[0x%x]" % ret)
    sys.exit()

本机仅支持time曝光模式，所以曝光方式只有三种，
手动
一次自动
连续自动
off: 0
once:1
continuous:2
在连续模式下，曝光只能在AutoExposureTimeLowerLimit属性和AutoExposureTimeUpperLimit之间
但是我们一般不用，固定曝光时长，稳定图像获取，一个稳定的帧率很重要
ret = self.cam.MV_CC_SetEnumValue("ExposureAuto",self.ExposureAuto)
if ret != 0:
    log.print_info("set ExposureAuto fail! ret[0x%x]" % ret)
    sys.exit()
相机增益有模拟增益和数字增益2种。模拟增益可将模拟信号放大；数字增益可将模数转换后的信号放大。
增益数值越高时，图像亮度也越高，同时图像噪声也会增加，对图像质量有所影响。且数字增益的噪声会比模拟增益的噪声更明显
若需要提高图像亮度，建议先增大相机的曝光时间；若曝光时间达到环境允许的上限不能满足要求，再考虑增大模拟增益；
若模拟增益设置为最大值还不能满足要求，最后再考虑调整数字增益

gain参数是模拟增益
gain auto用于确认是自动增益还是手动增益

相机数字增益默认为0且不启用，范围为−6 ~ 6 dB

相机亮度为一次自动或连续自动曝光和增益模式调整图像时的参考亮度。若相机为手动曝光模式，则亮度参数无效
亮度的更改是通过更改曝光时间和增益实现的，启用顺序为曝光-模拟增益-数字增益

本机的黑电平调节是平均灰度，在所有像素上直接加相同数值

白平衡需要在比赛前校准
（尽量不用自动白平衡矫正，效果上想调好要求比较严苛）
1. 将 Balance White Auto 参数由 Continuous 或 Once 切换为 off 即手动白平衡模式
2. 找到数值为 1024 的 R/G/B 中的某个分量，观察图像的 R/G/B 数值，调节其他两个分
   量的数值使得 R/G/B 三通道达到一致。此时图像色彩与实际色彩接近，完成白平衡校
   准。
windows上MVS客户端有白平衡矫正工具，记得在校正后将用户数据保存到相机中

彩色相机 Bayer 格式下不支持 Gamma 校正
相机仅在 Mono 格式和 YUV 格式下支持锐度功能
这部分更改可在主函数和主线程中实现，不需要让相机执行

降噪会导致图像锐度下降，大部分噪声影响不到实际图像处理，在二值化取蒙版部分已经滤掉，所以不启用

aoi是相机在一定范围内自动调节曝光时间和白平衡，以最大限度达到用户期望的画质
我们不需要自动曝光和自动白平衡，所以关闭aoi

PRNUC（明场矫正）为官方参数，官方并没有对这部分过多介绍，测试下来关闭与开启也没有过多效果
根据官方建议开启

'''