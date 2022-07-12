import cv2
import numpy as np
import math
import json
import time
from logger import log

class GetArmor:

    def __init__(self,debug,video_debug_set,color,mode):
        #传入变量私有化
        self.mode = mode
        self.color = color
        self.debug = debug
        self.video_debug_set = video_debug_set
        #记录opencv版本的标志位
        if int(cv2.__version__[0]) == 4:
            self.version = 1
        else:
            self.version = 0 
        self.read_aimbot()
        #初始化debug参数
        if self.debug:
            self.getvar_label = False
            self.frame_debug = np.array([0]).astype(np.float32)
            self.mask = np.array([0]).astype(np.float32)
            self.lightBarList = []
            self.realCenter_list = []
            self.x = -1
            self.y = -1
            self.t0 = time.time()
    
    def reinit(self,color,mode):
        #更改变量时重初始化，更新参数
        self.mode = mode
        self.color = color
        self.read_aimbot()

    def read_aimbot(self):
        #该函数用于读取自瞄参数
        if self.mode == 0:
            log.print_info('aimbot mode')
            self.path = './json/armor_find.json'
        elif self.mode == 3:
            log.print_info('sentry mode')
            self.path = './json/sentry_find.json'
        else:
            self.path = './json/armor_find.json'
        with open(self.path,'r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            if self.color == 0:
                hsvPara_high = load_dict["ImageProcess_red"]["hsvPara_high"]
                hsvPara_low = load_dict["ImageProcess_red"]["hsvPara_low"]
                self.hsvPara = np.array([hsvPara_low,hsvPara_high])
            elif self.color == 1:
                hsvPara_high = load_dict["ImageProcess_blue"]["hsvPara_high"]
                hsvPara_low = load_dict["ImageProcess_blue"]["hsvPara_low"]
                self.hsvPara = np.array([hsvPara_low,hsvPara_high])
            else:
                log.print_error('unknow color num')
                hsvPara_high = [-1,-1,-1]
                hsvPara_low = [-1,-1,-1]
                self.hsvPara = np.array([hsvPara_low,hsvPara_high])
                self.GBSet_size = [-1,-1]
            self.lowHue = hsvPara_low[0]
            self.lowSat = hsvPara_low[1]
            self.lowVal = hsvPara_low[2]
            self.highHue = hsvPara_high[0]
            self.highSat = hsvPara_high[1]
            self.highVal = hsvPara_high[2]
            #灯条筛选
            self.minlighterarea = load_dict["ArmorFind"]["minlighterarea"]
            self.maxlighterarea = load_dict["ArmorFind"]["maxlighterarea"]
            self.minlighterProp = load_dict["ArmorFind"]["minlighterProp"]
            self.maxlighterProp = load_dict["ArmorFind"]["maxlighterProp"]
            self.minAngleError = load_dict["ArmorFind"]["minAngleError"]
            self.maxAngleError = load_dict["ArmorFind"]["maxAngleError"]
            #装甲板筛选
            self.minarealongRatio = load_dict["ArmorFind"]["minarealongRatio"]
            self.maxarealongRatio = load_dict["ArmorFind"]["maxarealongRatio"]
            self.angleDiff = load_dict["ArmorFind"]["angleDiff"]
            self.lightBarAreaDiff = load_dict["ArmorFind"]["lightBarAreaDiff"]
            self.armorAngleMin = load_dict["ArmorFind"]["armorAngleMin"]
            self.minarmorArea = load_dict["ArmorFind"]["minarmorArea"]
            self.maxarmorArea = load_dict["ArmorFind"]["maxarmorArea"]
            self.minarmorProp = load_dict["ArmorFind"]["minarmorProp"]
            self.maxarmorProp = load_dict["ArmorFind"]["maxarmorProp"]
            self.minBigarmorProp = load_dict["ArmorFind"]["minBigarmorProp"]
            self.maxBigarmorProp = load_dict["ArmorFind"]["maxBigarmorProp"]
            #测距
            self.kh = load_dict["ArmorFind"]["kh"]
        with open('./json/common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.xCenter = load_dict["Aimbot"]["width"]/2
            self.yCenter = load_dict["Aimbot"]["height"]/2

    def GetArmor(self,frame):
        #识别装甲板主程序
        x, y, z = -1, -1, -1
        #根据debug参数更改图像来源格式
        if self.video_debug_set == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)
        mask = self.HSV_Process(frame)# 进行HSV图像处理
        lightBarList = self.GetLightBar(mask)# 进行获取灯条矩形列表
        x,y,z = self.CombineLightBar(lightBarList)#非卡尔曼滤波，将灯条拼凑成装甲板
        #根据debug参数确定显示图像
        if self.debug:
            self.mask = mask
            self.frame_debug = frame.copy()
            #每秒更新1次json文件中的参数
            if time.time()-self.t0 > 1:
                self.update_json()
                self.t0 = time.time()
        return x,y,z
    

    def HSV_Process(self,frame):
        #二值化图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(frame, self.hsvPara[0], self.hsvPara[1])
        return mask


    def GetLightBar(self,mask):
        #筛选灯条
        lightBarList = []
        #对于opencv3，轮廓寻找函数有3个返回值，对于opencv4只有两个
        if self.version:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        else:
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        #轮廓点数小于40默认误识别
        if(len(contours)>40):
            lightBarList = []
            return lightBarList
        for contour in range(len(contours)):
            if len(contours[contour]) >= 5:
                center, size, angle = cv2.fitEllipse(contours[contour])#反馈的长宽其实就是长的轴和短的轴，所以一定长宽比大于1
                if size[1] <= 0 or size[0] <= 0 or np.isnan(size[0]) or np.isnan(size[1]) :
                    continue
                rectProp = size[1]/size[0]#灯条长比宽，值大于1
                rectArea = size[0]*size[1]
                angleHori = abs(int(angle)%180)
                if  angleHori > self.minAngleError and angleHori < self.maxAngleError:
                    ## log.print_debug("angleHori = "+str(angleHori))
                    continue
                if rectProp < self.minlighterProp or rectProp > self.maxlighterProp:#得介于最小值最大值之间才要
                    ## log.print_debug("rectProp = "+str(rectProp))
                    continue
                if rectArea > self.maxlighterarea or rectArea < self.minlighterarea:
                    ## log.print_debug("rectArea = "+str(rectArea))
                    continue
                if self.debug:
                        lightBarList.append([center[0],center[1],size[0],size[1],angle,rectProp,angleHori,rectArea])#size[0]是灯条宽 size[1]是灯条长
                else:
                        lightBarList.append([center[0],center[1],size[0],size[1],angle]) 
                ## log.print_debug("size[0]="+str(size[0]),"size[1]="+str(size[1]))
            else :
                # log.print_debug("<5")
                pass

        if self.debug:
            #如果开启了debug模式，向类变量更新值
            self.lightBarList = lightBarList
        return lightBarList


    def CombineLightBar(self,lightBarList):
        #筛选装甲板
        realCenter_list = []
        realCenter = [-1, -1]
        x,y,z = -1,-1,-1
        #灯条数量没有两个，构建不出装甲板
        if len(lightBarList) <  2:
            # log.print_debug("NO Armor")
            return x, y, z
        for i in range(0,len(lightBarList)-1):
            for j in range(i+1,len(lightBarList)):
                x0, y0, l0, s0, a0 = lightBarList[i][0:5]#s0 s1是灯条长 l0 l1是灯条宽
                x1, y1, l1, s1, a1 = lightBarList[j][0:5]

                angleDiff = abs(a0 - a1)                           #灯条角度差
                areaDiff = abs(s0*l0 - s1*l1)                      #灯条面积差
                areaRatio = (s0*l0)/(s1*l1)                        #灯条面积比
                arealongRatio = s0/s1                              #灯条长长比
                areawidthRatio = l0/l1                             #灯条宽宽比
                xCenter = (x0+x1)/2                                #装甲板中心x值
                yCenter = (y0+y1)/2                                #装甲板中心y值     
                ylength = (s0+s1)/2                                #装甲板纵向长度
                xlength = self.EuclideanDistance([x0,y0],[x1,y1])  #装甲板横向长度
                armorProp = xlength/ylength                        #装甲板长宽比
                armorArea = xlength*ylength                        #装甲板面积
                angle = abs((y0-y1)/(x0-x1))                       #装甲板角度
                yixaingangleDiff = 3

                if((s0*l0 + s1*l1) < 500):
                    self.angleDiff = 8
                    yixaingangleDiff = 5
                #区分左右装甲板，算出灯板角度
                if x0 > x1:
                    angle = angle*180/math.pi
                else:
                    angle = -angle*180/math.pi
                #灯条长长过大过小不要
                if arealongRatio < self.minarealongRatio or arealongRatio > self.maxarealongRatio:
                    # log.print_debug("arealongRatio = "+str(arealongRatio)+str(self.minarealongRatio)+str(self.maxarealongRatio))
                    continue
                #灯条宽宽比过大过小不要
                if areawidthRatio < 0.35 or areawidthRatio > (2.86):
                    # log.print_debug("areawidthRatio = "+str(areawidthRatio))
                    continue
                #灯条角度差过大不要
                if angleDiff > self.angleDiff and angleDiff < 180-self.angleDiff:
                    # log.print_debug("angleDiff = "+str(angleDiff))
                    continue      
                #灯条yixiang角度差过大不要
                if angleDiff > 90 and angleDiff < 180 - yixaingangleDiff:
                    # log.print_debug("yixaingangleDiff = "+str(angleDiff))
                    continue
                #灯条面积比太大不要
                if areaRatio > 3.3 or areaRatio < (0.3):
                    # log.print_debug("areaRatio = "+str(areaRatio))
                    continue
                #灯条中心太近不要
                if abs(x0-x1) < 15:
                    # log.print_debug( "abs(x0-x1) < 15")
                    continue
                #灯条面积差太大不要
                if areaDiff > self.lightBarAreaDiff:
                    # log.print_debug("areaDiff = "+str(areaDiff))
                    continue
                #装甲板角度过偏不要
                if abs(angle) > self.armorAngleMin:
                    # log.print_debug(" abs(angle) > = "+str(abs(angle)))
                    continue
                #装甲板高度太小不要
                if ylength < 10:
                    # log.print_debug(" ylength < 10")
                    continue
                #装甲板面积太大或太小不要
                if armorArea < self.minarmorArea or armorArea > self.maxarmorArea:
                    # log.print_debug("armorArea < self.minarmorArea"+str(armorArea))
                    continue
				#两个灯条y轴高度差过大不要
                if(abs(y0-y1)>ylength*1.2):
                    # log.print_debug("两个灯条y轴高度差过大不要"+str(abs(y0-y1)/ylength))
                    continue
				# #装甲板长宽比太大或太小不要
                # if armorProp >= self.minarmorProp and armorProp <= self.maxarmorProp:
                #     kao = 1
                # elif armorProp >= self.minBigarmorProp and armorProp <= self.maxBigarmorProp:    
                #     kao = 2
                # else:
                #     continue    
                z = self.GetArmorDistance(s0,s1)
                z_distance = z
                realCenter_list.append([xCenter,yCenter,xlength,ylength,angle,z_distance])

            #挑选距离屏幕中心最近的装甲板
            temp_distence = -1        
            if len(realCenter_list) > 0: 
                for realCenter in realCenter_list:
                    armor_imgcenter_distence = self.EuclideanDistance([realCenter[0],realCenter[1]],[self.xCenter,self.yCenter])
                    if temp_distence > armor_imgcenter_distence or temp_distence == -1:
                        temp_distence = armor_imgcenter_distence
                        x = realCenter[0]
                        y = realCenter[1]
                        z = realCenter[5]
            else :
                # log.print_debug("no build Armor")
                pass
            if temp_distence == -1:
                x,y,z = -1,-1,-1

        if self.debug:
            #如果开启了debug模式，向类变量更新值
            self.realCenter_list = realCenter_list
            self.x = x
            self.y = y
        return x, y, z


    def GetArmorDistance(self,dShortside,dLongside):
        #单目视觉测距，简单的根据装甲板最长边的对应像素去计算
        #注意，调参时需要在hsv确定后再调整测距
        height = dShortside + dLongside
        distance = self.kh/height
        return distance

    def TrackerBar_create(self):
        #创建滑动条
        if self.debug:
            if self.getvar_label == False:
                self.getvar_label = True
            # Lower range colour sliders.
            cv2.createTrackbar('lowHue', 'colorTest', self.lowHue, 255, self.nothing)
            cv2.createTrackbar('lowSat', 'colorTest', self.lowSat, 255, self.nothing)
            cv2.createTrackbar('lowVal', 'colorTest', self.lowVal, 255, self.nothing)    
            # Higher range colour sliders.
            cv2.createTrackbar('highHue', 'colorTest', self.highHue, 255, self.nothing)
            cv2.createTrackbar('highSat', 'colorTest', self.highSat, 255, self.nothing)
            cv2.createTrackbar('highVal', 'colorTest', self.highVal, 255, self.nothing)
            
            cv2.createTrackbar('lightBarAngleMin0.0', 'armorTest', int(self.minAngleError*10), 3600, self.nothing)
            cv2.createTrackbar('lightBarAngleMax0.0', 'armorTest', int(self.maxAngleError*10), 3600, self.nothing)
            cv2.createTrackbar('lightBarAreaMin', 'armorTest', int(self.minlighterarea), 255, self.nothing)
            cv2.createTrackbar('lightBarAreaMax', 'armorTest', int(self.maxlighterarea), 10000, self.nothing)
            cv2.createTrackbar('lightBarL/WMin0.00', 'armorTest', int(self.minlighterProp*100), 500, self.nothing)
            cv2.createTrackbar('lightBarL/WMax0.00', 'armorTest', int(self.maxlighterProp*100), 3000, self.nothing)
            cv2.createTrackbar('lightBarAngleErr0.0', 'armorTest', int(self.angleDiff*10), 3600, self.nothing)
            cv2.createTrackbar('lightBarAreaErr', 'armorTest', int(self.lightBarAreaDiff), 10000, self.nothing)
            cv2.createTrackbar('lightBarL/LMax0.00', 'armorTest', int(self.maxarealongRatio*100), 300, self.nothing)
            cv2.createTrackbar('lightBarL/LMin0.00', 'armorTest', int(self.minarealongRatio*100), 100, self.nothing)
            cv2.createTrackbar('armorAngleMin0.0', 'armorTest', int(self.armorAngleMin*10), 3600, self.nothing)
            cv2.createTrackbar('armorAreaMin','armorTest', int(self.minarmorArea), 5000, self.nothing)
            cv2.createTrackbar('armorAreaMax', 'armorTest', int(self.maxarmorArea), 100000, self.nothing)
            cv2.createTrackbar('armorL/WMin0.00', 'armorTest', int(self.minarmorProp*100), 255, self.nothing)
            cv2.createTrackbar('armorL/WMax0.00', 'armorTest', int(self.maxarmorProp*100), 600, self.nothing)
            cv2.createTrackbar('大装甲板L/WMin0.00', 'armorTest', int(self.minBigarmorProp*100), 300, self.nothing)
            cv2.createTrackbar('大装甲板L/WMax0.00', 'armorTest', int(self.maxBigarmorProp*100), 600, self.nothing)
            cv2.createTrackbar('distance', 'armorTest', int(self.kh), 40000, self.nothing)


    def updata_argument(self):
        #根据debug更新参数
        if self.debug and self.getvar_label:
            lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
            lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
            lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
            highHue = cv2.getTrackbarPos('highHue', 'colorTest')
            highSat = cv2.getTrackbarPos('highSat', 'colorTest')
            highVal = cv2.getTrackbarPos('highVal', 'colorTest')
            hsvPara_high = [highHue,highSat,highVal]
            hsvPara_low = [lowHue,lowSat,lowVal]
            self.hsvPara = np.array([hsvPara_low,hsvPara_high])
            self.minAngleError = float(cv2.getTrackbarPos('lightBarAngleMin0.0', 'armorTest'))/10
            self.maxAngleError = float(cv2.getTrackbarPos('lightBarAngleMax0.0', 'armorTest'))/10
            self.minlighterarea = cv2.getTrackbarPos('lightBarAreaMin', 'armorTest')
            self.maxlighterarea = cv2.getTrackbarPos('lightBarAreaMax', 'armorTest')
            self.minlighterProp = float(cv2.getTrackbarPos('lightBarL/WMin0.00', 'armorTest'))/100
            self.maxlighterProp = float(cv2.getTrackbarPos('lightBarL/WMax0.00', 'armorTest'))/100
            self.angleDiff = float(cv2.getTrackbarPos('lightBarAngleErr0.0', 'armorTest'))/10
            self.lightBarAreaDiff = cv2.getTrackbarPos('lightBarAreaErr', 'armorTest')
            self.maxarealongRatio = float(cv2.getTrackbarPos('lightBarL/LMax0.00', 'armorTest'))/100
            self.minarealongRatio = float(cv2.getTrackbarPos('lightBarL/LMin0.00', 'armorTest'))/100
            self.armorAngleMin = float(cv2.getTrackbarPos('armorAngleMin0.0', 'armorTest'))/10
            self.minarmorArea = cv2.getTrackbarPos('armorAreaMin', 'armorTest')
            self.maxarmorArea = cv2.getTrackbarPos('armorAreaMax', 'armorTest')
            self.minarmorProp = float(cv2.getTrackbarPos('armorL/WMin0.00', 'armorTest'))/100
            self.maxarmorProp = float(cv2.getTrackbarPos('armorL/WMax0.00', 'armorTest'))/100
            self.minBigarmorProp = float(cv2.getTrackbarPos('大装甲板L/WMin0.00', 'armorTest'))/100
            self.maxBigarmorProp = float(cv2.getTrackbarPos('大装甲板L/WMax0.00', 'armorTest'))/100
            self.kh = float(cv2.getTrackbarPos('distance', 'armorTest'))

    
    def update_json(self):
        temp_list = [
            self.minAngleError ,\
            self.maxAngleError ,\
            self.minlighterarea ,\
            self.maxlighterarea ,\
            self.minlighterProp ,\
            self.maxlighterProp ,\
            self.angleDiff ,\
            self.lightBarAreaDiff ,\
            self.maxarealongRatio ,\
            self.minarealongRatio ,\
            self.armorAngleMin ,\
            self.minarmorArea ,\
            self.maxarmorArea ,\
            self.minarmorProp ,\
            self.maxarmorProp ,\
            self.minBigarmorProp ,\
            self.maxBigarmorProp ,\
            self.kh
        ]
        #这里用于校验，保证序列化写入json前的数字是正常的，防止程序在debug时出现bug，导致参数丢失
        if temp_list.count(0.) > 5 or temp_list.count(-1) > 0 or \
            self.hsvPara[1][1] < 250 or self.hsvPara[1][2] < 250:
            pass
        else:
            with open(self.path,'r',encoding = 'utf-8') as load_f:
                load_dict = json.load(load_f,strict=False)
                load_dict["ArmorFind"]["minAngleError"] = self.minAngleError 
                load_dict["ArmorFind"]["maxAngleError"] = self.maxAngleError 
                load_dict["ArmorFind"]["minlighterarea"] = self.minlighterarea 
                load_dict["ArmorFind"]["maxlighterarea"] = self.maxlighterarea
                load_dict["ArmorFind"]["minlighterProp"] = self.minlighterProp
                load_dict["ArmorFind"]["maxlighterProp"] = self.maxlighterProp
                load_dict["ArmorFind"]["angleDiff"] = self.angleDiff
                load_dict["ArmorFind"]["lightBarAreaDiff"] = self.lightBarAreaDiff
                load_dict["ArmorFind"]["maxarealongRatio"] = self.maxarealongRatio
                load_dict["ArmorFind"]["minarealongRatio"] = self.minarealongRatio
                load_dict["ArmorFind"]["armorAngleMin"] = self.armorAngleMin
                load_dict["ArmorFind"]["minarmorArea"] = self.minarmorArea
                load_dict["ArmorFind"]["maxarmorArea"] = self.maxarmorArea
                load_dict["ArmorFind"]["minarmorProp"] = self.minarmorProp
                load_dict["ArmorFind"]["maxarmorProp"] = self.maxarmorProp
                load_dict["ArmorFind"]["minBigarmorProp"] = self.minBigarmorProp
                load_dict["ArmorFind"]["maxBigarmorProp"] = self.maxBigarmorProp
                load_dict["ArmorFind"]["kh"] = self.kh
                if self.color == 0:
                    load_dict["ImageProcess_red"]["hsvPara_high"] = [int(x) for x in list(self.hsvPara[1])]
                    load_dict["ImageProcess_red"]["hsvPara_low"] = [int(x) for x in list(self.hsvPara[0])]
                elif self.color == 1:
                    load_dict["ImageProcess_blue"]["hsvPara_high"] = [int(x) for x in list(self.hsvPara[1])]
                    load_dict["ImageProcess_blue"]["hsvPara_low"] = [int(x) for x in list(self.hsvPara[0])]
                dump_dict = load_dict
            with open(self.path,'w',encoding = 'utf-8') as load_f:
                json.dump(dump_dict,load_f,indent=4,ensure_ascii=False)


    def EuclideanDistance(self,c,c0):
        #欧式距离计算
        return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)

    def nothing(self,*arg):
        #空的回调函数
        pass

    def get_debug_frame(self):
        #处理并返回显示所需要的debug图像
        lightBarList = self.lightBarList
        realCenter_list = self.realCenter_list
        x,y = self.x, self.y
        frame_debug = self.frame_debug
        mask = self.mask
        for i in range(len(lightBarList)):
            log.print_debug(lightBarList[i])
            cv2.ellipse(frame_debug,(int(lightBarList[i][0]),int(lightBarList[i][1])),(int(lightBarList[i][2]/2),int(lightBarList[i][3]/2)),lightBarList[i][4],0,360,(0,255,0),2,8)
        for i in range(len(realCenter_list)):
            cv2.ellipse(frame_debug,(int(realCenter_list[i][0]),int(realCenter_list[i][1])),(int(realCenter_list[i][2]/2),int(realCenter_list[i][3]/2)),realCenter_list[i][4],0,360,(255,0,255),3,2)
            cv2.circle(frame_debug,(int(realCenter_list[i][0]),int(realCenter_list[i][1])),5,(0,255,0),-1)  
            cv2.circle(frame_debug,(int(x),int(y)),10,(255,255,255),-1)    
        return frame_debug,mask
