import cv2
import numpy as np
import math
import json
import shutil
import time
import os

class GetArmor:

    def __init__(self,color,mode):
        #变量初始化
        self.mode = mode
        self.color = color
        self.read_aimbot()
        if self.show_debug:
            self.t0 = time.time()

    def read_aimbot(self):
        #该函数用于读取自瞄参数
        if self.mode == 0:
            self.path = './armor_find.json'
            self.path_copy = './armor_find_copy.json'
        elif self.mode == 3:
            self.path = './sentry_find.json'
            self.path_copy = './sentry_find_copy.json'
        else:
            print('error: mode num')
            self.path = './armor_find.json'
            self.path_copy = './armor_find_copy.json'
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
                print('color error:unknow error')
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
            #测距
            self.kh = load_dict["ArmorFind"]["kh"]
        with open('./common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.__xCenter = load_dict["Aimbot"]["width"]/2
            self.__yCenter = load_dict["Aimbot"]["height"]/2
        with open('./debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.video_debug_set = load_dict["Debug"]["video_debug_set"]
            self.show_debug = load_dict["Debug"]["show_debug"]
            self.kalmanfilter_enable = load_dict["Debug"]["kalmanfilter_enable"]

    def GetArmor(self,frame):
        #识别装甲板主程序
        x, y, z = -1, -1, -1
        #根据debug参数拷贝原图像,更改图像来源格式
        if self.video_debug_set == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2BGR)
        if self.show_debug:
            self.frame_debug = frame.copy()
        
        mask = self.HSV_Process(frame)# 进行HSV图像处理
        lightBarList = self.GetLightBar(mask)# 进行获取灯条矩形列表
        x,y,z = self.CombineLightBar(lightBarList)#非卡尔曼滤波，将灯条拼凑成装甲板
        #print('xyz:',int(x),int(y),int(z))
        #根据debug参数确定
        if self.show_debug:
            if self.kalmanfilter_enable:
                cv2.circle(self.frame_debug,(int(x),int(y)),5,(255,255,255),-1)
            cv2.imshow('frame_debug',self.frame_debug)
            cv2.imshow('mask',mask)
            self.updata_argument()
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
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)#这里为opencv3的写法，使用opencv4时只有两个回调参数。
        for contour in range(len(contours)):
            if len(contours[contour]) > 5:
                center, size, angle = cv2.fitEllipse(contours[contour])
                if size[1] <= 0 or size[0] <= 0:
                    continue
                rectProp = size[0]/size[1]
                rectArea = size[0]*size[1]
                angleHori = abs(int(angle)%180)
                if  angleHori < self.minAngleError or angleHori > self.maxAngleError:
                    continue
                elif rectProp < self.minlighterProp or rectProp > self.maxlighterProp:
                    continue
                elif rectArea > self.maxlighterarea or rectArea < self.minlighterarea:
                    continue
                else:
                    if self.show_debug:
                        lightBarList.append([center[0],center[1],size[0],size[1],angle,rectProp,angleHori,rectArea])
                    else:
                        lightBarList.append([center[0],center[1],size[0],size[1],angle]) 
        if self.show_debug:
            for i in range(len(lightBarList)):
                cv2.ellipse(self.frame_debug,(int(lightBarList[i][0]),int(lightBarList[i][1])),(int(lightBarList[i][2]),int(lightBarList[i][3])),lightBarList[i][4],0,360,(0,255,0),2,8)
        return lightBarList


    def CombineLightBar(self,lightBarList):
        #筛选装甲板
        realCenter_list = []
        realCenter = [-1, -1]
        x,y,z = -1,-1,-1
        #灯条数量没有两个，构建不出装甲板
        if len(lightBarList) <  2:
            return x, y, z
        for i in range(0,len(lightBarList)-1):
            for j in range(i+1,len(lightBarList)):
                x0, y0, l0, s0, a0 = lightBarList[i][0:5]
                x1, y1, l1, s1, a1 = lightBarList[j][0:5]

                angleDiff = abs(a0 - a1)                           #灯条角度差
                areaDiff = abs(s0*l0 - s1*l1)                      #灯条面积差
                areaRatio = s0*l0/s1*l1                            #灯条面积比
                arealongRatio = l0/l1                              #灯条长长比
                xCenter = (x0+x1)/2                                #装甲板中心x值
                yCenter = (y0+y1)/2                                #装甲板中心y值     
                ylength = (l0+l1)/2                                #装甲板纵向长度
                xlenght = self.EuclideanDistance([x0,y0],[x1,y1])  #装甲板横向长度
                armorProp = xlenght/ylength                        #装甲板长宽比
                armorArea = xlenght*ylength                        #装甲板面积
                angle = abs((y0-y1)/(x0-x1))                       #装甲板角度
                #区分左右装甲板，算出灯板角度
                if x0 > x1:
                    angle = angle*180/math.pi
                else:
                    angle = -angle*180/math.pi
                #灯条长长过大过小不要
                if arealongRatio < self.minarealongRatio or arealongRatio > self.minarealongRatio:
                    continue
                #灯条角度差过大不要
                if angleDiff > self.angleDiff and angleDiff < 180-self.angleDiff:
                    continue
                #灯条面积比太大不要
                if areaRatio > 3 or areaRatio < 0.28:
                    continue
                #灯条中心太近不要
                if abs(x0-x1) < 15:
                    continue
                #灯条面积差太大不要
                if areaDiff > self.lightBarAreaDiff:
                    continue
                #装甲板角度过偏不要
                if abs(angle) > self.armorAngleMin:
                    continue
                #装甲板高度太小不要
                if ylength < 10:
                    continue
                #装甲板面积太大或太小不要
                if armorArea < self.minarmorArea or armorArea > self.maxarmorArea:
                    continue
                #装甲板长宽比太大或太小不要
                if armorProp < self.minarmorProp or armorProp > self.maxarmorProp:
                    continue
                z = self.GetArmorDistance(min(l1,l0),max(l1,l0))
                realCenter_list.append([xCenter,yCenter,z])
            #挑选距离屏幕中心最近的装甲板
            temp_distence = -1        
            if len(realCenter_list) > 0: 
                for realCenter in realCenter_list:
                    armor_imgcenter_distence = self.EuclideanDistance([xCenter,yCenter],[self.__xCenter,self.__yCenter])
                    if temp_distence > armor_imgcenter_distence or temp_distence == -1:
                        temp_distence = armor_imgcenter_distence
                        x = realCenter[0]
                        y = realCenter[1]
                        z = realCenter[2]
            if temp_distence == -1:
                x,y,z = -1,-1,-1

        if self.show_debug:
            if len(realCenter_list) > 0:
                for i in range(len(realCenter_list)):
                    cv2.ellipse(self.frame_debug,(int(realCenter_list[i][0]),int(realCenter_list[i][1])),(int(realCenter_list[i][3]/2),int(realCenter_list[i][6]/2)),realCenter_list[i][5],0,360,(255,0,255),3,2)
                    cv2.circle(self.frame_debug,(int(realCenter_list[i][0]),int(realCenter_list[i][1])),5,(0,255,0),-1)                      
        print("distance"+str(z))
        return x, y, z


    def GetArmorDistance(self,dShortside,dLongside):
        #单目视觉测距，简单的根据装甲板最长边的对应像素去计算
        #注意，调参时需要在hsv确定后再调整测距
        height = dShortside + dLongside
        distance = self.kh/height
        return distance

    def EuclideanDistance(self,c,c0):
        #欧式距离计算
        return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)

    def nothing(self,*arg):
        #空的回调函数
        pass

    def TrackerBar_create(self):
        #创建滑动条,并备份参数
        if os.path.isfile(self.path_copy) is False:
            shutil.copy(self.path,self.path_copy)
        if self.show_debug:
            # Lower range colour sliders.
            cv2.createTrackbar('lowHue', 'colorTest', self.lowHue, 255, self.nothing)
            cv2.createTrackbar('lowSat', 'colorTest', self.lowSat, 255, self.nothing)
            cv2.createTrackbar('lowVal', 'colorTest', self.lowVal, 255, self.nothing)    
            # Higher range colour sliders.
            cv2.createTrackbar('highHue', 'colorTest', self.highHue, 255, self.nothing)
            cv2.createTrackbar('highSat', 'colorTest', self.highSat, 255, self.nothing)
            cv2.createTrackbar('highVal', 'colorTest', self.highVal, 255, self.nothing)
            
            cv2.createTrackbar('灯条角度最小值0.0', 'armorTest', int(self.minAngleError*10), 3600, self.nothing)
            cv2.createTrackbar('灯条角度最大值0.0', 'armorTest', int(self.maxAngleError*10), 3600, self.nothing)
            cv2.createTrackbar('灯条面积最小值', 'armorTest', int(self.minlighterarea), 255, self.nothing)
            cv2.createTrackbar('灯条面积最大值', 'armorTest', int(self.maxlighterarea), 5000, self.nothing)
            cv2.createTrackbar('灯条长宽比最小值0.00', 'armorTest', int(self.minlighterProp*100), 500, self.nothing)
            cv2.createTrackbar('灯条长宽比最大值0.00', 'armorTest', int(self.maxlighterProp*100), 3000, self.nothing)
            cv2.createTrackbar('灯条角度差0.0', 'armorTest', int(self.angleDiff*10), 3600, self.nothing)
            cv2.createTrackbar('灯条面积差', 'armorTest', int(self.lightBarAreaDiff), 6000, self.nothing)
            cv2.createTrackbar('灯条长长比最大值0.00', 'armorTest', int(self.maxarealongRatio*100), 300, self.nothing)
            cv2.createTrackbar('灯条长长比最小值0.00', 'armorTest', int(self.minarealongRatio*100), 100, self.nothing)
            cv2.createTrackbar('装甲板角度最小值0.0', 'armorTest', int(self.armorAngleMin*10), 3600, self.nothing)
            cv2.createTrackbar('装甲板面积最小值','armorTest', int(self.minarmorArea), 5000, self.nothing)
            cv2.createTrackbar('装甲板面积最大值', 'armorTest', int(self.maxarmorArea), 60000, self.nothing)
            cv2.createTrackbar('装甲板长宽比最小值0.00', 'armorTest', int(self.minarmorProp*100), 255, self.nothing)
            cv2.createTrackbar('装甲板长宽比最大值0.00', 'armorTest', int(self.maxarmorProp*100), 600, self.nothing)
            cv2.createTrackbar('测距参数', 'armorTest', int(self.kh), 40000, self.nothing)
            # cv2.createTrackbar('小装甲板距离参数0.00', 'armorTest', int(self.lsRatioMax*100), 20000, self.nothing)
            # cv2.createTrackbar('大装甲板距离参数0.00', 'armorTest', int(self.minarea*100), 20000, self.nothing)


    def updata_argument(self):
        #根据debug更新参数
        if self.show_debug:
            lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
            lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
            lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
            highHue = cv2.getTrackbarPos('highHue', 'colorTest')
            highSat = cv2.getTrackbarPos('highSat', 'colorTest')
            highVal = cv2.getTrackbarPos('highVal', 'colorTest')
            hsvPara_high = [highHue,highSat,highVal]
            hsvPara_low = [lowHue,lowSat,lowVal]
            self.hsvPara = np.array([hsvPara_low,hsvPara_high])
            self.minAngleError = float(cv2.getTrackbarPos('灯条角度最小值0.0', 'armorTest'))/10
            print(cv2.getTrackbarPos('灯条角度最大值0.0', 'armorTest'))
            self.maxAngleError = float(cv2.getTrackbarPos('灯条角度最大值0.0', 'armorTest'))/10
            self.minlighterarea = cv2.getTrackbarPos('灯条面积最小值', 'armorTest')
            self.maxlighterarea = cv2.getTrackbarPos('灯条面积最大值', 'armorTest')
            self.minlighterProp = float(cv2.getTrackbarPos('灯条长宽比最小值0.00', 'armorTest'))/100
            self.maxlighterProp = float(cv2.getTrackbarPos('灯条长宽比最大值0.00', 'armorTest'))/100
            self.angleDiff = float(cv2.getTrackbarPos('灯条角度差0.0', 'armorTest'))/10
            self.lightBarAreaDiff = cv2.getTrackbarPos('灯条面积差', 'armorTest')
            self.maxarealongRatio = float(cv2.getTrackbarPos('灯条长长比最大值0.00', 'armorTest'))/100
            self.minarealongRatio = float(cv2.getTrackbarPos('灯条长长比最小值0.00', 'armorTest'))/100
            self.armorAngleMin = float(cv2.getTrackbarPos('装甲板角度最小值0.0', 'armorTest'))/10
            self.minarmorArea = cv2.getTrackbarPos('装甲板面积最小值', 'armorTest')
            self.maxarmorArea = cv2.getTrackbarPos('装甲板面积最大值', 'armorTest')
            self.minarmorProp = float(cv2.getTrackbarPos('装甲板长宽比最小值0.00', 'armorTest'))/100
            self.maxarmorProp = float(cv2.getTrackbarPos('装甲板长宽比最大值0.00', 'armorTest'))/100
            self.kh = float(cv2.getTrackbarPos('测距参数', 'armorTest'))

    
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

