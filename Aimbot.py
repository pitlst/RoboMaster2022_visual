import cv2
import numpy as np
import math
import json
import shutil
import time
import os
from target_tracked import Tracker

class GetArmor:

    def __init__(self,color,mode):
        #变量初始化
        self.tracker = Tracker()
        self.mode = mode
        self.color = color
        self.read_aimbot()
        self.lastarmor_number_max = 21
        self.lastarmor_number = -1
        self.lastArmorCenter = [-1,-1,-1]
        self.lastframediff = []
        self.TrackerBar_label = 1
        self.x = 0
        self.y = 0
        self.z = 0
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
            self.minarea = load_dict["ArmorFind"]["minarea"]
            self.maxarea = load_dict["ArmorFind"]["maxarea"]
            self.minAngleError = load_dict["ArmorFind"]["minAngleError"]
            self.maxAngleError = load_dict["ArmorFind"]["maxAngleError"]
            self.minlighterarea = load_dict["ArmorFind"]["minlighterarea"]
            self.maxlighterarea = load_dict["ArmorFind"]["maxlighterarea"]
            self.minLongSide = load_dict["ArmorFind"]["minLongSide"]
            self.ratioMin = load_dict["ArmorFind"]["ratioMin"]
            self.ratioMax = load_dict["ArmorFind"]["ratioMax"]
            self.lsRatioMin = load_dict["ArmorFind"]["lsRatioMin"]
            self.lsRatioMax = load_dict["ArmorFind"]["lsRatioMax"]
            self.yDisMax = load_dict["ArmorFind"]["yDisMax"]
            self.angleErrMax = load_dict["ArmorFind"]["angleErrMax"]
            self.maxRatioXY = load_dict["ArmorFind"]["maxRatioXY"]
            self.minRatioXY = load_dict["ArmorFind"]["minRatioXY"]
            self.minCenterDisRatio = load_dict["ArmorFind"]["minCenterDisRatio"]
            self.maxCenterDisRatio = load_dict["ArmorFind"]["maxCenterDisRatio"]
            self.maxAlphaAg = load_dict["ArmorFind"]["maxAlphaAg"]
            self.kh = load_dict["ArmorFind"]["range_kh"]
            self.maxwhRatio = load_dict["ArmorFind"]["maxwhRatio"]
            self.minwhRatio = load_dict["ArmorFind"]["minwhRatio"]
        
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
        #根据debug参数确定是否开启卡尔曼滤波
        if self.kalmanfilter_enable:
            x,y,z = self.tracker.update(lightBarList)
        else:
            x,y,z = self.CombineLightBar(lightBarList)#非卡尔曼滤波，将灯条拼凑成装甲板

        #根据debug参数确定
        if self.show_debug:
            print('xyz:',int(x),int(y),int(z))
            if self.kalmanfilter_enable:
                cv2.circle(self.frame_debug,(int(x),int(y)),5,(255,255,255),-1)
            cv2.imshow('frame_debug',self.frame_debug)
            cv2.imshow('mask',mask)
            if time.time()-self.t0 > 1:
                self.updata_argument()
                self.update_json()
                self.t0 = time.time()
        #print(int(x),int(y),int(z))
        return x,y,z
    

    def HSV_Process(self,frame):
        #二值化图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(frame, self.hsvPara[0], self.hsvPara[1])
        return mask


    def GetLightBar(self,mask):
        lightBarList = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)#这里为opencv3的写法，使用opencv4时只有两个回调参数。
        for contour in range(len(contours)):
            center, size, angle = cv2.minAreaRect(contours[contour])
            vertices = cv2.boxPoints((center, size, angle)) #得到最小面积矩形的四个顶点
            if size[0]<size[1]:
                vector = center-(vertices[0]+vertices[3])/2 #得到长边和中心点垂直短边向量
                longSide = size[1]
                shortSide = size[0]
            else:
                vector = center-(vertices[0]+vertices[1])/2
                longSide = size[0]
                shortSide = size[1]
            angleHori = int(math.atan2(vector[1], vector[0]) * 180/math.pi) #计算中心点垂直短边向量和水平轴夹角
            area = longSide*shortSide
            if self.show_debug:
                if  abs(angleHori) < self.minAngleError or abs(angleHori) > self.maxAngleError:
                    #print(str(contour)+':angleHori错误:', abs(angleHori))
                    nowLightBar = False
                elif longSide < self.minLongSide:
                    #print(str(contour)+':longSide错误:',longSide)
                    nowLightBar = False
                elif longSide < self.ratioMin*shortSide or longSide > self.ratioMax*shortSide:
                    #print(str(contour)+':ratio错误:',longSide/(shortSide+0.0001))
                    nowLightBar = False
                elif area > self.maxlighterarea or area < self.minlighterarea:
                    #print(str(contour)+':area错误:',area)
                    nowLightBar = False
                else:
                    nowLightBar = True
                    #print('area:',area)
                    #print('ratio:',longSide/(shortSide+0.0001)) 
                    #print('longSide:',longSide)
                    #print('angleHori:', abs(angleHori))
                if nowLightBar:
                    lightBarList.append([center[0],center[1],longSide,shortSide,angleHori,area])
            else:
                if  abs(angleHori) < self.minAngleError or abs(angleHori) > self.maxAngleError:
                    nowLightBar = False
                elif longSide < self.minLongSide:
                    nowLightBar = False
                elif longSide < self.ratioMin*shortSide or longSide > self.ratioMax*shortSide:
                    nowLightBar = False
                elif area > self.maxlighterarea or area < self.minlighterarea:
                    nowLightBar = False
                else:
                    nowLightBar = True
                if nowLightBar:
                    lightBarList.append([center[0],center[1],longSide,shortSide,angleHori])
        if self.show_debug:
            for i in range(len(lightBarList)):
                cv2.putText(self.frame_debug, 'angleHori:'+str(lightBarList[i][4]), (5,5+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                cv2.putText(self.frame_debug, 'longSide:'+str(lightBarList[i][2]), (5,25+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                cv2.putText(self.frame_debug, 'ratio:'+str(lightBarList[i][2]/(lightBarList[i][3]+0.0001)), (5,45+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                cv2.putText(self.frame_debug, 'lighterarea:'+str(lightBarList[i][5]), (5,65+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                cv2.ellipse(self.frame_debug,(int(lightBarList[i][0]),int(lightBarList[i][1])),(int(lightBarList[i][2]/2),int(lightBarList[i][3]/2)),lightBarList[i][4],0,360,(255,255,255),3,2)
        return lightBarList


    def CombineLightBar(self,lightBarList):
        
        realCenter_list = []
        realCenter = [-1, -1]
        x,y,z = -1,-1,-1
        if len(lightBarList) <  2:
            x,y,z = -1,-1,-1
        else:
            for i in range(0,len(lightBarList)-1):
                for j in range(i+1,len(lightBarList)):
                    x0, y0, l0, s0, a0 = lightBarList[i][0:5]
                    x1, y1, l1, s1, a1 = lightBarList[j][0:5]

                    centerDis = self.EuclideanDistance([x0,y0],[x1,y1]) #centerDis为两灯条中心距离
                    angleErr = abs(abs(a0)-abs(a1)) #angleErr为两灯条角度之差。用于筛选
                    #print('a0:'+str(a0)+'     '+'a1:'+str(a1))

                    xCenter = (x0+x1)/2  #装甲板中心x值
                    yCenter = (y0+y1)/2  #装甲板中心y值         
                    lErr = max(l1,l0)/(min(l1,l0)+0.00001)#lErr是两灯条长度之比，用于筛选
                    xDis = abs(x0-x1)    #装甲板横向长度
                    yDis = abs(y0-y1)    #灯条中心纵向差值
                    ylength = (l0+l1)/2  #装甲板纵向长度
                    lsRatio = centerDis/ylength   #装甲板中心十字的比值
                    alpha = math.asin(yDis/centerDis)*180/math.pi   #装甲板角度
                    area = centerDis*ylength      #装甲板面积
                    whRatio = centerDis/(max(l0,l1)+0.00001)         #装甲板长宽比
                    z = self.GetArmorDistance(min(l1,l0),max(l1,l0))
                    if self.show_debug:
                        if area*z/100 < self.minarea or area*z/100 > self.maxarea:
                            print(str(i)+' '+str(j)+' 装甲板面积错误:',area*z/100)
                            nowArmor = False
                        elif lsRatio < self.lsRatioMin or lsRatio > self.lsRatioMax:
                            print(str(i)+' '+str(j)+' 装甲板中心十字比lsRatio错误:',lsRatio)
                            nowArmor = False
                        elif yDis/max(l0, l1)*11 > self.yDisMax:
                            print(str(i)+' '+str(j)+' 灯条中心纵向差值yDisMax错误:',yDis/max(l0, l1)*11)
                            nowArmor = False
                        elif angleErr*max(l0,l1)/100 > self.angleErrMax:
                            print(str(i)+' '+str(j)+' 灯条角度差angleErrMax错误:',angleErr*max(l0,l1)/100)
                            nowArmor = False
                        elif lErr > self.maxRatioXY or lErr < self.minRatioXY:
                            print(str(i)+' '+str(j)+' 两灯条长度之比maxRatioXY错误:',lErr)
                            nowArmor = False
                        elif centerDis < self.minCenterDisRatio*max(l0,l1)/z*100 or centerDis > self.maxCenterDisRatio*max(l0,l1)/z*100:
                            print(str(i)+' '+str(j)+' 两灯条中心距centerDis错误:',centerDis)
                            nowArmor = False
                        elif alpha > self.maxAlphaAg:
                            print(str(i)+' '+str(j)+' 装甲板角度alpha错误:',alpha)
                            nowArmor = False
                        elif whRatio > self.maxwhRatio or whRatio < self.minwhRatio:
                            print(str(i)+' '+str(j)+' 装甲板长宽比whRatio错误:',whRatio)
                            nowArmor = False
                        else:
                            nowArmor = True
                            # print('area:',area*z/100)
                            # print('lsRatio:',lsRatio)
                            # print('yDis:',yDis/max(l0, l1)*11)
                            # print('angleErr:',angleErr*max(l0,l1)/100)
                            # print('lErr:',lErr)
                            # print('centerDis:',centerDis/max(l0,l1))
                            # print('alpha:',alpha)
                            # print('whRatio:',whRatio)
                        if nowArmor:
                            realCenter_list.append([xCenter,yCenter,z,xDis,lsRatio,alpha,ylength,centerDis,lErr,angleErr*max(l0,l1)/100,yDis/max(l0, l1)*11,lsRatio,area*z,i,j])
                    else:
                        if area*z/100 < self.minarea or area*z/100 > self.maxarea:
                            nowArmor = False
                        elif lsRatio < self.lsRatioMin or lsRatio > self.lsRatioMax:
                            nowArmor = False
                        elif yDis/max(l0, l1)*11 > self.yDisMax:
                            nowArmor = False
                        elif angleErr*max(l0,l1)/100 > self.angleErrMax:
                            nowArmor = False
                        elif lErr > self.maxRatioXY or lErr < self.minRatioXY:
                            nowArmor = False
                        elif centerDis < self.minCenterDisRatio*max(l0,l1)/z*100 or centerDis > self.maxCenterDisRatio*max(l0,l1)/z*100:
                            nowArmor = False
                        elif alpha > self.maxAlphaAg:
                            nowArmor = False
                        elif whRatio > self.maxwhRatio or whRatio < self.minwhRatio:
                            nowArmor = False
                        else:
                            nowArmor = True
                        if nowArmor:
                            realCenter_list.append([xCenter,yCenter,z,area])
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
                    cv2.putText(self.frame_debug, '装甲板面积错误:'+str(realCenter_list[i][11]), (5,85+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                    cv2.putText(self.frame_debug, '装甲板长宽比lsRatio错误:'+str(realCenter_list[i][10]), (5,105+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                    cv2.putText(self.frame_debug, '灯条中心纵向差值yDisMax错误:'+str(realCenter_list[i][9]), (5,125+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                    cv2.putText(self.frame_debug, '灯条角度差angleErrMax错误:'+str(realCenter_list[i][8]), (5,145+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                    cv2.putText(self.frame_debug, '两灯条长度之比maxRatioXY错误:'+str(realCenter_list[i][7]), (5,165+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                    cv2.putText(self.frame_debug, '两灯条中心距centerDis错误:'+str(realCenter_list[i][6]), (5,185+int(i)*20), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                    cv2.putText(self.frame_debug, '装甲板角度alpha错误:'+str(alpha), (5,205), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (255,255,255) ,2)
                    cv2.ellipse(self.frame_debug,(int(realCenter_list[i][0]),int(realCenter_list[i][1])),(int(realCenter_list[i][3]/2),int(realCenter_list[i][6]/2)),realCenter_list[i][5],0,360,(255,0,255),3,2)
                    cv2.circle(self.frame_debug,(int(realCenter_list[i][0]),int(realCenter_list[i][1])),5,(0,255,0),-1)      
                #cv2.ellipse(self.frame_debug,(int(temp_realCenter[0]),int(temp_realCenter[1])),(int(temp_realCenter[3]/2),int(temp_realCenter[4]/2)),temp_realCenter[5],0,360,(255,0,255),3,2)
                #cv2.circle(self.frame_debug,(int(temp_realCenter[0]),int(temp_realCenter[1])),5,(0,255,0),-1)                         
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
            cv2.createTrackbar('minAngleError', 'LBTest', self.minAngleError, 360, self.nothing)
            cv2.createTrackbar('maxAngleError', 'LBTest', self.maxAngleError, 360, self.nothing)
            cv2.createTrackbar('minlighterarea', 'LBTest', self.minlighterarea, 1000, self.nothing)
            cv2.createTrackbar('maxlighterarea', 'LBTest', self.maxlighterarea, 30000, self.nothing)
            cv2.createTrackbar('minLongSide', 'LBTest', self.minLongSide, 80, self.nothing)
            cv2.createTrackbar('ratioMin', 'LBTest', int(self.ratioMin*10), 90, self.nothing)
            cv2.createTrackbar('ratioMax', 'LBTest', int(self.ratioMax*10), 300, self.nothing)
            cv2.createTrackbar('yDisMax', 'armorTest', int(self.yDisMax*10), 250, self.nothing)
            cv2.createTrackbar('angleErrMax', 'armorTest', int(self.angleErrMax*10), 300, self.nothing)
            cv2.createTrackbar('maxAlphaAg', 'armorTest', int(self.maxAlphaAg*10), 250, self.nothing)
            cv2.createTrackbar('maxRatioXY', 'armorTest', int(self.maxRatioXY*10), 90, self.nothing)
            cv2.createTrackbar('minRatioXY','armorTest', int(self.minRatioXY*10), 90, self.nothing)
            cv2.createTrackbar('minCenterDisRatio', 'armorTest', int(self.minCenterDisRatio*10), 50, self.nothing)
            cv2.createTrackbar('maxCenterDisRatio', 'armorTest', int(self.maxCenterDisRatio*10), 400, self.nothing)
            cv2.createTrackbar('lsRatioMin', 'armorTest', int(self.lsRatioMin*10), 70, self.nothing)
            cv2.createTrackbar('lsRatioMax', 'armorTest', int(self.lsRatioMax*10), 250, self.nothing)
            cv2.createTrackbar('minarea', 'armorTest', self.minarea, 10000, self.nothing)
            cv2.createTrackbar('maxarea', 'armorTest', self.maxarea, 90000, self.nothing)
            cv2.createTrackbar('maxwhRatio', 'armorTest', int(self.maxwhRatio*10), 90, self.nothing)
            cv2.createTrackbar('minwhRatio', 'armorTest', self.minwhRatio, 99, self.nothing)
            cv2.createTrackbar('kh', 'armorTest', int(self.kh), 40000, self.nothing)


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
            self.minAngleError = cv2.getTrackbarPos('minAngleError', 'LBTest')
            self.maxAngleError = cv2.getTrackbarPos('maxAngleError', 'LBTest')
            self.minlighterarea = cv2.getTrackbarPos('minlighterarea', 'LBTest')
            self.maxlighterarea = cv2.getTrackbarPos('maxlighterarea', 'LBTest')
            self.minLongSide = cv2.getTrackbarPos('minLongSide', 'LBTest')
            self.ratioMin = float(cv2.getTrackbarPos('ratioMin', 'LBTest'))/10
            self.ratioMax = float(cv2.getTrackbarPos('ratioMax', 'LBTest'))/10
            self.yDisMax = float(cv2.getTrackbarPos('yDisMax', 'armorTest'))/10
            self.angleErrMax = float(cv2.getTrackbarPos('angleErrMax', 'armorTest'))/10
            self.maxAlphaAg = float(cv2.getTrackbarPos('maxAlphaAg', 'armorTest'))/10
            self.maxRatioXY = float(cv2.getTrackbarPos('maxRatioXY', 'armorTest'))/10
            self.minRatioXY = float(cv2.getTrackbarPos('minRatioXY', 'armorTest'))/10
            self.minCenterDisRatio = float(cv2.getTrackbarPos('minCenterDisRatio', 'armorTest'))/10
            self.maxCenterDisRatio = float(cv2.getTrackbarPos('maxCenterDisRatio', 'armorTest'))/10
            self.lsRatioMin = float(cv2.getTrackbarPos('lsRatioMin', 'armorTest'))/10
            self.lsRatioMax = float(cv2.getTrackbarPos('lsRatioMax', 'armorTest'))/10
            self.minarea = cv2.getTrackbarPos('minarea', 'armorTest')
            self.maxarea = cv2.getTrackbarPos('maxarea', 'armorTest')
            self.maxwhRatio = float(cv2.getTrackbarPos('maxwhRatio', 'armorTest'))/10
            self.minwhRatio = cv2.getTrackbarPos('minwhRatio', 'armorTest')
            self.kh = cv2.getTrackbarPos('kh', 'armorTest')
    
    def update_json(self):
        temp_list = [
            self.minAngleError ,\
            self.maxAngleError ,\
            self.minlighterarea ,\
            self.maxlighterarea ,\
            self.minLongSide ,\
            self.ratioMin ,\
            self.ratioMax ,\
            self.yDisMax ,\
            self.angleErrMax ,\
            self.maxAlphaAg ,\
            self.maxRatioXY ,\
            self.minRatioXY ,\
            self.minCenterDisRatio ,\
            self.maxCenterDisRatio ,\
            self.lsRatioMin ,\
            self.lsRatioMax ,\
            self.minarea ,\
            self.maxarea ,\
            self.maxwhRatio ,\
            self.minwhRatio ,\
            self.kh
        ]
        #这里用于校验，保证序列化写入json前的数字是正常的，防止程序在debug时出现bug，导致参数丢失
        if temp_list.count(0.) > 5 or temp_list.count(-1) > 0 or \
            self.hsvPara[1][1] < 250 or self.hsvPara[1][2] < 250:
            pass
        else:
            with open(self.path,'r',encoding = 'utf-8') as load_f:
                load_dict = json.load(load_f,strict=False)
                load_dict["ArmorFind"]["minarea"] = self.minarea
                load_dict["ArmorFind"]["maxarea"] = self.maxarea
                load_dict["ArmorFind"]["minAngleError"] = self.minAngleError
                load_dict["ArmorFind"]["maxAngleError"] = self.maxAngleError
                load_dict["ArmorFind"]["minlighterarea"] = self.minlighterarea
                load_dict["ArmorFind"]["maxlighterarea"] = self.maxlighterarea
                load_dict["ArmorFind"]["minLongSide"] = self.minLongSide
                load_dict["ArmorFind"]["ratioMin"] = self.ratioMin
                load_dict["ArmorFind"]["ratioMax"] = self.ratioMax
                load_dict["ArmorFind"]["lsRatioMin"] = self.lsRatioMin
                load_dict["ArmorFind"]["lsRatioMax"] = self.lsRatioMax
                load_dict["ArmorFind"]["yDisMax"] = self.yDisMax
                load_dict["ArmorFind"]["angleErrMax"] = self.angleErrMax
                load_dict["ArmorFind"]["maxRatioXY"] = self.maxRatioXY
                load_dict["ArmorFind"]["minRatioXY"] = self.minRatioXY
                load_dict["ArmorFind"]["minCenterDisRatio"] = self.minCenterDisRatio
                load_dict["ArmorFind"]["maxCenterDisRatio"] = self.maxCenterDisRatio
                load_dict["ArmorFind"]["maxAlphaAg"] = self.maxAlphaAg
                load_dict["ArmorFind"]["range_kh"] = self.kh
                load_dict["ArmorFind"]["maxwhRatio"]= self.maxwhRatio
                load_dict["ArmorFind"]["minwhRatio"] = self.minwhRatio
                if self.color == 0:
                    load_dict["ImageProcess_red"]["hsvPara_high"] = [int(x) for x in list(self.hsvPara[1])]
                    load_dict["ImageProcess_red"]["hsvPara_low"] = [int(x) for x in list(self.hsvPara[0])]
                elif self.color == 1:
                    load_dict["ImageProcess_blue"]["hsvPara_high"] = [int(x) for x in list(self.hsvPara[1])]
                    load_dict["ImageProcess_blue"]["hsvPara_low"] = [int(x) for x in list(self.hsvPara[0])]
                dump_dict = load_dict
            with open(self.path,'w',encoding = 'utf-8') as load_f:
               json.dump(dump_dict,load_f,indent=4,ensure_ascii=False)

