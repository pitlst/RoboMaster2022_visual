import numpy as np
import math
import json


class AnglePredicted():
    def __init__(self):
        self.read_json()
        #帧差时间记录
        self.frame_diff_list = []
        #历史角度记录缓存
        self.history_angle_list = []
        #历史角度差记录缓存
        self.history_angle_diff_list = []
        #历史角速度记录器
        self.history_spd_list = []
        self.history_spd_list_2 = []
        #速度的历史记录器
        self.spd_list = []
        #历史大符半径的记录器
        self.history_hisdis_list = []

        self.last_delta_angle = None
        self.temp_delta_angle_list = []
        self.pass_num = 0



        #本次旋转方向
        self.detect = 0
        #程序执行次数计数
        self.count = 0
        self.count_2 = 0
        #历史最大角速度,最小角速度，角速度差值初始化
        self.spd_max = 0
        self.spd_min = 999
        self.spd_max_time = 0
        self.spd_min_time = 0
        #上次的加速度
        self.last_accelerate = 0
        #上次xy的记录
        self.x = 0
        self.y = 0
        #帧差时间初始化
        self.t0 = 0
        self.begin_time = None
        self.frame_diff = 0
        #是否检测初相位的标志位
        self.reset_big_energy = None
        #大幅的角度防抖
        self.last_angle = None
        #速度目标函数 spd = a*sin(w*t)+b
        #速度目标函数系数相位
        self.big_energy_t = None
        #打印日志文件的文件类
        self.f = open(self.path_log,'w')
    
    def read_json(self):
        #读取配置文件
        with open('./json/Energy_find_old.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.delta_angle_distance = load_dict["predicted"]["delta_angle_distance"]
            self.big_energy_a = load_dict["predicted"]["big_energy_a"]
            self.big_energy_w = load_dict["predicted"]["big_energy_w"]
            self.k = load_dict["predicted"]["big_energy_w"]
            self.path_log = load_dict["predicted"]["path_log"]
            self.predictAngle = load_dict["predicted"]["predictAngle"]
            self.delta_xy_distance = load_dict["predicted"]["delta_xy_distance"]
            self.max_spd = load_dict["predicted"]["max_spd"]
            self.history_angle_list_len_max = load_dict["predicted"]["history_angle_list_len_max"]
            self.reset_detect_dis = load_dict["predicted"]["reset_detect_dis"]
            self.history_hisdis_list_len_max = load_dict["predicted"]["history_hisdis_list_len_max"]
            self.delta_angle_max = load_dict["predicted"]["delta_angle_max"]
            self.history_angle_diff_list_len_max = load_dict["predicted"]["history_angle_diff_list_len_max"]
            self.frame_diff_list_len_max = load_dict["predicted"]["frame_diff_list_len_max"]
            self.big_energy_t_count_max = load_dict["predicted"]["big_energy_t_count_max"]
            self.history_spd_list_len_max = load_dict["predicted"]["history_spd_list_len_max"]
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.detect_debug = load_dict["Debug"]["detect_auto"]

    def NormalHit(self, Center, x, y, f_time,size):
        #预测前处理，将笛卡尔坐标转换为极坐标，对角度预测
        angle = -1
        hitX = -1
        hitY = -1
        hitDis = -1
        forecast_angle = 0
        label = 1
        self.t0 = f_time
        if self.begin_time == None:
            self.begin_time = self.t0
        if size == 1 or size == 4:
            self.detect = -1
        else:
            self.detect = 1


        if x == -1 or y == -1 or Center[0] == -1:
            vectorX = -1
            vectorY = -1
            label = 0
        else:
            vectorX = x - Center[0]
            vectorY = y - Center[1]

            #判断旋转方向
            #self.detect = self.judge_rotate_direct(vectorX,vectorY)
            #是否开启旋向识别
            #判断是否开始预测
            if self.detect == -1 or self.detect == 1:
                label = 1

            #通过中心坐标解算装甲板角度
            if vectorX > 0 and vectorY > 0:
                angle = math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX < 0 and vectorY > 0:
                angle = 180 - math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX < 0 and vectorY < 0:
                angle = 180 + math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX > 0 and vectorY < 0:
                angle = 360 - math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX == 0 and vectorY > 0:
                angle = 270
            elif vectorX == 0 and vectorY <= 0:
                angle = 90
            elif vectorY == 0 and vectorX > 0:
                angle = 0
            elif vectorY == 0 and vectorX <= 0:
                angle = 180
            else:
                label = 0

        #对大符半径进行滤波
        # if x != -1 or y != -1: 
        #     hitDis = self.EuclideanDistance([x, y], Center)
        #     self.history_hisdis_list.append(hitDis)
        # if len(self.history_hisdis_list) > 0:
        #     hitDis = np.mean(np.array(self.history_hisdis_list))
        # if len(self.history_hisdis_list) > self.history_hisdis_list_len_max:
        #     del self.history_hisdis_list[0]
        hitDis = self.EuclideanDistance([x, y], Center)

        if label:
            if size == 1 or size == 2:
                #小幅预测
                forecast_angle = self.energymac_forecast_small(angle)
            elif size == 4 or size == 5:
                #大幅正弦预测
                ret, forecast_angle = self.energymac_forecast_big(angle, vectorX, vectorY)
                if ret == -1:
                    return -1, -1
            else:
                forecast_angle = angle
                print('error: unknow label to hit')
            #若关闭自动旋向，不进行预测
            if self.detect_debug:
                forecast_angle = angle
            forecast_angle = forecast_angle/180*math.pi
            hitX = Center[0] + hitDis*math.cos(forecast_angle)
            hitY = Center[1] + hitDis*math.sin(forecast_angle)
            return hitX, hitY
        else:
            return -1,-1


    def energymac_forecast_small(self,angle):
        #小符预测
        #该变量为预测小符时候的提前角度，需要比赛前调参更改
        angle = angle + self.predictAngle*self.detect
        #angle = angle + 30*self.detect
        return angle

    def energymac_forecast_big(self,angle,x,y):
        #大符预测
        # if x!= -1 and y != -1:
        #     if len(self.history_angle_diff_list) == 0:
        #         self.history_angle_diff_list.append([x,y,self.t0])
        #     elif len(self.history_angle_diff_list) > 0:
        #         angle1 = math.atan2(self.history_angle_diff_list[-1][0], self.history_angle_diff_list[-1][1])
        #         angle2 = math.atan2(x, y)
        #         delta_angle = (angle2 - angle1)*180/math.pi
        #         if abs(delta_angle) < 5:
        #             if self.last_delta_angle == None:
        #                 self.last_delta_angle = delta_angle
        #             else:             
        #                 self.last_delta_angle = delta_angle*0.1 + self.last_delta_angle*0.9
        #             self.temp_delta_angle_list.append([self.last_delta_angle ,self.t0])
        #             mean_angle = np.mean(np.array(self.temp_delta_angle_list)[:,0])
        #             mean_time = np.mean(np.array(self.temp_delta_angle_list)[:,1])
        #         else:
        #             mean_angle = None
        #         if len(self.temp_delta_angle_list) > 30:
        #             del self.temp_delta_angle_list[0]
        #         #print(mean_angle)
        #         if mean_angle != None and abs(mean_angle)< 0.1:
        #             self.begin_time = mean_time
        #             self.pass_num = 0

        #         else:
        #             self.pass_num += 1
        #             if self.pass_num > 20:
        #                 self.temp_delta_angle_list = []
        #         self.history_angle_diff_list.append([x,y,self.t0])
        #     if len(self.history_angle_diff_list) > 4:
        #         del self.history_angle_diff_list[0]
        #     if self.begin_time != None:
        #         delta_t = self.t0 - self.begin_time
        #         forecast_angle = angle + (18 - 18*math.cos(self.big_energy_w*delta_t-math.pi/4))*self.detect
        #         return 1 ,forecast_angle
        #     else:
        #         return -1,angle
        # else:
        #     return -1,angle   
        #大符预测，未完工
        if x != -1 and y != -1:
            #记录历史角度信息
            if len(self.history_angle_diff_list) == 0:
                self.history_angle_diff_list.append([x,y,0,0,self.t0])
            elif len(self.history_angle_diff_list) > 0:
                angle1 = math.atan2(self.history_angle_diff_list[-1][0], self.history_angle_diff_list[-1][1])
                angle2 = math.atan2(x, y)
                delta_angle = (angle2 - angle1)*180/math.pi
                if abs(delta_angle) < self.delta_angle_max:
                    if self.history_angle_diff_list[-1][2] == 0:
                        self.history_angle_diff_list[-1] = [x,y,delta_angle,self.frame_diff_list[-1],self.t0]
                    else:
                        self.history_angle_diff_list.append([x,y,delta_angle,self.frame_diff_list[-1],self.t0])
                else:
                    if self.history_angle_diff_list[-1][2] == 0:
                        self.history_angle_diff_list[-1] = [x,y,0,0,self.t0]
                    else:
                        self.history_angle_diff_list.append([x,y,0,0,self.t0])
            if len(self.history_angle_diff_list) > 20:#self.history_angle_diff_list_len_max:
                del self.history_angle_diff_list[0]
            if len(self.history_angle_diff_list) > 5:
                spd = np.sum(np.array(self.history_angle_diff_list)[:,2])/np.sum(np.array(self.history_angle_diff_list)[:,3])
                #print(spd)
                #self.f.write(str(spd)+' '+str(self.t0)+'\n')
            else:
                spd = None
                accelerate = 0

            if spd != None:
                if self.big_energy_t == None or self.reset_big_energy == None:
                    #获取初相位
                    self.reset_big_energy = True
                    #self.correct_phase(spd)
                    accelerate = 0

                else:
                    pass
        else:
            spd = None
            accelerate = 0 
        if spd != None :#and mean != -1:
            ret = 1
            if abs(spd) < 20:#abs(mean)*0.6:
                #print(0)
                forecast_angle = angle #5*self.detect#+ 33*self.detect
            elif abs(spd) < 50:#abs(mean):
                forecast_angle = angle + 15*self.detect
                #print(1)
            elif abs(spd) < 90:#abs(mean)*1.4:
                forecast_angle = angle + 30*self.detect
                #print(2)
            elif abs(spd) > 90:#abs(mean)*1.4:
                forecast_angle = angle + 45*self.detect
                #print(3)
            else:
                forecast_angle = angle
                ret = -1
            return ret, forecast_angle
        else:
            return -1,angle
    
    def judge_rotate_direct(self,x,y):
        #判断符的旋转方向
        if len(self.history_angle_list) == 0:
            self.history_angle_list.append([x,y,0])
        elif len(self.history_angle_list) > 0:
            angle1 = math.atan2(self.history_angle_list[-1][0], self.history_angle_list[-1][1])
            angle2 = math.atan2(x, y)
            delta_angle = (angle2 - angle1)*180/math.pi
            if abs(delta_angle) < self.delta_angle_max:
                if self.history_angle_list[-1][2] == 0:
                    self.history_angle_list[-1] = [x,y,delta_angle]
                else:
                    self.history_angle_list.append([x,y,delta_angle])
            else:
                if self.history_angle_list[-1][2] == 0:
                    self.history_angle_list[-1] = [x,y,0]
                else:
                    self.history_angle_list.append([x,y,0])
        if len(self.history_angle_list) > 100:
            del self.history_angle_list[0]
        if len(self.history_angle_list) > 5:
            temp_angle = np.sum(np.array(self.history_angle_list)[:,2])
        else:
            temp_angle = 0
        if temp_angle > 0:
            return -1
        elif temp_angle < 0:
            return 1
        else:
            return 0


    def EuclideanDistance(self,c,c0):
        '''
        计算欧氏距离
        @para c(list):[x, y]
        @para c0(list):[x, y]
        @return double:欧氏距离
        '''
        return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)
