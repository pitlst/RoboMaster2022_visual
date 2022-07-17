import numpy as np
import math
import json
from logger import log

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

class AnglePredicted:
    def __init__(self,mode):
        #打符预测类初始化
        self.mode = mode
        self.read_json()
        #卡尔曼滤波初始化
        dt = 1.0/60
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        self.H = np.array([1, 0, 0]).reshape(1, 3)
        # 过程噪声Q
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]]) * self.Q_noise
        # 测量噪声R
        R = np.array([1.0]).reshape(1, 1) * self.R_noise
        self.kf = KalmanFilter(F = F, H = self.H, Q = Q, R = R)
        #旋转方向默认值为1
        self.detect = 1
        #判断旋向初始化
        self.history_angle_list = []
        #大符历史记录初始化
        self.history_angle_diff_list = []
        self.begin_time = None
    
    def reinit(self,mode):
        #打符预测类重初始化
        self.mode = mode
        #self.read_json()
    
    def read_json(self):
        #读取配置文件
        with open('./json/Energy_parameter.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.predict_small = load_dict["predicted"]["predict_small"]
            self.predict_big = load_dict["predicted"]["predict_big"]
            self.R_noise = float(load_dict["predicted"]["R_noise"])
            self.Q_noise = float(load_dict["predicted"]["Q_noise"])
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.detect_debug = load_dict["Debug"]["detect_auto"]
            self.predict_debug = load_dict["Debug"]["predict_debug"]

    def NormalHit(self,msg,f_time):
        #预测前处理，将笛卡尔坐标转换为极坐标，对角度预测
        x,y,center = msg
        self.t0 = f_time
        if self.begin_time == None:
            self.begin_time = self.t0

        if x == -1 or y == -1 or -1 in center:
            vectorX = -1
            vectorY = -1
            angle = -1
            hitDis = -1
        else:
            vectorX = x - center[0]
            vectorY = y - center[1]

            #判断旋转方向
            if self.detect_debug:
                self.detect = self.judge_rotate_direct(vectorX,vectorY)
            else:
                if self.mode in [1,4]:
                    self.detect = -1
                else:
                    self.detect = 1
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
                angle = -1

            #计算大符半径
            hitDis = self.EuclideanDistance([x, y], center)

        if angle != -1:
            if self.predict_debug:
                if self.mode in [1,2]:
                    #小幅预测
                    forecast_angle = self.energymac_forecast_small(angle)
                elif self.mode in [4,5]:
                    #大幅预测
                    pre_angle = self.energymac_forecast_big(vectorX,vectorY)
                    #分段映射防止突变
                    if pre_angle < 8:
                        pre_angle = pre_angle
                    elif pre_angle > 130:
                        pre_angle = 130
                    forecast_angle = angle + abs(pre_angle)*self.predict_big*self.detect
                else:
                    forecast_angle = angle
                    log.print_error('unknow label to hit')
            else:
                forecast_angle = angle
            forecast_angle = forecast_angle/180*math.pi
            hitX = center[0] + hitDis*math.cos(forecast_angle)
            hitY = center[1] + hitDis*math.sin(forecast_angle)
            self.x = hitX
            self.y = hitY
            return hitX, hitY, 1
        else:
            return -1,-1,-1


    def energymac_forecast_small(self,angle):
        #小符预测
        angle = angle + self.predict_small*self.detect
        return angle

    def energymac_forecast_big(self,x,y):
        #大符预测
        pre_angle = 0
        if len(self.history_angle_diff_list) == 0:
            self.history_angle_diff_list.append([x,y,self.t0])
        elif len(self.history_angle_diff_list) > 0:
            angle1 = math.atan2(self.history_angle_diff_list[-1][0], self.history_angle_diff_list[-1][1])
            angle2 = math.atan2(x, y)
            delta_angle = (angle2 - angle1)*180/math.pi
            delta_time = (self.t0 - self.history_angle_diff_list[-1][2])
            if abs(delta_angle) < 3:
                self.kf.update(delta_angle/delta_time)
                pre_angle = np.dot(self.H, self.kf.predict())[0]
                self.history_angle_diff_list.append([x,y,self.t0])
            else:
                self.history_angle_diff_list = []
        #限制记录器大小
        if len(self.history_angle_diff_list) > 4:
            del self.history_angle_diff_list[0]
        return pre_angle
    
    @count
    def judge_rotate_direct(self,x,y):
        #判断符的旋转方向
        if len(self.history_angle_list) == 0:
            self.history_angle_list.append([x,y,0])
        elif len(self.history_angle_list) > 0:
            angle1 = math.atan2(self.history_angle_list[-1][0], self.history_angle_list[-1][1])
            angle2 = math.atan2(x, y)
            delta_angle = (angle2 - angle1)*180/math.pi
            if abs(delta_angle) < 5:
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
        #根据累积的角度判断旋向
        if temp_angle > 0:
            return -1
        elif temp_angle < 0:
            return 1
        else:
            return 0

    def EuclideanDistance(self,c,c0):
        #计算欧氏距离
        return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)
    
    def get_debug_msg(self):
        #返回输出坐标
        return [self.x,self.y]

#卡尔曼滤波
class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


