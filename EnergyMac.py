import cv2
import math
import json
import time
import numpy as np
from logger import log
from openvino.inference_engine import IECore
from Eenergy_predicted import AnglePredicted

class GetEnergyMac:
    
    def __init__(self,debug,video_debug_set,color):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        #深度学习打符类初始化
        #输入变量私有化
        self.color = color
        self.debug = debug
        self.video_debug_set = video_debug_set
        #深度学习初始化
        self.__openvino_init()
        #读取对应json获取参数
        self.__read_energy()
        #初始化打符预测类
        self.AnglePredicted_class = AnglePredicted()
        #记录opencv版本的标志位
        if int(cv2.__version__[0]) == 4:
            self.version = 1
        else:
            self.version = 0 
        if self.debug:
            self.t0 = time.time()

    def __read_energy(self):
        #该函数用于读取打符参数并进行适当处理
        with open('./json/Energy_parameter.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.MaxRsS = load_dict["tradition"]["MaxRsS"]
            self.MinRsS = load_dict["tradition"]["MinRsS"]
            self.MaxRsRatio = load_dict["tradition"]["MaxRsRatio"]
            self.predictAngle = load_dict["tradition"]["predictAngle_small"] 
            self.pass_number_max = load_dict["tradition"]["pass_number_max"] 
            self.nms_distence_max = load_dict["tradition"]["nms_distence_max"] 
            self.fan_armor_distence_max = load_dict["deep"]["fan_armor_distence_max"]
            self.fan_armor_distence_min = load_dict["deep"]["fan_armor_distence_min"]
            self.armor_R_distance_max = load_dict["deep"]["armor_R_distance_max"]
            self.armor_R_distance_min = load_dict["deep"]["armor_R_distance_min"] 
            self.center_dis_y = load_dict["deep"]["center_dis_y"] 
            self.center_dis_x = load_dict["deep"]["center_dis_x"] 
            self.model_path = load_dict["deep"]["model_path"]
        with open('./json/Energy_find.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            if self.color == 1:
                self.hsv_low = np.array(load_dict["hsv"]["hsv_red_low"]) 
                self.hsv_high = np.array(load_dict["hsv"]["hsv_red_high"]) 
            else:
                self.hsv_low = np.array(load_dict["hsv"]["hsv_blue_low"]) 
                self.hsv_high = np.array(load_dict["hsv"]["hsv_blue_high"]) 
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.Energy_R_debug = load_dict["Debug"]["Energy_R_debug"]
            self.predict_debug = load_dict["Debug"]["predict_debug"]
        with open('./json/common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.frame_size = load_dict["Energy_mac"]["width"]
        #选取模型输入大小
        self.model_img_size = self.h
        self.last_x = self.model_img_size/2
        self.last_y = self.model_img_size/2
        # 这部分是大符各中心的距离关系，不需要频繁更改。
        # 实际上对于性能达到正常水平的模型来说，所有在这里的参数都是多余的，这些都只是保险措施，应该用不上。
        self.fan_armor_distence_max = self.fan_armor_distence_max*self.model_img_size
        self.fan_armor_distence_min = self.fan_armor_distence_min*self.model_img_size
        self.armor_R_distance_max = self.armor_R_distance_max*self.model_img_size
        self.armor_R_distance_min = self.armor_R_distance_min*self.model_img_size
        #同方向局部nms最大值
        self.nms_distence_max = self.nms_distence_max*self.model_img_size
        # 这部分为最大最小面积
        self.MaxRsS = self.MaxRsS*(self.model_img_size**2)
        self.MinRsS = self.MinRsS*(self.model_img_size**2)
        #关于中心丢失判断的相关初始化
        self.last_center = [-1,-1]
        self.pass_number = 0
        
    def __openvino_init(self):
        #openvino模块的初始化
        self.ieCore = IECore()
        self.net = self.ieCore.read_network(model=self.model_path)
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape
        self.exec_net = self.ieCore.load_network(network=self.net, device_name='GPU')
        #预先获取修改维度后的网格矩阵
        test_image = np.zeros((1,3,self.h,self.w))
        res = self.exec_net.infer(inputs={self.input_blob: test_image})
        #这部分anchors随着训练时的参数变化，一般不用再次聚类去求。
        self.anchors = np.array([[[[[[23,29]]],  [[[43,55]]],  [[[73,105]]]]],[[[[[146,217]]],  [[[231,300]]],  [[[335,433]]]]],[[[[[4,5]]],  [[[8,10]]],  [[[13,16]]]]]])
        self.p_mat,self.p_gd = self.mat_process(res)

    def reinit(self,color,mode):
        #根据更新的颜色信息更换变量
        self.color = color
        self.mode = mode
        self.__read_energy()
    
    def mat_process(self,res):        
        #预先获取扩充维度的网格矩阵
        i = 0
        premat = []
        pregrid = []
        for r in res:
            _,_,nx,ny,cao = res[r].shape

            xv, yv = np.meshgrid([np.arange(ny)], [np.arange(nx)])
            zz = np.stack((xv, yv), 2).reshape((1,1,ny,nx,2)) 

            grid = np.concatenate((zz,zz,zz),1)        #这里的意思就是拼接同样的矩阵，让形状和之前返回的相同
            aa = np.ones_like(grid)*self.anchors[i]
            premat.append(aa)
            pregrid.append(grid)

            i += 1
        return premat,pregrid
    
    
    def process(self,res):
        #对网络输出进行处理
        z = []
        stride = np.array([16,32,8])
        i = 0
        for r in res:
            _,_,nx,ny,cao = res[r].shape

            grid = self.p_gd[i]
            aa = self.p_mat[i]
            # 选取置信度符合标准的继续处理
            # 在将其他值拆解为正常相对值之前优先处理置信度，减少之后需要的计算
            xc = res[r][..., 4] > 0
            # 同步处理其他需要的返回   
            y = res[r][xc]
            gg = grid[xc]
            aa = aa[xc]

            y = self.sigmoid(y)
            
            '''
            处理如下，其中 anchor_grid是缩放系数默认为1.
            tx = sigmoid(gtx) * 2 - 0.5
            ty = sigmoid(gty) * 2 - 0.5
            tw = (sigmoid(gtw) * 2) ** 2
            th = (sigmoid(gth) * 2) ** 2
            gtx：ground truth x， tx：transfer x

            如果不谈级效率的取舍，正常逻辑可参考下面
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))
            其中x[i]是模型的输出就是预测出来的值

            通过了这种转换,让模型输出的xy控制在[-0.5, 1.5]， wh控制在[0, 4]，
            xy变成了相对于anchor中心点的偏移量，wh变成了相对于anchor的wh的缩放系数，
            通过学习这样的xywh来学习bbox，这就是在yolov5中anchor的真实面目，包括训练推理预测的anchor都是以这格式进行传输。
            而预测出来的xywh也是要通过上面式子的反式子推出来就是预测的bbox了

            这里的gg和aa实际上就是提前处理好的anchor，只需要根据定义把值相加或相乘
            0，1为xy,2,3为wh,4是前景背景区分的置信度，用于表示该预测框是否有目标
            5，6，7是预定的三种标签的置信度，训练时定义多少就是多少，顺序也不变
            最后一个是角度信息
            ''' 
            y[:,0:2] = (y[:,0:2] * 2.0 - 0.5 + gg) * stride[i]  
            y[:,2:4] = (y[:,2:4] * 2) ** 2 * aa 
            z.append(y.reshape(1, -1,cao))
            i += 1
            
        pred = np.concatenate(z, 1)
        return pred

    def my_nms(self,p):
        #因为大符的预测框很难有较大交错，所以直接按类和中心距离nms即可，我们的nms没有iou计算
        #my_nms既做了nms，也筛出了唯一的中心坐标和待击打坐标
        #注意，待打击点是会有坐标突变的，但是中心不会，因此，中心坐标如果突变，需要筛去
        result = []
        center = []
        for i in p[0]:
            cls = self.get_cls(i)
            if cls == 2:
                #对中心做nms
                if len(center) == 0:
                    center.append(i)
                else:
                    add = True
                    for t,center_t in enumerate(center):
                        if (i[0]-center_t[0])**2+(i[1]-center_t[1])**2 < self.nms_distence_max**2:
                            add = False
                            if center_t[4] < i[4]:
                                center[t] = i
                                break
                    if add:
                        center.append(i)
            else:
                #对armor和full做nms
                if len(result) == 0:
                    result.append(i)
                else:
                    add = True
                    for table,j in enumerate(result):
                        cls_j = self.get_cls(j)
                        if cls_j == cls:
                            #nms 非极大值抑制
                            if (i[0]-j[0])**2+(i[1]-j[1])**2 < self.nms_distence_max**2:
                                add = False
                                if j[4] < i[4]:
                                    result[table] = i
                                    break
                    if add:
                        result.append(i)
        return center,result
    

    def center_filter(self,center,center_tradition):
        #对R进行筛选，根据debug参数选择方案，有待优化
        Center = []
        if self.Energy_R_debug == 0:
            #纯深度学习，未完成
            if len(center):
                for j,temp_j in enumerate(center):
                    if len(Center) == 0:
                        Center = temp_j
                    else:
                        if Center[4] < temp_j[4]:
                            Center = temp_j
            else:
                Center = []
        elif self.Energy_R_debug == 1:
            #纯传统视觉,未完成
            Center = center_tradition[0]
        elif self.Energy_R_debug == 2:
            #对传统center与深度学习center取交集
            if len(center_tradition):
                for j, temp_j in enumerate(center):
                    label = 1
                    for i,temp_i in enumerate(center_tradition):
                        if (temp_i[0]-temp_j[0])**2+(temp_i[1]-temp_j[1])**2 < self.nms_distence_max**2:
                            label = 0
                            break
                    if label:
                        del center[j]

                for j,temp_j in enumerate(center):
                    if len(Center) == 0:
                        Center = temp_j
                    else:
                        if Center[4] < temp_j[4]:
                            Center = temp_j
            else:
                Center = []
        elif self.Energy_R_debug == 3:
            #深度学习丢失后用传统视觉弥补，未完成
            if center_tradition[0] != -1 or center_tradition[1] != -1:
                if center[0] == -1 or center[1] == -1 or (center[0]-self.last_center[0])**2+(center[1]-self.last_center[1])**2>300:
                    center = center_tradition
            if center[0] != -1 or center[1] != -1:
                self.last_center = center
                if self.pass_number != 0:
                    self.pass_number = 0
            elif self.last_center[0] != -1 or self.last_center[1] != -1:
                center = self.last_center
                self.pass_number += 1
                if self.pass_number > self.pass_number_max:
                    self.last_center = [-1,-1]
            else:
                Center = []
        else:
            Center = []

        if len(Center):
            Center = Center[0:4]
        else:
            Center = [-1,-1,-1,-1]

        return Center
    
    def energy_filter(self,center,result):
        #对神经网络输出的目标进行传统上的筛选，减少误识别
        hit_pos = [[-1,-1,-1]]
        for i in result:
            cls = self.get_cls(i)
            if cls == 0:
                #这里是筛选识别的装甲板与旋转中心R的距离
                if center[0] != -1:
                    if (center[0]-i[0])**2+(center[1]-i[1])**2 > self.armor_R_distance_min**2 and \
                        (center[0]-i[0])**2+(center[1]-i[1])**2 < self.armor_R_distance_max**2:
                        pos_true = True
                    else:
                        pos_true = False
                else:
                    pos_true = False

                if pos_true:
                    for j in result:
                        cls_j = self.get_cls(j)
                        if cls_j == 1:
                            #筛选扇叶是否击打，判断装甲中心附近是否有完整扇叶中心，给一个像素的欧式距离
                            if (j[0]-i[0])**2+(j[1]-i[1])**2 < self.fan_armor_distence_max**2 and \
                                (j[0]-i[0])**2+(j[1]-i[1])**2 > self.fan_armor_distence_min**2:
                                pos_true = False
                                break
                            else:
                                pos_true = True

                if pos_true:
                    hit_pos = [[i[0]*self.model_img_size/self.h,i[1]*self.model_img_size/self.w,1]]
                    break
        return hit_pos

    def GetHitPointDL(self, frame, f_time, size):
        #保护图像变量用来画
        x, y, z = -1, -1, -1
        #预处理部分
        if self.video_debug_set == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
        if frame.shape[:-1] != (self.h, self.w):
            frame_reasize = cv2.resize(frame,(self.w,self.h))
        else:
            frame_reasize = frame
        # 传统视觉部分,根据debug参数启用
        # 传统视觉识别R是为了弥补家里的大符的无奈之举，比赛时一般不会出现如此极端的情况
        if self.Energy_R_debug:
            mask = self.HSV_Process(frame_reasize)
            center_tradition = self.FindRsignScope(mask)
        else:
            center_tradition = [-1,-1,-1,-1]
        #深度学习部分
        frame_deal = frame_reasize.astype('float32')
        frame_deal = frame_deal/255  #像素归一化
        frame_deal = frame_deal.transpose((2,0,1))
        frame_deal = np.expand_dims(frame_deal,axis=0)
        #推理
        res = self.exec_net.infer(inputs={self.input_blob: frame_deal})
        #后处理
        pred = self.process(res)
        center, result= self.my_nms(pred)
        #筛选中心
        center = self.center_filter(center,center_tradition)
        if center[0] == -1 or center[1] == -1:
            x,y,z = -1,-1,-1
        else:
            z = 0
            #对找到的R进行补偿使其接近实际旋转中心    
            center[0] = center[0] + self.center_dis_x
            center[1] = center[1] + self.center_dis_y
            #筛选待击打装甲板
            hit_pos = self.energy_filter(center,result)
            x = float(hit_pos[0][0])
            y = float(hit_pos[0][1])
            #预测
            if self.predict_debug:
                x,y = self.AnglePredicted_class.NormalHit(center, x, y, f_time, size)
                x = x/self.model_img_size*self.frame_size
                y = y/self.model_img_size*self.frame_size


        #画出图像,
        if self.debug:
            self.img4 = frame.copy()
            self.img = frame_reasize.copy()
            self.img2 = frame_reasize.copy()
            self.img3 = frame_reasize.copy()
            result = np.array(result)
            self.img = self.draw_pred(self.img,pred[0],center)
            self.img = self.draw_center(self.img,center)
            self.img2 = self.draw_pred(self.img2,result,center)
            self.img2 = self.draw_center(self.img2,center)
            cv2.circle(self.img2,(int(hit_pos[0][0]),int(hit_pos[0][1])),int(self.fan_armor_distence_max),self.colors[2],1)
            cv2.circle(self.img2,(int(hit_pos[0][0]),int(hit_pos[0][1])),int(self.fan_armor_distence_min),self.colors[2],1)
            cv2.circle(self.img2,(int(hit_pos[0][0]),int(hit_pos[0][1])),int(self.nms_distence_max),self.colors[0],1)
            cv2.circle(self.img2,(int(center[0]),int(center[1])),int(self.armor_R_distance_max),self.colors[2],1)
            cv2.circle(self.img2,(int(center[0]),int(center[1])),int(self.armor_R_distance_min),self.colors[2],1)
            cv2.circle(self.img4,(int(x),int(y)),4,(255,255,255),-1)
            for c_t in center_tradition:
                cv2.circle(self.img3,(int(c_t[0]),int(c_t[1])),8,(255,255,255),-1)
            self.updata_argument()
            #每隔1秒更新一次json文件
            if time.time()-self.t0 > 1:
                self.update_json()
                self.t0 = time.time()

        return x,y,z


    def HSV_Process(self,frame):
        #图像二值化
        frame = cv2.GaussianBlur(frame,(5,5),0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(frame, self.hsv_low, self.hsv_high)
        if self.debug:
            cv2.imshow('mask',mask)
        return mask


    def FindRsignScope(self,mask):
        #筛选中心R
        Center_return = []
        mask = cv2.dilate(mask, (13,13))
        #对于opencv3，轮廓寻找函数有3个返回值，对于opencv4只有两个
        if self.version:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contoursLength = len(contours)
        for c in range(contoursLength):
            contoursFine = True
            center, size, angle = cv2.minAreaRect(contours[c])
            #得到长边和中心点垂直短边向量
            if size[0]<size[1]:
                longSide = size[1]
                shortSide = size[0]
            else:
                longSide = size[0]
                shortSide = size[1]
            if longSide*shortSide > self.MaxRsS:
                contoursFine = False
            if longSide*shortSide < self.MinRsS:
                contoursFine = False
            if longSide > self.MaxRsRatio*shortSide:
                contoursFine = False
            if contoursFine:
                Center_return.append([center[0],center[1],longSide,shortSide])
                break
        return Center_return

    @staticmethod
    def draw_pred(img,pred,center):
        #画出来的，按自己喜好来就可以了，深度学习检测基本不用实时调参，不咋重要
        #这个画扇叶和装甲板
        colors = [[255,255,0],[0,255,0],[0,255,255]]
        if pred.all() != None or len(center):
            for det in pred:
                x = det[0]
                y = det[1]
                h = det[2]
                w = det[3]
                vectorX = x - center[0]
                vectorY = y - center[1]
                if det[5]>det[6]:
                    if det[5] > det[7]:
                        label = 0
                    else:
                        label = 2
                else:
                    if det[6] > det[7]:
                        label = 1
                    else:
                        label = 2
                box = []

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
                angle = angle/180*math.pi

                if label == 0:
                    angle = angle - math.pi/2
                x0 = x - h/2*math.cos(angle) + w/2*math.sin(angle)
                y0 = y - h/2*math.sin(angle) - w/2*math.cos(angle)
                box.append([x0,y0])
                x1 = x - h/2*math.cos(angle) - w/2*math.sin(angle)
                y1 = y - h/2*math.sin(angle) + w/2*math.cos(angle)
                box.append([x1,y1])
                x2 = x + h/2*math.cos(angle) - w/2*math.sin(angle)
                y2 = y + h/2*math.sin(angle) + w/2*math.cos(angle)
                box.append([x2,y2])
                x3 = x + h/2*math.cos(angle) + w/2*math.sin(angle)
                y3 = y + h/2*math.sin(angle) - w/2*math.cos(angle)
                box.append([x3,y3])
                box = np.array(box,dtype = 'int32')

                cv2.drawContours(img,[box],0,colors[label],1)
                cv2.circle(img,(int(x),int(y)),4,colors[label],-1)
        else:
            log.print_error('No box is available')
        return img

    @staticmethod
    def draw_center(img,center):
        #画出来的，按自己喜好来就可以了，深度学习检测基本不用实时调参，不咋重要
        angle = 0
        box = []
        if len(center):
            x = center[0]
            y = center[1]
            w = center[2]
            h = center[3]
            x0 = x - h/2*math.cos(angle) + w/2*math.sin(angle)
            y0 = y - h/2*math.sin(angle) - w/2*math.cos(angle)
            box.append([x0,y0])
            x1 = x - h/2*math.cos(angle) - w/2*math.sin(angle)
            y1 = y - h/2*math.sin(angle) + w/2*math.cos(angle)
            box.append([x1,y1])
            x2 = x + h/2*math.cos(angle) - w/2*math.sin(angle)
            y2 = y + h/2*math.sin(angle) + w/2*math.cos(angle)
            box.append([x2,y2])
            x3 = x + h/2*math.cos(angle) + w/2*math.sin(angle)
            y3 = y + h/2*math.sin(angle) - w/2*math.cos(angle)
            box.append([x3,y3])
            box = np.array(box,dtype = 'int32')          
            cv2.drawContours(img,[box],0,(0,255,0),1)
            cv2.circle(img,(int(center[0]),int(center[1])),4,(0,255,0),-1)
        else:
            log.print_error('No box is available')
        return img


    def EuclideanDistance(self,c,c0):
        #计算欧氏距离
        return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)
    
    def get_cls(self,det):
        #该函数用于获取一个框的标签
        if det[5]>det[6]:
            if det[5] > det[7]:
                label = 0
            else:
                label = 2
        else:
            if det[6] > det[7]:
                label = 1
            else:
                label = 2
        return label

    def TrackerBar_create(self):
        #创建滑动条,并备份参数
        if self.debug:
            # Lower range colour sliders.
            cv2.createTrackbar('lowHue', 'energyTest', self.hsv_low[0], 255, self.nothing)
            cv2.createTrackbar('lowSat', 'energyTest', self.hsv_low[1], 255, self.nothing)
            cv2.createTrackbar('lowVal', 'energyTest', self.hsv_low[2], 255, self.nothing)    
            # Higher range colour sliders.
            cv2.createTrackbar('highHue', 'energyTest', self.hsv_high[0], 255, self.nothing)
            cv2.createTrackbar('highSat', 'energyTest', self.hsv_high[1], 255, self.nothing)
            cv2.createTrackbar('highVal', 'energyTest', self.hsv_high[2], 255, self.nothing)

            cv2.createTrackbar('MaxRsS0.0000', 'energyTest', int(self.MaxRsS/self.model_img_size*10000), 100, self.nothing)
            cv2.createTrackbar('MinRsS0.0000', 'energyTest', int(self.MinRsS/self.model_img_size*10000), 100, self.nothing)
            cv2.createTrackbar('MaxRsRatio0.000', 'energyTest', int(self.MaxRsRatio/self.model_img_size/self.model_img_size*1000), 100, self.nothing)
            cv2.createTrackbar('fan_armor_distence_max0.000', 'energyTest', int(self.fan_armor_distence_max/self.model_img_size*1000), 255, self.nothing)
            cv2.createTrackbar('fan_armor_distence_min0.000', 'energyTest', int(self.fan_armor_distence_min/self.model_img_size*1000), 255, self.nothing)
            cv2.createTrackbar('armor_R_distance_max0.000', 'energyTest', int(self.armor_R_distance_max/self.model_img_size*1000), 255, self.nothing)
            cv2.createTrackbar('armor_R_distance_min0.000', 'energyTest', int(self.armor_R_distance_min/self.model_img_size*1000), 255, self.nothing)


    def updata_argument(self):
        #根据debug更新参数
        if self.debug:
            lowHue = cv2.getTrackbarPos('lowHue', 'energyTest')
            lowSat = cv2.getTrackbarPos('lowSat', 'energyTest')
            lowVal = cv2.getTrackbarPos('lowVal', 'energyTest')
            highHue = cv2.getTrackbarPos('highHue', 'energyTest')
            highSat = cv2.getTrackbarPos('highSat', 'energyTest')
            highVal = cv2.getTrackbarPos('highVal', 'energyTest')
            self.hsv_high = np.array([highHue,highSat,highVal])
            self.hsv_low = np.array([lowHue,lowSat,lowVal])
            self.MaxRsS = float(cv2.getTrackbarPos('MaxRsS0.0000', 'energyTest'))/10000*self.model_img_size
            self.MinRsS = float(cv2.getTrackbarPos('MinRsS0.0000', 'energyTest'))/10000*self.model_img_size
            self.MaxRsRatio = float(cv2.getTrackbarPos('MaxRsRatio0.000', 'energyTest'))/1000*self.model_img_size
            self.fan_armor_distence_max = float(cv2.getTrackbarPos('fan_armor_distence_max0.000', 'energyTest'))/1000*self.model_img_size
            self.fan_armor_distence_min = float(cv2.getTrackbarPos('fan_armor_distence_min0.000', 'energyTest'))/1000*self.model_img_size
            self.armor_R_distance_max = float(cv2.getTrackbarPos('armor_R_distance_max0.000', 'energyTest'))/1000*self.model_img_size
            self.armor_R_distance_min = float(cv2.getTrackbarPos('armor_R_distance_min0.000', 'energyTest'))/1000*self.model_img_size

    def update_json(self):
        #更新json文件中的参数
        if self.debug:
            with open('./json/Energy_find.json','r',encoding = 'utf-8') as load_f:
                load_dict = json.load(load_f,strict=False)
                if self.color == 1:
                    load_dict["hsv"]["hsv_red_high"] = [int(x) for x in list(self.hsv_high)]
                    load_dict["hsv"]["hsv_red_low"] = [int(x) for x in list(self.hsv_low)]
                else:
                    load_dict["hsv"]["hsv_blue_high"] = [int(x) for x in list(self.hsv_high)]
                    load_dict["hsv"]["hsv_blue_low"] = [int(x) for x in list(self.hsv_low)]
                load_dict["EnergyFind"]["MaxRsS"] = self.MaxRsS/self.model_img_size
                load_dict["EnergyFind"]["MinRsS"] = self.MinRsS/self.model_img_size
                load_dict["EnergyFind"]["MaxRsRatio"] = self.MaxRsRatio/self.model_img_size
                load_dict["EnergyFind"]["fan_armor_distence_max"] = self.fan_armor_distence_max/self.model_img_size
                load_dict["EnergyFind"]["fan_armor_distence_min"] = self.fan_armor_distence_min/self.model_img_size
                load_dict["EnergyFind"]["armor_R_distance_max"] = self.armor_R_distance_max/self.model_img_size
                load_dict["EnergyFind"]["armor_R_distance_min"] = self.armor_R_distance_min/self.model_img_size
                dump_dict = load_dict
            with open('./json/Energy_find.json','w',encoding = 'utf-8') as load_f:
               json.dump(dump_dict,load_f,indent=4,ensure_ascii=False)
    
    def sigmoid(self,inx):
        #对sigmoid函数的优化，避免了出现极大的数据溢出
        inx[inx[:]<-10] = -10
        return 1.0/(1+np.exp(-inx))

    def nothing(self,*arg):
        #空的回调函数
        pass

    def get_debug_frame(self):
        #返回显示所需要的debug图像
        return self.img, self.img2, self.img3, self.img4