import cv2
import math
import json
import numpy as np
from openvino.inference_engine import IECore
from Eenergy_predicted import AnglePredicted

class GetEnergyMac:
    
    def __init__(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

        #读取对应json获取参数
        self.read_energy()
        #深度学习初始化
        self.__openvino_init()
        #初始化预测类
        self.AnglePredicted_class = AnglePredicted()
        self.model_img_size = self.h
        # 这里注意，416这个数字是根据所用模型选取的，这个数字和网络期望输入有区别，
        # 其实际含义是训练时归一化选取的数字，这方面根据性能和精度的需要随时更改，
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

        #关于中心丢失判断的相关初始化
        self.last_center = [-1,-1]
        self.pass_number = 0

        #坐标防抖的历史xy值
        self.history_xy_list = []
        self.x = -1
        self.x = -1

        # 以下为传统视觉筛选R的参数
        # 这部分传统视觉识别R是为了弥补家里的大符的无奈之举，比赛时一般不会出现如此极端的情况，同样的没有编写相关的滑动条调参代码。
        # 这部分为最大最小面积
        self.MaxRsS = self.MaxRsS*(self.model_img_size**2)
        self.MinRsS = self.MinRsS*(self.model_img_size**2)
                        
        #卡尔曼滤波的初始化
        # self.kalman = cv2.KalmanFilter(4,2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        # self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        # self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        # self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.1
        # self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 30
        # self.last_mes = self.current_mes = np.array((2,1),np.float32)
        # self.last_pre = self.current_pre = np.array((2,1),np.float32)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter('./output.avi',fourcc, 30.0, (416,416))
    

    def read_energy(self):
        #该函数用于读取打符参数
        with open('./json/Energy_find.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            armor = load_dict["Color"]["armor"]
            full = load_dict["Color"]["full"]
            R = load_dict["Color"]["R"]
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
            self.hsv_low = np.array(load_dict["tradition"]["hsv_low"]) 
            self.hsv_high = np.array(load_dict["tradition"]["hsv_high"]) 
            self.colors = [armor,full,R]
        with open('./json/debug.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.Energy_debug = load_dict["Debug"]["Energy_debug"]
            self.video_debug_set = load_dict["Debug"]["video_debug_set"]
            self.Energy_R_debug = load_dict["Debug"]["Energy_R_debug"]
            self.predict_debug = load_dict["Debug"]["predict_debug"]
        with open('./json/common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.frame_size = load_dict["Energy_mac"]["width"]
        
    def __openvino_init(self):
        #openvino模块的初始化
        self.ieCore = IECore()
        self.net = self.ieCore.read_network(model='./model/bestyao_2_416.xml')  
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1

        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape
        self.exec_net = self.ieCore.load_network(network=self.net, device_name='CPU')

        #预先获取修改维度后的网格矩阵
        test_image = np.zeros((1,3,self.h,self.w))
        res = self.exec_net.infer(inputs={self.input_blob: test_image})
        #这部分anchors随着训练时的参数变化，一般不用再次聚类去求。
        self.anchors = np.array([[[[[[23,29]]],  [[[43,55]]],  [[[73,105]]]]],[[[[[146,217]]],  [[[231,300]]],  [[[335,433]]]]],[[[[[4,5]]],  [[[8,10]]],  [[[13,16]]]]]])
        self.p_mat,self.p_gd = self.mat_process(res)

    def sigmoid(self,inx):
        inx[inx[:]<-10] = -10
        #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    
    def mat_process(self,res):        
        #预先获取扩充维度的网格矩阵
        i = 0
        premat = []
        pregrid = []
        for r in res:
            _,_,nx,ny,cao = res[r].shape

            xv, yv = np.meshgrid([np.arange(ny)], [np.arange(nx)])
            zz = np.stack((xv, yv), 2).reshape((1,1,ny,nx,2))   #这里有疑问，为什么最后这个重塑形状用的是2

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

    def my_nms(self,p,center_tradition):
        #因为大符的预测框很难有较大交错，所以直接按类和中心距离nms即可
        #my_nms既做了nms，也筛出了唯一的中心坐标和待击打坐标
        #注意，待打击点是会有坐标突变的，但是中心不会
        #因此，中心坐标如果突变，需要筛去
        result = []
        center = [-1,-1,-1,-1]
        center_conf = -1
        hit_pos = [[-1,-1,-1]]
        for i in p[0]:
            cls = self.get_cls(i)
            if cls == 2:
                if i[4]>center_conf:
                    center[0] = i[0]*self.model_img_size/self.h
                    center[1] = i[1]*self.model_img_size/self.w
                    center[2] = i[2]*self.model_img_size/self.h
                    center[3] = i[3]*self.model_img_size/self.w
                    center_conf = i[4]
            else:
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
            return center, hit_pos, result

        #对找到的R进行补偿使其接近实际旋转中心    
        center[0] = center[0] + self.center_dis_x
        center[1] = center[1] + self.center_dis_y
        
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
        return center,hit_pos,result
    

    def GetHitPointDL(self, frame, f_time, size):
        #size是大小符
        #保护图像变量用来画
        x, y, z = -1, -1, -1
        #预处理部分，把一切处理放到这里比较好，传递还能小点
        if self.video_debug_set == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
        if self.Energy_debug:
            img3 = frame.copy()
        if frame.shape[:-1] != (self.h, self.w):
            frame = cv2.resize(frame,(self.w,self.h))
        #传统视觉部分,根据debug参数启用
        if self.Energy_R_debug:
            frame_tran = frame.copy()
            mask = self.HSV_Process(frame_tran)
            center_tradition = self.FindRsignScope(mask)
        else:
            center_tradition = [-1,-1,-1,-1]
        #深度学习部分
        
        if self.Energy_debug:
            img = frame.copy()
            img2 = frame.copy()
        
        frame = frame.astype('float32')
        frame = frame/255  #像素归一化
        frame = frame.transpose((2,0,1))
        frame = np.expand_dims(frame,axis=0)
        #推理
        res = self.exec_net.infer(inputs={self.input_blob: frame})
        
        #后处理
        pred = self.process(res)
        center, hit_pos, result= self.my_nms(pred,center_tradition)
        x = float(hit_pos[0][0])
        y = float(hit_pos[0][1])
        if center[0] == -1:
            x = -1
            y = -1
            z = -1
        else:
            z = 0
        

        if self.predict_debug and z == 0:
            #预测这块，小符就是一个固定角度
            #总而言之，就是在角度坐标系下，做到延迟小，连贯的预测坐标
            #这块需要经过（反复设计+反复打击实验）的过程，我并不能给出最完美的解决办法，但如果你们想不出好办法，下面离散化的做法也不是不可以
            #在设计之前，先进行数据分析，可以下楼采一采各种转速的符获取的打击角度的原始数据，在这些数据上试试你的想法
            #角度坐标系的解算，角度的连续获取和滤波应该很容易看懂，不赘述了
            #x,y = self.NormalHit(center, x, y, f_time)
            x,y = self.AnglePredicted_class.NormalHit(center, x, y, f_time, size)

        #画出图像,
        if self.Energy_debug:
            result = np.array(result)
            img = self.draw_pred(img,pred[0],center)
            img = self.draw_center(img,center)
            img2 = self.draw_pred(img2,result,center)
            img2 = self.draw_center(img2,center)
            cv2.circle(img2,(int(hit_pos[0][0]),int(hit_pos[0][1])),int(self.fan_armor_distence_max),self.colors[2],1)
            cv2.circle(img2,(int(hit_pos[0][0]),int(hit_pos[0][1])),int(self.fan_armor_distence_min),self.colors[2],1)
            cv2.circle(img2,(int(hit_pos[0][0]),int(hit_pos[0][1])),int(self.nms_distence_max),self.colors[0],1)
            cv2.circle(img2,(int(center[0]),int(center[1])),int(self.armor_R_distance_max),self.colors[2],1)
            cv2.circle(img2,(int(center[0]),int(center[1])),int(self.armor_R_distance_min),self.colors[2],1)
            cv2.imshow("img",img)
            cv2.imshow("img2",img2)
            self.video_writer.write(img2)

        #x,y = self.anti_shake_xy(x,y)
        x = x/self.model_img_size*self.frame_size
        y = y/self.model_img_size*self.frame_size
        #坐标防抖：卡尔曼滤波
        # if size == 2:
        #     if x < 0 or y < 0 or z < 0:
        #         x = -1
        #         y = -1   
        #         self.x = x
        #         self.y = y         
        #     else:
        #         if self.x == -1 or self.y == -1:
        #             self.kalman.statePost[0] = np.float32(x)
        #             self.kalman.statePost[1] = np.float32(y)
        #             self.x = x
        #             self.y = y
        #             x = -1
        #             y = -1   
        #         else:
        #             self.current_mes = np.array([[np.float32(x)],[np.float32(y)]])
        #             self.kalman.correct(self.current_mes)
        #             self.current_pre = self.kalman.predict()
        #             self.x = x
        #             self.y = y
        #             x = self.current_pre[0]
        #             y = self.current_pre[1]
        #画出最终输出xy坐标
        if self.Energy_debug:
            cv2.circle(img3,(int(x),int(y)),4,(255,255,255),-1)
            cv2.imshow("img3",img3)

        return x,y,z

    def anti_shake_xy(self,x,y):
        #坐标防抖
        if x < 0 or y < 0:
            # self.history_xy_list = []
            return -1,-1
        elif len(self.history_xy_list) >= 0 and len(self.history_xy_list) <= 3:
            self.history_xy_list.append([x,y])
            return x,y
        elif len(self.history_xy_list) > 3 and (x - self.history_xy_list[-1][0])**2+(y - self.history_xy_list[-1][1])**2 < 400:
            self.history_xy_list.append([x,y])
            x_new = (np.sum(np.array(self.history_xy_list)[:,0])-np.max(np.array(self.history_xy_list)[:,0])-np.min(np.array(self.history_xy_list)[:,0]))/(len(self.history_xy_list)-2)
            y_new = (np.sum(np.array(self.history_xy_list)[:,1])-np.max(np.array(self.history_xy_list)[:,1])-np.min(np.array(self.history_xy_list)[:,1]))/(len(self.history_xy_list)-2)
            if len(self.history_xy_list) > 10:
                del self.history_xy_list[0]
            return x_new, y_new
        elif len(self.history_xy_list) > 3 and (x - self.history_xy_list[-1][0])**2+(y - self.history_xy_list[-1][1])**2 > 400:
            self.history_xy_list = []
            self.history_xy_list.append([x,y])
            return x,y
        else:
            return -1,-1

    def HSV_Process(self,frame):
        #图像二值化
        frame = cv2.GaussianBlur(frame,(5,5),0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(frame, self.hsv_low, self.hsv_high)
        if self.Energy_debug:
            cv2.imshow('mask',mask)
        return mask


    def FindRsignScope(self,mask):
        #筛选中心R
        Center_return = [-1,-1,-1,-1]
        mask = cv2.dilate(mask, (13,13))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
            if longSide*shortSide > self.MaxRsS:#400
                contoursFine = False
            if longSide*shortSide < self.MinRsS:#200
                contoursFine = False
            if longSide > self.MaxRsRatio*shortSide:#1.7:
                contoursFine = False
            if contoursFine:
                Center_return = [center[0],center[1],longSide,shortSide]
                break
        return Center_return

    def draw_pred(self,img,pred,center):
        #画出来的，按自己喜好来就可以了，深度学习检测基本不用实时调参，不咋重要
        #这个画扇叶和装甲板
        if pred.all() != None or len(center):
            for det in pred:
                x = det[0]
                y = det[1]
                h = det[2]
                w = det[3]
                vectorX = x - center[0]
                vectorY = y - center[1]
                label = self.get_cls(det)
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
                cv2.drawContours(img,[box],0,self.colors[label],1)
                cv2.circle(img,(int(x),int(y)),4,self.colors[label],-1)
                #cv2.putText(img,'CSYS:'+str(int(x))+' '+str(int(y)),(int(x)+10,int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                #cv2.putText(img,'angle:'+str(int(temp_angle)),(int(x)+10,int(y)+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        else:
            print('error:No box is available')
        return img

    def draw_center(self,img,center):
        #画出来的，按自己喜好来就可以了，深度学习检测基本不用实时调参，不咋重要
        #这个中心的R
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
            cv2.drawContours(img,[box],0,self.colors[2],1)
            cv2.circle(img,(int(center[0]),int(center[1])),4,self.colors[2],-1)
            #cv2.putText(img,'CSYS:'+str(int(center[0]))+' '+str(int(center[1])),(int(center[0])+10,int(center[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        else:
            print('error:No box is available')
        return img


    def EuclideanDistance(self,c,c0):
        '''
        计算欧氏距离
        @para c(list):[x, y]
        @para c0(list):[x, y]
        @return double:欧氏距离
        '''
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

