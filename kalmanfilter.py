from filterpy.kalman import KalmanFilter
import numpy as np
#对卡尔曼滤波的简单封装，方便调用

class KalmanBoxTracker:
    def __init__(self,bbox,img=None):
        self.kf = KalmanFilter(dim_x=10, dim_z=5)
        # 10个状态变量[x,y,short,angle,long,vx,vy,vs,vangle,vlong]，5个观测输入[x,y,long,short,angle]
        self.kf.F = np.array([[1,0,0,0,0,1,0,0,0,0],
                                [0,1,0,0,0,0,1,0,0,0],
                                [0,0,1,0,0,0,0,1,0,0],
                                [0,0,0,1,0,0,0,0,1,0],  
                                [0,0,0,0,1,0,0,0,0,1],
                                [0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,0,0,1]
                                ])
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0]
                                ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[5:,5:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[5:,5:] *= 0.01

        self.kf.x[:5] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.history = []
        self.angle_history = []
        self.hit_streak = 0

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        if bbox != []:
            self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        min_ = 0
        if((self.kf.x[7]+self.kf.x[2])<=min_): # 面积太小，即将消亡
            self.kf.x[0] = -1
        self.kf.predict()
        if(self.time_since_update>0):
            self.hit_streak = 0
            self.time_since_update += 1
        if len(self.history) == 10:
            self.history = self.history[1:10]
        if len(self.angle_history) == 10:
            self.angle_history = self.angle_history[1:10]
        history = self.convert_x_to_bbox(self.kf.x)
        self.history.append(history)
        self.angle_history.append(history[0][-1])
        his = self.history[-1][0]
        return [his[0],his[1],his[2],his[3],his[4]]

    def convert_bbox_to_z(self,bbox):
        """
        lightbar:[x,y,long,short,angle]
        s = long*short
        去掉ratio，因为short比较不稳定
        """
        x = bbox[0]
        y = bbox[1]
        s = bbox[3]
        long = bbox[2]
        angle = bbox[4]
        return np.array([x,y,s,angle,long]).reshape((5,1))

    def convert_x_to_bbox(self,out):
        """
        Takes a bounding box in the centre form [x,y,s,angle,long] and returns it in the form
        [x,y,long,short,angle] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        x = out[0]
        y = out[1]
        long = out[4]
        short = out[2]
        angle = out[3]
        return np.array([x,y,long,short,angle]).reshape((1,5))

