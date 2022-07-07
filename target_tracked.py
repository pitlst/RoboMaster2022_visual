import json
from kalmanfilter import KalmanBoxTracker

class Tracker:
    def __init__(self):
        self.read_json()
        self.history_armor = []
        self.trackers = []
        self.armor_state = [] # 每条轨迹的状态：旋转、平移、闪烁（闪烁包括丢失，对灯条无需区分）
    
    def read_json(self):
        #读取json文件中的参数
        with open('./json/armor_find.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            self.kh = load_dict["ArmorFind"]["range_kh"]
        with open('./json/common.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            width = load_dict["Aimbot"]["width"]
            height = load_dict["Aimbot"]["height"]
            self.conf = load_dict["kalman_filter"]["conf"]
            self.delay_fps = load_dict["kalman_filter"]["delay_fps"]
            self.off_center_fps = load_dict["kalman_filter"]["off_center_fps"]
            self.center = [width/2,height/2]

            
    def associate_trk_to_lb(self, lightbars, armor_pred):
        # 给历史装甲打上标记（matched，unmatched，died）
        matched_indexes = [[],[]]
        #遍历历史装甲板
        for am,armor in enumerate(armor_pred):
            if armor == []:
                break
            lb = armor[0]
            rb = armor[1]
            lb_matched = False
            rb_matched = False
            lb_matched_score = 999
            rb_matched_score = 999
            lb_matched_index = -1
            rb_matched_index = -1 
            #遍历当前帧下的灯条
            for i, ltb in enumerate(lightbars):
                #对灯条列表长度做筛选
                if len(ltb) != 6:
                    break
                #获得两灯条的相似度分数，越小代表越相似
                score = self.get_sorce(lb,ltb)
                if score < lb_matched_score:
                    lb_matched_index = i
                    lb_matched_score = score
                score = self.get_sorce(rb,ltb)
                if score < rb_matched_score:
                    rb_matched_index = i
                    rb_matched_score = score

                #如果找到一个当前灯条与历史左灯条和历史右灯条都最相近，通过score判断与谁更相近
                if lb_matched_score < self.conf and rb_matched_score < self.conf and lb_matched_index == rb_matched_index:
                    if lb_matched_score < rb_matched_score:
                        rb_matched_score = 999
                    else:
                        lb_matched_score = 999
            #小于即代表找到了对应匹配的灯条
            if lb_matched_score < self.conf:
                lb_matched = True
            if rb_matched_score < self.conf:
                rb_matched = True

            #给灯条打上标记
            if lb_matched:
                lightbars[lb_matched_index][5] = 0
                if rb_matched:
                    lightbars[rb_matched_index][5] = 1
                    matched_indexes[0].append(lb_matched_index)
                    matched_indexes[1].append(rb_matched_index)
                    armor_pred[am][2] = 0
                    # 替换预测值为观测值
                    armor_pred[am][0] = lightbars[lb_matched_index]
                    armor_pred[am][1] = lightbars[rb_matched_index]
                else:
                    lightbars[lb_matched_index][5] = 2
                    lightbars[rb_matched_index][5] = 3
                    armor_pred[am][2] = 2
            else:
                if rb_matched:
                    lightbars[rb_matched_index][5] = 2
                    lightbars[lb_matched_index][5] = 3
                    armor_pred[am][2] = 2
        return armor_pred, matched_indexes, lightbars
    

    def combine_armor(self, lightbars, matched_indexes):
        # 对观测灯条浅浅组合一下,返回与历史无关的新装甲
        armor_combined = []
        screen_lightbars = []
        #只对历史找得到相对应匹配灯条的灯条进行新装甲匹配
        for i, lightbar in enumerate(lightbars):
            if (i in matched_indexes[0] or matched_indexes[1]) or not lightbar[-1]:
                screen_lightbars.append[lightbar]

        if len(screen_lightbars):
            for i in range(len(screen_lightbars)):
                for j in range(i+1, len(screen_lightbars)):
                    #判断装甲板合不合格
                    now_armor = self.judge_armor_right(screen_lightbars[i], screen_lightbars[j])
                    if now_armor:
                        #判断灯条左右，对灯条排序
                        if screen_lightbars[i][0]-screen_lightbars[j][0] < 0:
                            armor_combined.append([screen_lightbars[i], screen_lightbars[j]])
                        else:
                            armor_combined.append([screen_lightbars[j], screen_lightbars[i]])

        return armor_combined

    
    def judge_armor_right(self, lb, rb):
        #判断装甲板合不合格
        x0, y0, l0, s0, a0 = lb[0:5]
        x1, y1, l1, s1, a1 = rb[0:5]
        center_dis = self.EuclideanDistance([x0,y0],[x1,y1])
        angle_err = abs(a0-a1)
        if a0*a1 < 0 and abs(a0)>1 and abs(a1)>1:
            angle_err *= 2
        l_err = max(l1,l0)/(min(l1,l0)+0.00001)
        y_dis = abs(y0-y1)
        now_armor = True
        if y_dis/max(l0,l1)*11 > 9:
            now_armor = False
        elif angle_err*max(l0,l1)/100 > 9:
            now_armor = False
        elif l_err > 7:
            now_armor = False
        elif center_dis < 1*max(l0,l1):
            now_armor = False
        elif center_dis > 5.3*max(l0,l1):
            now_armor = False        
        return now_armor



    def get_final_armors(self, armor_his, armor_now):
        # 找到最终两个armor进入维护，并且重新组建数据结构，方便kalman和tracker更新
        # [lb0,lb1,matched/unmatched/single_unmatched,flash_time,off_time]
        lb = []
        rb = []
        flash_time = 0  # 闪烁时长
        off_time = 0    # 偏心时长
        flag = False
        
        died_x = -1
        died_y = -1
        gyro_died = False
       
        if len(armor_his) and len(armor_his[0]):
            xc = (armor_his[0][0][0]+armor_his[0][1][0])/2 #装甲板中心x
            yc = (armor_his[0][0][1]+armor_his[0][1][1])/2 #装甲板中心y
            len_x = abs(armor_his[0][0][0]-armor_his[0][1][0]) #装甲板x方向长度
            off_center = self.judge_off_center([xc, yc], len_x)
            if off_center:
                armor_his[0][4] += 1
            state = armor_his[0][2]
            gyro_flag, l_dying, r_dying = False, False, False #更新偏心状态，判断是否会消亡
            if state == 0:
                # print("state",state)
                # if matched
                if l_dying and r_dying:
                    # 两条都死，应该是搞错了，哈哈
                    # 直接pass
                    pass
                elif not l_dying and not r_dying:
                    # 正常
                    # 偏心判断
                    
                    armor_his[0][3] = 0 # 将丢失清零
                    # print("off_time", armor_his[0][4])
                    if 1:#armor_his[0][4] < self.off_center_fps:
                        lb = armor_his[0][0]
                        rb = armor_his[0][1]
                        flash_time = armor_his[0][3]
                        # 再判断一次装甲合不合规矩
                        flag = self.judge_armor_right(lb, rb)
                        #print("flag:",flag)
                else:
                    # 陀螺消亡
                    died_x = xc
                    died_y = yc
                    gyro_died = True
            elif state == 1:
                if armor_his[0][3] > self.delay_fps:
                    pass
                else:
                    if not l_dying and not r_dying:
                        armor_his[0][3] += 1
                        lb = armor_his[0][0]
                        rb = armor_his[0][1]
                        flash_time = armor_his[0][3]
                        off_time = self.off_center_fps
                        flag = True
                    elif l_dying and r_dying:
                        pass
                    else:
                        died_x = xc
                        died_y = yc
                        gyro_died = True
            elif state == 2:
                # 只匹配到单个，基本是被遮挡了，或者转动消失，judge_lb_state出现了失效
                # 考虑一下judge_lb_state比较迟钝的情况,没有提前判断出陀螺要消亡
                died_x = xc
                died_y = yc
                gyro_died = True
        if flag:
            if state == 0:
                return 1, lb, rb, flash_time, off_time
            else:
                return 2, lb, rb, flash_time, off_time
        else:
            min_dis = 99999
            target_index = -1
            if gyro_died: # 找一下离上块跟踪装甲最近的
                for i,armor in enumerate(armor_now):
                    xc0 = (armor[0][0]+armor[1][0])/2
                    yc0 = (armor[0][1]+armor[1][1])/2
                    dis = self.EuclideanDistance([xc0,yc0], [died_x,died_y])
                    if dis < min_dis:
                        min_dis = dis
                        target_index = i
            else: # 找离中心最近的
                for j,armor in enumerate(armor_now):
                    xc0 = (armor[0][0]+armor[1][0])/2
                    yc0 = (armor[0][1]+armor[1][1])/2
                    dis = self.EuclideanDistance([xc0,yc0], self.center)
                    if dis < min_dis:
                        min_dis = dis
                        target_index = j
            if target_index != -1:
                return 3, armor_now[target_index][0], armor_now[target_index][1], 0, 0
            else:
                return 0,0,0,0,0
    
    def get_sorce(self,lightbar_history,lightbars_new):
        #获取两灯条的相似分数,不区分新旧装甲板
        x0, y0, l0, s0, a0 = lightbar_history[0:5]
        x1, y1, l1, s1, a1 = lightbars_new[0:5]
        pixel_dis = self.EuclideanDistance([x0, y0], [x1, y1])/(l0+l1)      
        long_ratio = max(l1/(l0+l1), l0/(l0+l1))-0.5                          
        delta_angle = abs(a0-a1)/10 
        score = ((1+pixel_dis)**3)*(1+long_ratio)*(1+delta_angle)-1
        return score

    def judge_off_center(self, c, scale):
        '''
        判断偏心，如果这里要做得特别精准，还是需要反投影把提前量加在像素坐标上
        '''
        distance = self.EuclideanDistance(c, self.center)
        if distance/(scale+0.01) < 2:
            return False
        else:
            return True

    def update(self, lightbars):
        armor_pred = []
        for i, armor in enumerate(self.trackers):
            lb_trk = armor[0]
            rb_trk = armor[1]
            arr = [lb_trk.predict(), rb_trk.predict(), 1, armor[3], armor[4]]
            armor_pred.append(arr)
        if not len(self.trackers):
            armor_pred.append([])
        
        # armor_pred 为kalman给出的历史装甲左右灯条的预测值
        # 灯条观测值和kalman先匹配，如果匹配，打上标记（matched，unmatched，single_matched,single_unmatched)
        # 给历史装甲打上标记（matched，unmatched，died）
        armor_pred_new, matched_indexes, new_lightbars = self.associate_trk_to_lb(lightbars, armor_pred)       
        # 对观测灯条浅浅组合一下,返回与历史无关的新装甲
        armor_combined = self.combine_armor(new_lightbars, matched_indexes)
        # 找到最终跟随目标，优先考虑历史matched，判断历史unmatched是否超过时限
        flag, lb, rb, t0, t1 = self.get_final_armors(armor_pred_new, armor_combined)
        if flag == 1:
            #update matched trackers
            self.trackers[0][0].update(lb[0:5])
            self.trackers[0][1].update(rb[0:5])
            self.trackers[0][2] = 0
            self.trackers[0][3] = t0
            self.trackers[0][4] = t1
        elif flag == 2:
            self.trackers[0][0].update(lb[0:5])
            self.trackers[0][1].update(rb[0:5])
            self.trackers[0][2] = 1
            self.trackers[0][3] = t0
            self.trackers[0][4] = t1
        elif flag == 3:
            self.trackers = []
            trk1 = KalmanBoxTracker(lb[0:5])
            trk2 = KalmanBoxTracker(rb[0:5])
            self.trackers.append([trk1,trk2,0,0,0])
        else:
            self.trackers = []
            return -1,-1,-1
        x = (lb[0]+rb[0])/2
        y = (lb[1]+rb[1])/2
        z = self.GetArmorDistance(min(lb[2],rb[2]),max(lb[2],rb[2]))
        return x, y, z

    def GetArmorDistance(self,dShortside,dLongside):
        #单目视觉测距，简单的根据装甲板最长边的对应像素去计算
        #注意，调参时需要在hsv确定后再调整测距
        height = dShortside + dLongside
        distance = self.kh/height
        return distance

    def EuclideanDistance(self,c,c0):
        '''
        计算欧氏距离
        @para c(list):[x, y]
        @para c0(list):[x, y]
        @return double:欧氏距离
        '''
        return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)