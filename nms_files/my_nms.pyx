def my_nms(p,nms_distence_max):
    '''
    因为大符的预测框很难有较大交错，所以直接按类和中心距离nms即可，我们的nms没有iou计算
    my_nms既做了nms，也筛出了唯一的中心坐标和待击打坐标
    注意，待打击点是会有坐标突变的，但是中心不会，因此，中心坐标如果突变，需要筛去
    '''
    result = []
    center = []
    for i in p[0]:
        cls = get_cls(i)
        if cls == 2:
            #对中心做nms
            if len(center) == 0:
                center.append(i)
            else:
                add = True
                for t,center_t in enumerate(center):
                    if (i[0]-center_t[0])**2+(i[1]-center_t[1])**2 < nms_distence_max**2:
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
                    cls_j = get_cls(j)
                    if cls_j == cls:
                        #nms 非极大值抑制
                        if (i[0]-j[0])**2+(i[1]-j[1])**2 < nms_distence_max**2:
                            add = False
                            if j[4] < i[4]:
                                result[table] = i
                                break
                if add:
                    result.append(i)
    return center,result


def get_cls(det):
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