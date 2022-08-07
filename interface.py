import sys
import qdarkstyle
import json
import time
from PyQt5.QtWidgets import(
    QMainWindow, QSlider,QApplication,QLabel,
    QWidget,QScrollArea,QVBoxLayout,QHBoxLayout,QTabWidget,
)
from PyQt5.QtCore import Qt,QSize


class Opencv_slider(QMainWindow):
    '''
    使用pyqt6创建的滑动条类
    '''
    def __init__(self,name,mode,color):
        super().__init__() 
        self.change_label = 0                  #初始化相关变量
        self.sld = []
        self.mode = mode
        self.color = color
        self.showMaximized()                   #设置窗口全屏显示
        self.setWindowTitle(name)              #设置窗口的标题
        self.setStyleSheet("QLabel{font-size:18px;}")      #设置字体大小
        self.read_json()                       #读取json文件获取参数
        self.init_ui()                         #初始化ui
        self.slider_create()                   #创建滑动条
    

    def init_ui(self):
        '''
        初始化ui,实例化需要的控件并且建立关系
        '''
        self.centralWidget = QWidget(self)
        self.centralWidget.setObjectName("centralWidget")
        desktop = QApplication.desktop()
        self.desktop_size = QSize(int(desktop.width()),int(desktop.height()*0.9))
        self.horizontalLayout = QHBoxLayout(self.centralWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
 
        self.tabWidget = QTabWidget(self.centralWidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setMinimumSize(self.desktop_size)
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.setCurrentIndex(0)
 
        self.tab_1 = QWidget()
        self.tab_1.setObjectName("tab_1")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName("tab_4")

        self.tabWidget.addTab(self.tab_1, "hsv调参")
        self.tabWidget.addTab(self.tab_2, "能量机关筛选参数")
        self.tabWidget.addTab(self.tab_3, "灯条筛选参数")
        self.tabWidget.addTab(self.tab_4, "装甲板筛选参数")
 
        self.weight_sa_1 = self.widget_contents_create(self.tab_1)
        self.weight_sa_2 = self.widget_contents_create(self.tab_2)
        self.weight_sa_3 = self.widget_contents_create(self.tab_3)
        self.weight_sa_4 = self.widget_contents_create(self.tab_4)
        self.change_label = 1

        self.setCentralWidget(self.centralWidget)
        

    def read_json(self):
        '''
        读取json文件获取滑动条初值和个数
        '''
        if self.mode == 3:
            path = './json/sentry_find.json'
        else:
            path = './json/armor_find.json'
        with open(path,'r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            if self.color == 0:
                armor_hsv_high = load_dict["ImageProcess_red"]["hsvPara_high"]
                armor_hsv_low = load_dict["ImageProcess_red"]["hsvPara_low"]
            elif self.color == 1:
                armor_hsv_high = load_dict["ImageProcess_red"]["hsvPara_high"]
                armor_hsv_low = load_dict["ImageProcess_red"]["hsvPara_low"]
            else:
                armor_hsv_high = [0,0,0]
                armor_hsv_low = [0,0,0]
            self.armor_h_high = armor_hsv_high[0]
            self.armor_s_high = armor_hsv_high[1]
            self.armor_v_high = armor_hsv_high[2]
            self.armor_h_low = armor_hsv_low[0]
            self.armor_s_low = armor_hsv_low[1]
            self.armor_v_low = armor_hsv_low[2]
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
            self.lightBarAreaDiff = load_dict["ArmorFind"]["lightBarAreaDiff"]
            self.armorAngleMin = load_dict["ArmorFind"]["armorAngleMin"]
            self.minarmorArea = load_dict["ArmorFind"]["minarmorArea"]
            self.maxarmorArea = load_dict["ArmorFind"]["maxarmorArea"]
            self.minarmorProp = load_dict["ArmorFind"]["minarmorProp"]
            self.maxarmorProp = load_dict["ArmorFind"]["maxarmorProp"]
            self.minBigarmorProp = load_dict["ArmorFind"]["minBigarmorProp"]
            self.maxBigarmorProp = load_dict["ArmorFind"]["maxBigarmorProp"]
            self.angleDiff_near = load_dict["ArmorFind"]["angleDiff_near"]
            self.angleDiff_far = load_dict["ArmorFind"]["angleDiff_far"]
            self.minareawidthRatio = load_dict["ArmorFind"]["minareawidthRatio"]
            self.maxareawidthRatio = load_dict["ArmorFind"]["maxareawidthRatio"]
            self.minareaRatio = load_dict["ArmorFind"]["minareaRatio"]
            self.maxareaRatio = load_dict["ArmorFind"]["maxareaRatio"]
            self.area_limit = load_dict["ArmorFind"]["area_limit"]
            self.xcenterdismax = load_dict["ArmorFind"]["xcenterdismax"]
            self.ylengthmin = load_dict["ArmorFind"]["ylengthmin"]
            self.ylengcenterRatio = load_dict["ArmorFind"]["ylengcenterRatio"]
            self.yixaingangleDiff_near = load_dict["ArmorFind"]["yixaingangleDiff_near"]
            self.yixaingangleDiff_far = load_dict["ArmorFind"]["yixaingangleDiff_far"]
            #测距
            self.kh = load_dict["ArmorFind"]["kh"]
        with open('./json/Energy_find.json','r',encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f,strict=False)
            if self.color == 0:
                energy_hsv_low =load_dict["hsv"]["hsv_blue_low"]
                energy_hsv_high = load_dict["hsv"]["hsv_blue_high"]
            elif self.color == 1:
                energy_hsv_low = load_dict["hsv"]["hsv_red_low"]
                energy_hsv_high = load_dict["hsv"]["hsv_red_high"] 
            else:
                energy_hsv_high = [0,0,0]
                energy_hsv_low = [0,0,0]
            self.energy_h_high = energy_hsv_high[0]
            self.energy_s_high = energy_hsv_high[1]
            self.energy_v_high = energy_hsv_high[2]
            self.energy_h_low = energy_hsv_low[0]
            self.energy_s_low = energy_hsv_low[1]
            self.energy_v_low = energy_hsv_low[2]
            self.MaxRsS = load_dict["EnergyFind"]["MaxRsS"]
            self.MinRsS = load_dict["EnergyFind"]["MinRsS"]
            self.MaxRsRatio = load_dict["EnergyFind"]["MaxRsRatio"]
            self.fan_armor_distence_max = load_dict["EnergyFind"]["fan_armor_distence_max"]
            self.fan_armor_distence_min = load_dict["EnergyFind"]["fan_armor_distence_min"]
            self.armor_R_distance_max = load_dict["EnergyFind"]["armor_R_distance_max"]
            self.armor_R_distance_min = load_dict["EnergyFind"]["armor_R_distance_min"] 
        #滑动条中英文对照表
        self.name_dict = {
            'armor_h_low':'自瞄_lowHue',
            'armor_s_low':'自瞄_lowSat',
            'armor_v_low':'自瞄_lowVal',
            'armor_h_high':'自瞄_highHue',
            'armor_s_high':'自瞄_highSat',
            'armor_v_high':'自瞄_highVal',
            'energy_h_high':'打符_highHue',
            'energy_s_high':'打符_highSat',
            'energy_v_high':'打符_highVal',
            'energy_h_low':'打符_lowHue',
            'energy_s_low':'打符_lowSat',
            'energy_v_low':'打符_lowVal',

            'MaxRsS':'R的最大面积',
            'MinRsS':'R的最小面积',
            'MaxRsRatio':'R的最大长宽比',
            'fan_armor_distence_max':'装甲板与扇叶中心最大值',
            'fan_armor_distence_min':'装甲板与扇叶中心最小值',
            'armor_R_distance_max':'装甲板与R距离的最大值',
            'armor_R_distance_min':'装甲板与R距离的最小值',

            'minlighterarea':'最小灯条面积',
            'maxlighterarea':'最大灯条面积',
            'minlighterProp':'最小灯条长比宽',
            'maxlighterProp':'最大灯条长比宽',
            'minAngleError':'最小灯条角度',
            'maxAngleError':'最大灯条角度',

            'maxarealongRatio':'灯条长长比最大值',
            'minarealongRatio':'灯条长长比最小值',
            'lightBarAreaDiff':'灯条面积差最大值',
            'armorAngleMin':'装甲板角度最小值',
            'minarmorArea':'装甲板面积最小值',
            'maxarmorArea':'装甲板面积最大值',
            'minarmorProp':'小装甲板长宽比最小值',
            'maxarmorProp':'小装甲板长宽比最大值',
            'minBigarmorProp':'大装甲板长宽比最小值',
            'maxBigarmorProp':'大装甲板长宽比最小值',
            'angleDiff_near':'近距离灯条角度差最大值',
            'angleDiff_far':'远距离灯条角度差最大值',
            'minareawidthRatio':'自瞄_lowHue',
            'maxareawidthRatio':'自瞄_lowHue',
            'minareaRatio':'灯条面积比最小值',
            'maxareaRatio':'灯条面积比最大值',
            'area_limit':'远近距离区分——面积',
            'xcenterdismax':'灯条横向中心距最大值',
            'ylengthmin':'装甲板高度最小值',
            'ylengcenterRatio':'灯条纵向高度差（灯条纵向距离与装甲板纵向长度比）',
            'yixaingangleDiff_near':'近距离灯条异向角度差',
            'yixaingangleDiff_far':'远距离灯条异向角度差',
            'kh':'测距参数',

        }
    
    def widget_contents_create(self,widget):
        '''
        初始化标签页里的滚动条页,返回用于添加滑动条的QWidget
        '''
        widget_btn = QWidget(widget)
        widget_btn.setObjectName("widget_btn")
        widget_btn.adjustSize()
 
        horizontalLayout_3 = QHBoxLayout(widget_btn)
        horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        horizontalLayout_3.setObjectName("horizontalLayout_3")
 
        scrollArea = QScrollArea(widget_btn)
        scrollArea.adjustSize()
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName("scrollArea")
 
        scrollAreaWidgetContents = QWidget(widget_btn)
        scrollAreaWidgetContents.adjustSize()
        scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        
        
        verticalLayout = QVBoxLayout(scrollAreaWidgetContents)
        verticalLayout.setObjectName("verticalLayout")

        scrollArea.setWidget(scrollAreaWidgetContents)
        horizontalLayout_3.addWidget(scrollArea)

        return widget_btn,verticalLayout,scrollAreaWidgetContents
    
    def slider_create(self):
        '''
        创建滑动条
        '''
        self.TrackerBar_create('armor_h_low',self.armor_h_low,255,self.weight_sa_1)
        self.TrackerBar_create('armor_s_low',self.armor_s_low,255,self.weight_sa_1)
        self.TrackerBar_create('armor_v_low',self.armor_v_low,255,self.weight_sa_1)
        self.TrackerBar_create('armor_h_high',self.armor_h_high,255,self.weight_sa_1)
        self.TrackerBar_create('armor_s_high',self.armor_s_high,255,self.weight_sa_1)
        self.TrackerBar_create('armor_v_high',self.armor_v_high,255,self.weight_sa_1)
        self.TrackerBar_create('energy_h_high',self.energy_h_high,255,self.weight_sa_1)
        self.TrackerBar_create('energy_s_high',self.energy_s_high,255,self.weight_sa_1)
        self.TrackerBar_create('energy_v_high',self.energy_v_high,255,self.weight_sa_1)
        self.TrackerBar_create('energy_h_low',self.energy_h_low,255,self.weight_sa_1)
        self.TrackerBar_create('energy_s_low',self.energy_s_low,255,self.weight_sa_1)
        self.TrackerBar_create('energy_v_low',self.energy_v_low,255,self.weight_sa_1)

        self.TrackerBar_create('MaxRsS',self.MaxRsS,10000,self.weight_sa_2)
        self.TrackerBar_create('MinRsS',self.MinRsS,10000,self.weight_sa_2)
        self.TrackerBar_create('MaxRsRatio',self.MaxRsRatio,2000,self.weight_sa_2)
        self.TrackerBar_create('fan_armor_distence_max',self.fan_armor_distence_max,500,self.weight_sa_2)
        self.TrackerBar_create('fan_armor_distence_min',self.fan_armor_distence_min,255,self.weight_sa_2)
        self.TrackerBar_create('armor_R_distance_max',self.armor_R_distance_max,1000,self.weight_sa_2)
        self.TrackerBar_create('armor_R_distance_min',self.armor_R_distance_min,500,self.weight_sa_2)

        self.TrackerBar_create('minlighterarea',self.minlighterarea,255,self.weight_sa_3)
        self.TrackerBar_create('maxlighterarea',self.maxlighterarea,10000,self.weight_sa_3)
        self.TrackerBar_create('minlighterProp',self.minlighterProp,500,self.weight_sa_3)
        self.TrackerBar_create('maxlighterProp',self.maxlighterProp,3000,self.weight_sa_3)
        self.TrackerBar_create('minAngleError',self.minAngleError,3600,self.weight_sa_3)
        self.TrackerBar_create('maxAngleError',self.maxAngleError,3600,self.weight_sa_3)

        self.TrackerBar_create('maxarealongRatio', self.maxarealongRatio, 300, self.weight_sa_4)
        self.TrackerBar_create('minarealongRatio', self.minarealongRatio, 100, self.weight_sa_4)
        self.TrackerBar_create('lightBarAreaDiff', self.lightBarAreaDiff, 10000, self.weight_sa_4)
        self.TrackerBar_create('armorAngleMin', self.armorAngleMin, 3600, self.weight_sa_4)
        self.TrackerBar_create('minarmorArea', self.minarmorArea, 5000, self.weight_sa_4)
        self.TrackerBar_create('maxarmorArea', self.maxarmorArea, 100000, self.weight_sa_4)
        self.TrackerBar_create('minarmorProp', self.minarmorProp, 255, self.weight_sa_4)
        self.TrackerBar_create('maxarmorProp', self.maxarmorProp, 600, self.weight_sa_4)
        self.TrackerBar_create('minBigarmorProp', self.minBigarmorProp, 300, self.weight_sa_4)
        self.TrackerBar_create('maxBigarmorProp', self.maxBigarmorProp, 600, self.weight_sa_4)

        self.TrackerBar_create('angleDiff_near', self.angleDiff_near, 100, self.weight_sa_4)
        self.TrackerBar_create('angleDiff_far', self.angleDiff_far, 100, self.weight_sa_4)
        self.TrackerBar_create('minareawidthRatio', self.minareawidthRatio, 600, self.weight_sa_4)
        self.TrackerBar_create('maxareawidthRatio', self.maxareawidthRatio, 600, self.weight_sa_4)
        self.TrackerBar_create('minareaRatio', self.minareaRatio, 600, self.weight_sa_4)
        self.TrackerBar_create('maxareaRatio', self.maxareaRatio, 600, self.weight_sa_4)
        self.TrackerBar_create('area_limit', self.area_limit, 600, self.weight_sa_4)
        self.TrackerBar_create('xcenterdismax', self.xcenterdismax, 600, self.weight_sa_4)
        self.TrackerBar_create('ylengthmin', self.ylengthmin, 10, self.weight_sa_4)
        self.TrackerBar_create('ylengcenterRatio', self.ylengcenterRatio, 10, self.weight_sa_4)
        self.TrackerBar_create('yixaingangleDiff_near', self.yixaingangleDiff_near, 10, self.weight_sa_4)
        self.TrackerBar_create('yixaingangleDiff_far', self.yixaingangleDiff_far, 10, self.weight_sa_4)
        self.TrackerBar_create('kh', self.kh, 40000, self.weight_sa_4)
    
    def TrackerBar_create(self,name,value,range_max,GroupBox):
        '''
        创建单个滑动条
        '''
        temp_name = self.name_dict[name]

        _,verticalLayout,scrollAreaWidgetContents = GroupBox
        sld = QSlider(Qt.Orientation.Horizontal,scrollAreaWidgetContents)
        sld.setRange(0,range_max)                          #设置滑动条范围
        sld.setValue(int(value))                           #设置滑动条初始值
        sld.setMinimumSize(500,30)
        sld.setMaximumSize(1200,1000)
        sld.adjustSize()
        sld.valueChanged[int].connect(self.changeValue)    #绑定信号，随时更新滑动条的值    
        sld.setTickPosition(sld.TickPosition.TicksBelow)   #左对齐
        
        label = QLabel(str(temp_name)+':'+str(value))           #初始化上标文字
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)     #左对齐
        label.setMinimumSize(500,30)
        label.adjustSize()

        verticalLayout.addWidget(label)
        verticalLayout.addWidget(sld)

        self.sld.append([sld,label,name,value])

    def changeValue(self):
        #值更改时候的回调函数,实时显示对应的滑动条更改
        signalSource = self.sender()
        #寻找对应的label
        for i,s in enumerate(self.sld):
            if s[0] is signalSource:
                label = s[1]
                temp_i = i 
                break
        text_list = str(label.text()).split(':')
        new_name = text_list[0]+':'+str(signalSource.value())
        label.setText(new_name)
        self.sld[temp_i][3] = signalSource.value()
    
    def resizeEvent(self, event):
        '''
        改变窗口大小时触发的回调函数,根据窗口大小重绘
        '''
        if self.change_label:
            windows_size = self.size()
            for temp_sa in [self.weight_sa_1,self.weight_sa_2,self.weight_sa_3,self.weight_sa_4]:
                widget_btn,_,_ = temp_sa
                widget_btn.resize(int(windows_size.width()-10),int(windows_size.height()-30))
    
    def update_json(self):
        '''
        更新json里的值
        '''
        t0 = time.time()
        if self.t1 == None:
            self.t1 = t0
        if self.t1 - t0 > 0.5:
            self.t1 = t0
        ...


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Opencv_slider('Opencv_slider',0,0)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    ex.show()
    app.exec()
