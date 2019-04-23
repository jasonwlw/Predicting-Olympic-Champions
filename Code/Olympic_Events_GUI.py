import sys
from PyQt4 import QtGui,QtCore
from PyQt4.QtCore import SIGNAL

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

import Olympic_Analysis as OA


### TODO: MAKE TEXT IN VARIABLE DESCRIPTION READ ONLY (LOW PRIORITY)
### TODO: DATA TRANSFORMATION TAB TO CONVERT TO BMI WITH VISUALS
### TODO: DATA NORMALIZATION TAB FOR LOGISTIC REGRESSION (MINMAX OR STANDARDIZATION, CHOOSE VARIABLES)
### TODO: CLASS BALANCING TAB FOR CLASS IMBALANCE (CHOOSE HOW TO BALANCE, IF SELECT DIFFERENT ONE CAN JUST RELOAD ENTIRE Y
### TODO: PART OF DATASET)
### TODO: RESEARCH LITERATURE FOR OLYMPIC MEDAL WINNERS



class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        ### Create preprocessing object
        self.prep = OA.PreProcessing()
        #self.df1,self.df2 = self.Read_Data()

        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Olympics")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))

        self.connect(self, SIGNAL('main2closed()'), self.clearVars)

        self.standard = False

        ### For drop down menu
        ### Some variables do not work; non-numeric don't
        #self.var_list = self.prep.df1.columns.values
        #self.var_list = np.insert(self.var_list,0,'')

        ### Need to add way in Olympic_Analysis to plot Medal histogram, it;s important
        self.var_list = ['','Age','Height','Weight','Year','Medal']



        '''
        extractAction = QtGui.QAction("&GET TO THE CHOPPAH!!!", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Leave The App')
        extractAction.triggered.connect(self.close_application)

        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        '''

        self.home()


    def Read_Data(self):
        df1 = pd.read_csv('athlete_events.csv')
        df2 = pd.read_csv('noc_regions.csv')
        return (df1, df2)

    def home(self):

        ### Set main widget
        wid = QtGui.QWidget(self)
        self.setCentralWidget(wid)

        ### Create tab widget
        tabs = QtGui.QTabWidget()

        ### Create tabs

        tab1 = QtGui.QWidget()
        tab2 = QtGui.QWidget()
        tab3 = QtGui.QWidget()
        tab4 = QtGui.QWidget()
        tab5 = QtGui.QWidget()
        tab6 = QtGui.QWidget()

        tabs.resize(250, 150)

        ### tab 1 features (home tab)

        ### Set tab layout
        vBoxlayout = QtGui.QVBoxLayout(tab1)
        vBoxlayout.addStretch(1)

        ### Reset Button to reset dataframe
        btn0 = QtGui.QPushButton("Reset Dataframe",self)
        btn0.clicked.connect(self.Reset_Data)
        btn0.resize(btn0.minimumSizeHint())
        btn0.move(0,50)
        vBoxlayout.addWidget(btn0)


        ### Button to pull up variable description
        btn00 = QtGui.QPushButton("Variable Description",self)
        btn00.clicked.connect(self.Variable_Descrip)
        btn00.resize(btn00.minimumSizeHint())
        btn00.move(0,50)
        vBoxlayout.addWidget(btn00)

        tab1.setLayout(vBoxlayout)

        ### tab2 features (imputation tab)
        vBoxlayout = QtGui.QVBoxLayout(tab2)
        vBoxlayout.addStretch(1)

        ### Impute Mean Ages Button
        btn1 = QtGui.QPushButton("Age Imputation By Country", self)
        btn1.clicked.connect(self.Impute_Age_Mean_Wrap)
        btn1.resize(btn1.minimumSizeHint())
        btn1.move(0, 100)
        vBoxlayout.addWidget(btn1)
        #hBoxLayout = QtGui.QHBoxLayout(tab2)
        #hBoxLayout.addStretch(1)
        #hBoxLayout.addLayout(vBoxlayout)
        vBoxlayout.addStretch(2)

        ### Impute Mean Height Button
        btn2 = QtGui.QPushButton("Height Imputation By Country", self)
        btn2.clicked.connect(self.Impute_Height_Mean_Wrap)
        btn2.resize(btn2.minimumSizeHint())
        btn2.move(0, 100)
        vBoxlayout.addWidget(btn2)
        vBoxlayout.addStretch(3)

        ### Impute Mean Weight Button
        btn3 = QtGui.QPushButton("Weight Imputation By Country", self)
        btn3.clicked.connect(self.Impute_Weight_Mean_Wrap)
        btn3.resize(btn3.minimumSizeHint())
        btn3.move(0, 100)
        vBoxlayout.addWidget(btn3)
        vBoxlayout.addStretch(4)

        ### Impute Mean from entire dataset
        btn4 = QtGui.QPushButton("Impute All By Country Mean")
        btn4.clicked.connect(self.Impute_All_Country_Wrap)
        btn4.resize(btn4.minimumSizeHint())
        btn4.move(0, 100)
        vBoxlayout.addWidget(btn4)
        vBoxlayout.addStretch(5)

        ### Impute Mean Ages Button
        btn1 = QtGui.QPushButton("Age Imputation By Dataset", self)
        btn1.clicked.connect(self.Impute_Age_Dataset_Wrap)
        btn1.resize(btn1.minimumSizeHint())
        btn1.move(0, 100)
        vBoxlayout.addWidget(btn1)
        #hBoxLayout = QtGui.QHBoxLayout(tab2)
        #hBoxLayout.addStretch(1)
        #hBoxLayout.addLayout(vBoxlayout)
        vBoxlayout.addStretch(2)

        ### Impute Mean Height Button
        btn2 = QtGui.QPushButton("Height Imputation", self)
        btn2.clicked.connect(self.Impute_Height_Dataset_Wrap)
        btn2.resize(btn2.minimumSizeHint())
        btn2.move(0, 100)
        vBoxlayout.addWidget(btn2)
        vBoxlayout.addStretch(3)

        ### Impute Mean Weight Button
        btn3 = QtGui.QPushButton("Weight Imputation", self)
        btn3.clicked.connect(self.Impute_Weight_Dataset_Wrap)
        btn3.resize(btn3.minimumSizeHint())
        btn3.move(0, 100)
        vBoxlayout.addWidget(btn3)
        vBoxlayout.addStretch(4)

        ### Impute with Country Averages
        btn5 = QtGui.QPushButton("Impute All By Dataset Mean")
        btn5.clicked.connect(self.Impute_All_Dataset_Wrap)
        btn5.resize(btn5.minimumSizeHint())
        btn5.move(0, 100)
        vBoxlayout.addWidget(btn5)
        vBoxlayout.addStretch(6)

        self.progress = QtGui.QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        vBoxlayout.addWidget(self.progress)




        tab2.setLayout(vBoxlayout)

        ### tab3 (feature selection)
        vBoxlayout = QtGui.QVBoxLayout(tab3)

        btn30 = QtGui.QPushButton("Plot Correlations")
        btn30.clicked.connect(self.Plot_Corr)
        btn30.resize(btn30.minimumSizeHint())
        btn30.move(0,100)
        vBoxlayout.addWidget(btn30)

        tab3.setLayout(vBoxlayout)

        ###tab 4 (data transformation)
        vBoxlayout = QtGui.QVBoxLayout(tab4)

        btn40 = QtGui.QPushButton("Standardize Variables")
        btn40.clicked.connect(self.Standardize)
        btn40.resize(btn40.minimumSizeHint())
        btn40.move(0,100)
        vBoxlayout.addWidget(btn40)


        btn41 = QtGui.QPushButton("Enter Year Split")
        btn41.clicked.connect(self.getint)
        self.le41 = 1990
        vBoxlayout.addWidget(btn41)
        #vBoxlayout.addWidget(le41)

        tab4.setLayout(vBoxlayout)

        ### tab5 (visualization tab)

        vBoxlayout = QtGui.QVBoxLayout(tab5)

        ### Drop Down Menu for variables
        combo = QtGui.QComboBox(self)
        combo.currentIndexChanged.connect(self._cb_currentIndexChanged)
        combo.addItems(self.var_list)
        vBoxlayout.addWidget(combo)

        ### Plot button
        btn50 = QtGui.QPushButton("Plot Current Selection")
        btn50.clicked.connect(self.Plot_Var)
        btn50.resize(btn50.minimumSizeHint())
        btn50.move(0,100)
        vBoxlayout.addWidget(btn50)

        ### Plot canvas
        #self.figure = Figure()
        #self.canvas = FigureCanvas(self.figure)
        #vBoxlayout.addWidget(self.canvas)

        tab5.setLayout(vBoxlayout)

        ### tab 6 (modeling)
        vBoxlayout = QtGui.QVBoxLayout(tab6)

        btn60 = QtGui.QPushButton("Decision Tree")
        btn60.clicked.connect(self.Decision_Tree)
        btn60.resize(btn60.minimumSizeHint())
        btn60.move(0,100)
        vBoxlayout.addWidget(btn60)

        btn61 = QtGui.QPushButton("Logistic Regression")
        btn61.clicked.connect(self.Log_Reg)
        btn61.resize(btn61.minimumSizeHint())
        btn61.move(0,100)
        vBoxlayout.addWidget(btn61)

        self.progress_model = QtGui.QProgressBar(self)
        self.progress_model.setGeometry(200, 80, 250, 20)
        vBoxlayout.addWidget(self.progress_model)

        tab6.setLayout(vBoxlayout)

        ### Add tabs to widget
        tabs.addTab(tab1, "Home")
        tabs.addTab(tab2, "Imputation")
        tabs.addTab(tab3, "Feature Selection")
        tabs.addTab(tab4, "Data Transformation")
        tabs.addTab(tab5, "Visualization")
        tabs.addTab(tab6, "Modeling")


        ### NEED TO FIGURE OUT HOW TO DO PROGRESS BARS WITH FUNCTIONS IN MODULE
        #self.progress = QtGui.QProgressBar(self)
        #self.progress.setGeometry(200, 80, 250, 20)

        mainlayout = QtGui.QVBoxLayout()
        mainlayout.addWidget(tabs)

        wid.setLayout(mainlayout)

        self.show()


    def Reset_Data(self):
        self.prep = OA.PreProcessing()

    def Variable_Descrip(self):
        self.other_window = Variables()
        self.connect(self.other_window, SIGNAL('closed()'), self.VarsClosed)
        self.other_window.show()

    def VarsClosed(self):
        self.emit(SIGNAL('main2closed()'))

    def clearVars(self):
        del self.other_window
        self.other_window = None

    def Impute_Age_Mean_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,0)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Impute_Weight_Mean_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,1)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Impute_Height_Mean_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,2)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Impute_All_Country_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,3)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Impute_Age_Dataset_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,4)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Impute_Weight_Dataset_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,5)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Impute_Height_Dataset_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,6)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Impute_All_Dataset_Wrap(self):
        self.progress.setRange(0,0)
        self.Imputation = TaskThread(self.prep,7)
        self.Imputation.taskFinished.connect(self.Imputation_Finished)
        self.Imputation.start()

    def Imputation_Finished(self):
        self.progress.setRange(0,1)

    def _cb_currentIndexChanged(self, idx):
        self.idx = idx
        self.plot_data = self.prep.Plot_Histogram(self.var_list[idx])

    def Plot_Var(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.axes.clear()
        self.figure.axes.append(self.plot_data)
        #self.figure.canvas.draw()
        plt.xlabel(self.var_list[self.idx])
        plt.title(self.var_list[self.idx])
        plt.show()

    def Plot_Corr(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.axes.clear()
        self.plot_data = self.prep.Plot_Corr()
        self.figure.axes.append(self.plot_data)
        plt.show()

    def Standardize(self):
        self.standard = True

    def getint(self):
        num, ok = QtGui.QInputDialog.getDouble(self, "integer input dialog", "enter a number")

        if ok:
            self.le2 = num
            self.prep.Drop_Years(self.le2)

    def Decision_Tree(self):
        self.progress_model.setRange(0,0)
        self.prep.Train_Test_Split()
        if self.standard:
            self.prep.Standardize()
        self.Modeling = TaskThread_Model(self.prep,0)
        self.Modeling.taskFinished.connect(self.Modeling_Finished)
        self.Modeling.start()

    def Log_Reg(self):
        self.progress_model.setRange(0, 0)
        self.prep.Train_Test_Split()
        if self.standard:
            self.prep.Standardize()
        self.Modeling = TaskThread_Model(self.prep,1)
        self.Modeling.taskFinished.connect(self.Modeling_Finished)
        self.Modeling.start()

    def Modeling_Finished(self):
        self.progress_model.setRange(0,1)

    def close_application(self):
        print("whooaaaa so custom!!!")
        sys.exit()

class Variables(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setGeometry(50,50,500,300)
        self.button = QtGui.QPushButton('Quit', self)
        self.connect(self.button, SIGNAL('clicked()'), self.close)
        self.button.move(0,50)
        self.text_edit = QtGui.QTextEdit()
        self.setCentralWidget(self.text_edit)
        self.text = open('scratch.txt','r').read()
        self.text_edit.setText(self.text)

class TaskThread(QtCore.QThread):
    taskFinished = QtCore.pyqtSignal()
    def __init__(self,prep_obj,imp_type):
        super(QtCore.QThread,self).__init__()
        self.prep = prep_obj
        self.imp_type = imp_type

    def run(self):
        if self.imp_type == 0:
            self.prep.Impute_Age_Mean()
        elif self.imp_type == 1:
            self.prep.Impute_Weight_Mean()
        elif self.imp_type == 2:
            self.prep.Impute_Height_Mean()
        elif self.imp_type == 3:
            self.prep.Impute_All_Country()
        elif self.imp_type == 4:
            self.prep.Impute_Age_Dataset()
        elif self.imp_type == 5:
            self.prep.Impute_Weight_Dataset()
        elif self.imp_type == 6:
            self.prep.Impute_Height_Dataset()
        elif self.imp_type == 7:
            self.prep.Impute_All_Dataset()
        self.taskFinished.emit()

class TaskThread_Model(QtCore.QThread):
    taskFinished = QtCore.pyqtSignal()
    def __init__(self,prep_obj,model_type):
        super(QtCore.QThread,self).__init__()
        self.prep = prep_obj
        self.model_type = model_type

    def run(self):
        if self.model_type == 0:
            self.prep.Decision_Tree()
        elif self.model_type == 1:
            self.prep.Log_Reg()
        self.taskFinished.emit()


def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()