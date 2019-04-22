import sys
from PyQt4 import QtGui,QtCore
from PyQt4.QtCore import SIGNAL
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

import Olympic_Analysis as OA


### TODO: LOAD TEXT OF VARIABLE DESCRIPTION INTO WINDOW
### TODO: ADD MODELING TAB
### TODO: ADD BEFORE AND AFTER HISTOGRAMS OF DATA FOR IMPUTATION
### TODO: ADD FEATURE SELECTION TAB, VISUALS SHOWING HOW DATASET CHANGES
### TODO: FIGURE OUT PROGRESS BARS FOR IMPUTATION
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

        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("PyQT tuts!")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))

        self.connect(self, SIGNAL('main2closed()'), self.clearVars)

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
        btn1 = QtGui.QPushButton("Age Imputation", self)
        btn1.clicked.connect(self.prep.Impute_Age_Mean)
        btn1.resize(btn1.minimumSizeHint())
        btn1.move(0, 100)
        vBoxlayout.addWidget(btn1)
        #hBoxLayout = QtGui.QHBoxLayout(tab2)
        #hBoxLayout.addStretch(1)
        #hBoxLayout.addLayout(vBoxlayout)
        vBoxlayout.addStretch(2)

        ### Impute Mean Height Button
        btn2 = QtGui.QPushButton("Height Imputation", self)
        btn2.clicked.connect(self.prep.Impute_Height_Mean)
        btn2.resize(btn2.minimumSizeHint())
        btn2.move(0, 100)
        vBoxlayout.addWidget(btn2)
        vBoxlayout.addStretch(3)

        ### Impute Mean Weight Button
        btn3 = QtGui.QPushButton("Weight Imputation", self)
        btn3.clicked.connect(self.prep.Impute_Weight_Mean)
        btn3.resize(btn3.minimumSizeHint())
        btn3.move(0, 100)
        vBoxlayout.addWidget(btn3)

        tab2.setLayout(vBoxlayout)

        ### Impute Mean Age from entire dataset

        ### Add tabs to widget
        tabs.addTab(tab1, "Home")
        tabs.addTab(tab2, "Imputation")
        tabs.addTab(tab3, "Tab 3")
        tabs.addTab(tab4, "Tab 4")

        #tabs.show()

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
        text_edit = QtGui.QPlainTextEdit()
        text = open('scratch.txt').read()
        text_edit.setPlainText(text)


def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()