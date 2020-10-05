# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import time
import os
import sys

from multiprocessing import Process, Pipe, cpu_count, Array
from ctypes import c_double
from tqdm import tqdm
from matplotlib import pyplot as plt

#own imports
from BH import Particle, Cell, Tree, BHF_kickstart, connection_receiveAndClose, processes_joinAndTerminate
from ODEInt import leapfrog
from Animator import AnimateOrbit
from GalGen import generate


if __name__ == "__main__":
    def particles2arr(particles):
        r = np.array([p.r for p in particles])
        v = np.array([p.v for p in particles])
        return r,v
    
    def updateparticles(r,v, particles):
        for indx,p in enumerate(particles):
            p.r = r[indx]
            p.v = v[indx]
        return particles

    def GetSituation(r,colors):
        plt.figure(figsize=(10,10))
        plt.scatter([p.r[0] for p in particles],[p.r[1] for p in particles],color=colors,s=0.4)
        plt.grid()
        plt.ylim(-400,400)
        plt.xlim(-400,400)
        plt.show()

    class Ui_MainWindow(object):
        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(800, 600)
            palette = QtGui.QPalette()
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(170, 127, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(127, 63, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(42, 0, 127))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(56, 0, 170))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(170, 127, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(170, 127, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(127, 63, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(42, 0, 127))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(56, 0, 170))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(170, 127, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
            brush = QtGui.QBrush(QtGui.QColor(42, 0, 127))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(170, 127, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(127, 63, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(42, 0, 127))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(56, 0, 170))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(42, 0, 127))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(42, 0, 127))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(85, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
            MainWindow.setPalette(palette)
            font = QtGui.QFont()
            font.setPointSize(12)
            font.setBold(True)
            font.setWeight(75)
            MainWindow.setFont(font)
            MainWindow.setAutoFillBackground(False)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.label = QtWidgets.QLabel(self.centralwidget)
            self.label.setGeometry(QtCore.QRect(170, 20, 471, 91))
            self.label.setBaseSize(QtCore.QSize(17, 0))
            self.label.setMouseTracking(True)
            self.label.setFocusPolicy(QtCore.Qt.NoFocus)
            self.label.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.label.setFrameShadow(QtWidgets.QFrame.Plain)
            self.label.setLineWidth(15)
            self.label.setTextFormat(QtCore.Qt.AutoText)
            self.label.setScaledContents(False)
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label.setWordWrap(False)
            self.label.setIndent(-5)
            self.label.setObjectName("label")
            self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit.setGeometry(QtCore.QRect(200, 200, 121, 21))
            self.lineEdit.setObjectName("lineEdit")
            self.label_2 = QtWidgets.QLabel(self.centralwidget)
            self.label_2.setGeometry(QtCore.QRect(160, 150, 191, 41))
            self.label_2.setBaseSize(QtCore.QSize(17, 0))
            self.label_2.setMouseTracking(True)
            self.label_2.setFocusPolicy(QtCore.Qt.NoFocus)
            self.label_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.label_2.setFrameShadow(QtWidgets.QFrame.Plain)
            self.label_2.setLineWidth(15)
            self.label_2.setTextFormat(QtCore.Qt.AutoText)
            self.label_2.setScaledContents(False)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setWordWrap(False)
            self.label_2.setIndent(-5)
            self.label_2.setObjectName("label_2")
            self.label_3 = QtWidgets.QLabel(self.centralwidget)
            self.label_3.setGeometry(QtCore.QRect(160, 200, 31, 21))
            self.label_3.setBaseSize(QtCore.QSize(17, 0))
            self.label_3.setMouseTracking(True)
            self.label_3.setFocusPolicy(QtCore.Qt.NoFocus)
            self.label_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.label_3.setFrameShadow(QtWidgets.QFrame.Plain)
            self.label_3.setLineWidth(15)
            self.label_3.setTextFormat(QtCore.Qt.AutoText)
            self.label_3.setScaledContents(False)
            self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            self.label_3.setWordWrap(False)
            self.label_3.setIndent(-5)
            self.label_3.setObjectName("label_3")
            self.label_4 = QtWidgets.QLabel(self.centralwidget)
            self.label_4.setGeometry(QtCore.QRect(160, 220, 31, 21))
            self.label_4.setBaseSize(QtCore.QSize(17, 0))
            self.label_4.setMouseTracking(True)
            self.label_4.setFocusPolicy(QtCore.Qt.NoFocus)
            self.label_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.label_4.setFrameShadow(QtWidgets.QFrame.Plain)
            self.label_4.setLineWidth(15)
            self.label_4.setTextFormat(QtCore.Qt.AutoText)
            self.label_4.setScaledContents(False)
            self.label_4.setAlignment(QtCore.Qt.AlignCenter)
            self.label_4.setWordWrap(False)
            self.label_4.setIndent(-5)
            self.label_4.setObjectName("label_4")
            self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_2.setGeometry(QtCore.QRect(200, 220, 121, 21))
            self.lineEdit_2.setObjectName("lineEdit_2")
            self.label_5 = QtWidgets.QLabel(self.centralwidget)
            self.label_5.setGeometry(QtCore.QRect(120, 260, 71, 21))
            self.label_5.setBaseSize(QtCore.QSize(17, 0))
            self.label_5.setMouseTracking(True)
            self.label_5.setFocusPolicy(QtCore.Qt.NoFocus)
            self.label_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.label_5.setFrameShadow(QtWidgets.QFrame.Plain)
            self.label_5.setLineWidth(15)
            self.label_5.setTextFormat(QtCore.Qt.AutoText)
            self.label_5.setScaledContents(False)
            self.label_5.setAlignment(QtCore.Qt.AlignCenter)
            self.label_5.setWordWrap(False)
            self.label_5.setIndent(-5)
            self.label_5.setObjectName("label_5")
            self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_3.setGeometry(QtCore.QRect(200, 260, 121, 21))
            self.lineEdit_3.setObjectName("lineEdit_3")
            self.label_6 = QtWidgets.QLabel(self.centralwidget)
            self.label_6.setGeometry(QtCore.QRect(160, 240, 31, 21))
            self.label_6.setBaseSize(QtCore.QSize(17, 0))
            self.label_6.setMouseTracking(True)
            self.label_6.setFocusPolicy(QtCore.Qt.NoFocus)
            self.label_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.label_6.setFrameShadow(QtWidgets.QFrame.Plain)
            self.label_6.setLineWidth(15)
            self.label_6.setTextFormat(QtCore.Qt.AutoText)
            self.label_6.setScaledContents(False)
            self.label_6.setAlignment(QtCore.Qt.AlignCenter)
            self.label_6.setWordWrap(False)
            self.label_6.setIndent(-5)
            self.label_6.setObjectName("label_6")
            self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_4.setGeometry(QtCore.QRect(200, 240, 121, 21))
            font = QtGui.QFont()
            font.setPointSize(12)
            font.setBold(True)
            font.setWeight(75)
            self.lineEdit_4.setFont(font)
            self.lineEdit_4.setObjectName("lineEdit_4")
            self.label_7 = QtWidgets.QLabel(self.centralwidget)
            self.label_7.setGeometry(QtCore.QRect(420, 150, 191, 41))
            self.label_7.setBaseSize(QtCore.QSize(17, 0))
            self.label_7.setMouseTracking(True)
            self.label_7.setFocusPolicy(QtCore.Qt.NoFocus)
            self.label_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.label_7.setFrameShadow(QtWidgets.QFrame.Plain)
            self.label_7.setLineWidth(15)
            self.label_7.setTextFormat(QtCore.Qt.AutoText)
            self.label_7.setScaledContents(False)
            self.label_7.setAlignment(QtCore.Qt.AlignCenter)
            self.label_7.setWordWrap(False)
            self.label_7.setIndent(-5)
            self.label_7.setObjectName("label_7")
            self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
            self.progressBar.setGeometry(QtCore.QRect(200, 430, 331, 23))
            self.progressBar.setProperty("value", 0)
            self.progressBar.setInvertedAppearance(False)
            self.progressBar.setObjectName("progressBar")
            self.lcdNumber = QtWidgets.QLCDNumber(self.centralwidget)
            self.lcdNumber.setGeometry(QtCore.QRect(93, 430, 101, 23))
            self.lcdNumber.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.lcdNumber.setFrameShadow(QtWidgets.QFrame.Plain)
            self.lcdNumber.setSmallDecimalPoint(True)
            self.lcdNumber.setDigitCount(8)
            self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Outline)
            self.lcdNumber.setProperty("value", 0.0)
            self.lcdNumber.setProperty("intValue", 0)
            self.lcdNumber.setObjectName("lcdNumber")
            self.pushButton = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton.setGeometry(QtCore.QRect(430, 200, 171, 81))
            self.pushButton.setObjectName("pushButton")
            self.pushButton.clicked.connect(self.clicked)
            MainWindow.setCentralWidget(self.centralwidget)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
            self.menubar.setObjectName("menubar")
            self.menuWindow = QtWidgets.QMenu(self.menubar)
            self.menuWindow.setObjectName("menuWindow")
            MainWindow.setMenuBar(self.menubar)
            self.menubar.addAction(self.menuWindow.menuAction())

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            self.label.setText(_translate("MainWindow", "Interaction window for simulating galactic mergers"))
            self.lineEdit.setText(_translate("MainWindow", "0.5"))
            self.label_2.setText(_translate("MainWindow", "Parameters: "))
            self.label_3.setText(_translate("MainWindow", "θ"))
            self.label_4.setText(_translate("MainWindow", "dt"))
            self.lineEdit_2.setText(_translate("MainWindow", "0.005"))
            self.label_5.setText(_translate("MainWindow", "Frames"))
            self.lineEdit_3.setText(_translate("MainWindow", "500"))
            self.label_6.setText(_translate("MainWindow", "Np"))
            self.lineEdit_4.setText(_translate("MainWindow", "1000"))
            self.label_7.setText(_translate("MainWindow", "Run:"))
            self.pushButton.setText(_translate("MainWindow", "Simulate!"))
            self.menuWindow.setTitle(_translate("MainWindow", "Window"))

        def run(self,Nparticles, Mtot, r0, disp, θ, dt, frames):
            r, v = generate(Nparticles, Mtot, r0, disp)
            L = 300#2 * np.linalg.norm(r[-1])

            particles = [Particle(r[i], v[i], m=Mtot / Nparticles) for i in range(Nparticles)] #:,i
            colors = ['orange' if i == 10 else 'b' for i in range(Nparticles)]

            SDV = [v] # Storage of Data for V
            SDR = [r] # Storage of Data for R
            for frame in tqdm(range(frames)):
                percentage = int(frame/frames * 100)
                self.progressBar.setValue(percentage)
                self.lcdNumber.setProperty("intValue", frame)
                QtWidgets.QApplication.processEvents()
                time.sleep(0.001) # MUST BE HERE BECAUSE OF GIL

                #GetSituation(r,colors)
                # compute the location of the Center of Mass (COM) and total mass for the
                # ROOT cell
                Rgal_CM = np.sum([p.m * p.r for p in particles]) / np.sum([p.m for p in particles])
                Mgal = np.sum([p.m for p in particles])

                # initialize ROOT cell
                ROOT = Cell(np.array([0, 0]), L, parent=None, M=Mgal, R_CM=Rgal_CM)

                #BUILD TREE
                Tree(ROOT, particles)

                ################################################
                ##    COMPUTE FORCES USING MULTIPROCESSING ##
                ################################################
                N_CPU = cpu_count() #get the number of CPU cores
                PLATFORM = sys.platform #get the patform on which this script is running

                #NN defines the slice ranges for the particle array.
                #We want to split the particles array in N_CPU-1 parts, i.e.  the number
                #of
                #feasible subprocesses on this machine.
                NN = int(Nparticles / (N_CPU - 1))

                #If the platform is 'win32' we will use pipes.  The parent connector will
                #be
                #stored in the connections list.
                if PLATFORM == 'win32':
                    connections = []
                processes = [] #array with process instances

                #create a multiprocessing array for the force on each particle in shared
                #memory
                mp_Forces = Array(c_double, 2 * Nparticles)
                #create a 2D numpy array sharing its memory location with the
                #multiprocessing
                #array
                Forces = np.frombuffer(mp_Forces.get_obj(), dtype=c_double).reshape((Nparticles, 2))

                #spawn the processes
                for i in range(N_CPU - 1):
                    #ensure that the last particle is also included when Nparticles / (N_CPU
                    #-
                    #1) is not an integer
                    if i == N_CPU - 2:
                        if PLATFORM == 'win32':
                            parent_conn, child_conn = Pipe() #create a duplex Pipe
                            p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:], Mtot, r0), kwargs=dict(θ=θ, conn=child_conn)) #spawn process
                            p.start() #start process
                            parent_conn.send(Forces[i * NN:]) #send Forces array through Pipe
                            connections.append(parent_conn)
                        else:
                            p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:], Mtot, r0), kwargs=dict(Forces=Forces[i * NN:], θ=θ)) #spawn process
                            p.start() #start process
                    else:
                        if PLATFORM == 'win32':
                            parent_conn, child_conn = Pipe() #create a duplex Pipe
                            p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN], Mtot, r0), kwargs=dict(θ=θ, conn=child_conn)) #spawn process
                            p.start() #start process
                            parent_conn.send(Forces[i * NN:(i + 1) * NN]) #send Forces array through Pipe
                            connections.append(parent_conn)
                        else:
                            p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN], Mtot, r0), kwargs=dict(Forces=Forces[i * NN:(i + 1) * NN], θ=θ)) #spawn process
                            p.start() #start process
                        processes.append(p)

                #if platform is 'win32' => receive filled Forces arrays through Pipe
                if PLATFORM == 'win32':
                    Forces = np.concatenate([connection_receiveAndClose(conn) for conn in connections], axis=0)

                #join and terminate all processes
                processes_joinAndTerminate(processes)

                r,v = particles2arr(particles)

                if frame == 0: 
                    r, v, dummy = leapfrog(r, Forces, v, dt=dt, init=True)
                else:
                    if frame % 2 == 0:
                        r, v, vstore = leapfrog(r, Forces, v, dt=dt)
                        SDR.append(r)
                        SDV.append(vstore)
                    else:
                        r, v, vstore = leapfrog(r, Forces, v, dt=dt)

                particles = updateparticles(r, v, particles)

            outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
            np.savez(outfile,r=np.array(SDR, dtype=object))
            AnimateOrbit(outfile, len(SDR))

        def clicked(self):

            Nparticles = int(self.lineEdit_4.text())
            theta = float(self.lineEdit.text())
            dt = float(self.lineEdit_2.text())
            Mtot = 10 ** 9
            r0 = 20
            disp = 1600
            frames = int(self.lineEdit_3.text())
            print("-" * 10)
            print("\nPARAMETERS: ")
            print(f" theta  = {theta}")
            print(f" dt  = { dt }")
            print(f" Frames  = { frames }")
            print(f" Nparticles  = { Nparticles }")
            print("-" * 10)

            self.run(Nparticles, Mtot, r0, disp, theta, dt, frames)

    class mywindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(mywindow, self).__init__()

            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)


    app = QtWidgets.QApplication([])
    application = mywindow()
    application.show()
    sys.exit(app.exec())