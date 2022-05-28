from array import array
from distutils.util import convert_path
from lib2to3.pytree import convert
from turtle import window_height
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
from qcrop.ui import QCrop

from PyQt5.QtGui import QPixmap, QImage, QTransform
from PIL import Image,ImageEnhance, ImageQt

from PIL import Image as im

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class image_editor(QMainWindow):
    def __init__(self):
        super(image_editor, self).__init__()
        loadUi('ui-image-editor.ui', self)
        self.exist_img = False
        self.original_image = None
        self.will_change_img = None
        self.tmp = None
        self.window_title="Image Editor"
        self.setWindowTitle(self.window_title)
        self.tabWidget.hide()
        self.image_label_1.hide()
        self.image_label_2.hide()
        self.original_label.hide()
        self.processed_label.hide()
        self.reset_btn.hide()
        self.scrollArea.hide()
        self.back_btn.hide()
       
        self.openImage_btn_new.triggered.connect(self.browse_image)
        self.reset_btn.clicked.connect(self.reset_image)
        self.saveImage_btn.triggered.connect(self.save_image) 
        self.action_exit.triggered.connect(self.exit)

        self.processed_filter = None
        self.list_state = ["","",""]
        self.list_image_for_undo = []

        # --- scroll
        self.dial_rotatio.valueChanged.connect(self.rotation_dial)
        self.slider_scaling.valueChanged.connect(self.scaling)
        self.slider_gamma.valueChanged.connect(self.Gamma)
        self.slider_gaussian.valueChanged.connect(self.gaussian_filter)
        self.slider_dilate_erosior.valueChanged.connect(self.dilate_erosion)
        self.slider_log.valueChanged.connect(self.log)
        self.cb_canny_edge_detector.stateChanged.connect(self.canny_edg_detector)
        self.slider_min.valueChanged.connect(self.canny_edg_detector)
        self.slider_max.valueChanged.connect(self.canny_edg_detector)
        self.slider_true_tone_1.valueChanged.connect(self.exponential_function)
        self.slider_true_tone_2.valueChanged.connect(self.exponential_function)
        self.slider_true_tone_3.valueChanged.connect(self.exponential_function)
        self.slider_true_tone_4.valueChanged.connect(self.exponential_function)
        self.slider_60tv_val.valueChanged.connect(self.tv60)
        self.slider_60tv_threshold.valueChanged.connect(self.tv60)
        self.slider_brightness.valueChanged.connect(self.brightness)
        self.slider_changeBlur.valueChanged.connect(self.changeBlur)

        #------- tab edit
        self.edit_crop_btn.clicked.connect(self.crop)
        self.edit_flipping.clicked.connect(self.flipImage)
        self.edit_shearing.clicked.connect(self.shearing)
        self.edit_mirroring.clicked.connect(self.mirrorImg)

        # ---- Smoothing
        self.actionBlur.triggered.connect(self.blur)
        self.actionBox_Filter.triggered.connect(self.box_filter)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionBilateral_Filter.triggered.connect(self.bilateral_filter)

        #---- Filter menu
        self.actionMedian_threshold.triggered.connect(self.median_threshold)
        self.actionDirectional_Filtering_1.triggered.connect(self.directional_filtering1)
        self.actionDirectional_Filtering_2.triggered.connect(self.directional_filtering2)
        self.actionDirectional_Filtering_3.triggered.connect(self.directional_filtering3)

        #-----view
        self.actionZoom_In.triggered.connect(self.zoom_out)
        self.actionZoom_Out.triggered.connect(self.zoom_in)

        #--- effect
        self.effect_sepia_btn.clicked.connect(self.effect_sepia)
        self.effect_black_btn.clicked.connect(self.effect_black)
        self.effect_cartoon_btn.clicked.connect(self.effect_cartoon)
        self.effect_contrast_btn.clicked.connect(self.effect_contrast)
        self.effect_gray_btn.clicked.connect(self.effect_gray)
        self.effect_handDraw_btn.clicked.connect(self.effect_handDraw)
        self.effect_negative_btn.clicked.connect(self.effect_negative)
        #----filter
        self.filter_focus_btn.clicked.connect(self.filter_focus)
        self.filter_blue_curacao_btn.clicked.connect(self.filter_blue_curacao)
        self.filter_dream_btn.clicked.connect(self.filter_dream)

        self.back_btn.clicked.connect(self.back)

    def browse_image(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.original_image  = cv2.imread(self.filename)
        self.tmp = self.original_image 
        self.exist_img = True
        self.update_img_original(self.original_image)
        self.tabWidget.show()
        self.image_label_1.show()
        self.image_label_2.show()
        self.original_label.show()
        self.processed_label.show()
        self.reset_btn.show()
        self.note_label.hide()     
        self.scrollArea.show()
        self.back_btn.show()

    def open_image(self):
        fname = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def exit(self):
        QApplication.instance().quit()

    def reset_image(self):
        if self.exist_img == True:
            self.image_label_2.clear()
            self.slider_scaling.setValue(1)
            self.slider_gamma.setValue(10)
            self.slider_gaussian.setValue(0)
            self.slider_dilate_erosior.setValue(1)
            self.slider_log.setValue(1)
            self.slider_min.setValue(1)
            self.slider_max.setValue(1)
            self.slider_true_tone_1.setValue(0)
            self.slider_true_tone_2.setValue(0)
            self.slider_true_tone_3.setValue(0)
            self.slider_true_tone_4.setValue(0)
            self.slider_60tv_val.setValue(1)
            self.slider_60tv_threshold.setValue(1)
            self.slider_brightness.setValue(0)
            self.slider_changeBlur.setValue(0)
            self.update_img(self.original_image)
            self.processed_filter = self.will_change_img
            self.list_image_for_undo.clear()
            self.list_state.clear()
            self.list_state = ["","",""]
            
        else:
            QMessageBox.warning(self, 'Reset Image Error', 'There is no image that can be reset.')    

    def save_image(self):
        if self.will_change_img:
            options = QFileDialog.Options()
            filter = 'JPEG (*.jpg);;PNG (*.png);;Bitmap (*.bmp)'
            self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', filter, options=options)
            self.will_change_img.save(self.save_path)
        else:
            QMessageBox.warning(self, 'Saving Error', 'There is no image that have been changed.')
       

    #-------

    def update_img_original(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.will_change_img = image
        self.image_label_1.setPixmap(QtGui.QPixmap.fromImage(image))
        self.image_label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def update_img(self,image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.will_change_img = image.copy()
        self.image_label_2.setPixmap(QPixmap.fromImage(image))
        self.image_label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def img_to_cv(self, image):
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()
        return cv_image

    def save_state(self, state): # save state for each filter and slider
        self.list_state.append(state)
        del self.list_state[0] 
        print("state-image:",self.list_state) 
        n = len(self.list_state)
        if (self.list_state[n-1] != self.list_state[n-2]):
            self.processed_filter = self.will_change_img 
            
                

    def back(self): # back to prev image
        n = len(self.list_image_for_undo)  
        self.list_state.append("back")
        del self.list_state[0] 
        print("state-image:",self.list_state)
        print("list image:", len(self.list_image_for_undo))
        if (n>1):
            self.processed_filter = self.list_image_for_undo[n-2]
            if(str(type(self.processed_filter)) == "<class 'numpy.ndarray'>"):
                self.update_img(self.processed_filter)
            elif (str(type(self.processed_filter)) == "<class 'PyQt5.QtGui.QImage'>"):
                self.image_label_2.setPixmap(QPixmap.fromImage(self.processed_filter))
            del self.list_image_for_undo[n-1]
        elif(n == 1):
            self.update_img(self.original_image)
            del self.list_image_for_undo[n-1]   

    # ------------------------------------  scroll
      
    def rotation_dial(self, angle):
        self.save_state("rotation")  
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        rows, cols, steps = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
        self.update_img(img) 
        self.list_image_for_undo.append(img) 
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def scaling(self,c):
        self.save_state("scaling")
        scaling_value = self.slider_scaling.value()
        self.value_scaling.setText(str(scaling_value))
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        img = cv2.resize(img, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.update_img(img)     
        self.list_image_for_undo.append(img)    
       # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]
        
    def Gamma(self, gamma):
        gamma_value = self.slider_gamma.value()
        self.value_gamma.setText(str(gamma_value))  

        self.save_state("gamma")  
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        gamma = gamma*0.1    
        invGamma = 1.0 /gamma
        # xây dựng bảng tra cứu ánh xạ các giá trị pixel [0, 255] thành
        # giá trị gamma đã điều chỉnh của họ
        # sau đó được nâng lên thành lũy thừa của gamma nghịch đảo - giá trị này sau đó được lưu trữ trong table
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        img = cv2.LUT(img, table)
        self.update_img(img)     
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def gaussian_filter(self, g):
        gaussian_value = self.slider_gaussian.value()
        self.value_gaussian.setText(str(gaussian_value))

        self.save_state("gaussian") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        img = cv2.GaussianBlur(img, (5, 5), g)
        self.update_img(img) 
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def dilate_erosion(self , iter):
        dilate_erosion_value = self.slider_dilate_erosior.value()
        self.value_dilate_erosion.setText(str(dilate_erosion_value))

        self.save_state("dilate_erosion") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        if iter > 0 :
            #ấy ma trận có kích thước 5 làm hạt nhân
            # #bạn muốn làm xói mòn / làm giãn một hình ảnh nhất định.
            kernel = np.ones((4, 7), np.uint8)
            img = cv2.erode(img, kernel, iterations=iter)
        else :
            kernel = np.ones((2, 6), np.uint8)
            img = cv2.dilate(img, kernel, iterations=iter*-1)
        self.update_img(img)
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def log(self, c):
        log_value = self.slider_log.value()
        self.value_log.setText(str(log_value))

        self.save_state("log") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        img_2 = np.uint8(np.log(img))
        img_2 = cv2.threshold(img_2, 2, 225, cv2.THRESH_BINARY)[1]
        self.update_img(img_2) 
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def canny_edg_detector(self):
        min_value = self.slider_min.value()
        self.value_min.setText(str(min_value))
        max_value = self.slider_max.value()
        self.value_max.setText(str(max_value))

        self.save_state("canny_edg_detector") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        if self.cb_canny_edge_detector.isChecked():
            #Chuyển đổi sang graycsal
            can = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(can, self.slider_min.value(), self.slider_max.value())
        self.update_img(img)  
        self.list_image_for_undo.append(img)  
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def exponential_function(self):
        self.save_state("exponential_function") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        res = img.copy()
        exp =  self.slider_true_tone_1.value()
        exp = 1 + exp / 100
        s1 = self.slider_true_tone_2.value()
        s2 = self.slider_true_tone_3.value()
        s3 = self.slider_true_tone_4.value()
        table = np.array([min((i ** (1+exp/100)), 255) for i in np.arange(0, 256)]).astype("uint8")  #

        for i in range(3):
            if i in (s1,s2):  # if channel is present
                res[:, :, i] = res[:, :, i] # increasing the values if channel selected
            else:
                if s3:  # for light
                    res[:, :, i]
                    exp = 2-exp   # reducing value to make the channels light
                else:  # for dark
                    res[:, :, i] = 0  # converting the whole channel to 0
        # generating table for exponential function
        img = cv2.LUT(res, table)
        self.update_img(img)
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]
        

    def tv60(self):
        val_value = self.slider_60tv_val.value()
        self.value_val.setText(str(val_value))
        threshold_value = self.slider_60tv_threshold.value()
        self.value_threshold.setText(str(threshold_value))

        self.save_state("tv'60") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val = self.slider_60tv_val.value()
        thresh = self.slider_60tv_threshold.value()
        for i in range(height):
           for j in range(width):
               if np.random.randint(100) <= thresh:
                   if np.random.randint(2) == 0:
                       gray[i, j] = min(gray[i, j] + np.random.randint(0, val + 1),
                                        255)  # adding noise to image and setting values > 255 to 255.
                   else:
                       gray[i, j] = max(gray[i, j] - np.random.randint(0, val + 1),
                                        0)  # subtracting noise to image and setting values < 0 to 0.
        img = gray
        self.update_img(img)
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def brightness(self):
        self.value_brightness.setText(str(self.slider_brightness.value()))
        self.save_state("brightness") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        lim = 255 - self.slider_brightness.value()
        v[v>lim] = 255
        v[v<=lim] += self.slider_brightness.value()
        final_hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
        self.update_img(img)   
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    def changeBlur(self):
        self.value_changeBlur.setText(str(self.slider_changeBlur.value()))
        self.save_state("changBlur") 
        image = ImageQt.fromqimage(self.processed_filter)
        img = self.img_to_cv(image)
        kernel_size = (self.slider_changeBlur.value()+1, self.slider_changeBlur.value()+1) # +1 is to avoid 0
        img = cv2.blur(img, kernel_size)
        self.update_img(img)
        self.list_image_for_undo.append(img)
        # delete image when slider more three
        if(self.list_state[0] == self.list_state[1] == self.list_state[2]):
            del self.list_image_for_undo[len(self.list_image_for_undo)-2]

    # ------------------------------- tab edit

    def mirrorImg(self):
        mirror = QTransform().scale(-1, 1)
        pixmap = QPixmap(self.will_change_img)
        self.list_state.append("mirrirImg")
        del self.list_state[0]
        mirrored = pixmap.transformed(mirror)
        self.will_change_img = QImage(mirrored)
        self.image_label_2.setPixmap(QPixmap.fromImage(self.will_change_img))
        self.list_image_for_undo.append(self.will_change_img)

    def flipImage(self):
        transform90 = QTransform().rotate(90)
        self.list_state.append("flipImage")
        del self.list_state[0]
        pixmap = QPixmap(self.will_change_img)
        rotated = pixmap.transformed(transform90, mode=Qt.SmoothTransformation)
        self.will_change_img = QImage(rotated)
        self.image_label_2.setPixmap(QPixmap.fromImage(self.will_change_img))  
        self.list_image_for_undo.append(self.will_change_img) 

    def crop(self):
        pixmap = QPixmap(self.will_change_img)
        self.list_state.append("crop")
        del self.list_state[0]
        crop_tool = QCrop(pixmap)
        status = crop_tool.exec()
        if status == 1:
            cropped_image = crop_tool.image
            self.will_change_img = QImage(cropped_image)
            self.image_label_2.setPixmap(QPixmap.fromImage(self.will_change_img))
        self.list_image_for_undo.append(self.will_change_img)    

    def shearing(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("shearing")
        del self.list_state[0]
        rows, cols, ch = img.shape
        #ma trân chuyển đổi liên kết ảnh
        pts1 = np.float32([[50,50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 150], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (cols, rows))
        self.update_img(img) 
        self.list_image_for_undo.append(img)

    # Smoothing
    def blur(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("blur")
        del self.list_state[0]
        rows, cols, ch = img.shape
        img = cv2.blur(img, (5, 5))
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def box_filter(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("box_filter")
        del self.list_state[0] 
        img = cv2.boxFilter(img, -1,(20,20))
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def median_filter(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("median_filter")
        del self.list_state[0]
        img = cv2.medianBlur(img, 5)
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def bilateral_filter(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("bilateral_filter")
        del self.list_state[0]
        img = cv2.bilateralFilter(img,9,75,75)
        self.update_img(img)
        self.list_image_for_undo.append(img)

    # filter menu
    def median_threshold(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("median_threshold")
        del self.list_state[0]
        grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img,5)
        retval, threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = threshold
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def directional_filtering1(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("directional_filtering1")
        del self.list_state[0]
        kernel = np.ones((3, 3), np.float32) / 9
        img = cv2.filter2D(img, -1, kernel)
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def directional_filtering2(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("directional_filtering2")
        del self.list_state[0]
        kernel = np.ones((5, 5), np.float32) / 9
        img = cv2.filter2D(img, -1, kernel)
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def directional_filtering3(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("directional_filtering3")
        del self.list_state[0]
        kernel = np.ones((7, 7), np.float32) / 9
        img = cv2.filter2D(img, -1, kernel)
        self.update_img(img)
        self.list_image_for_undo.append(img)

    #---- view
    def zoom_out(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("zoom_out")
        del self.list_state[0]
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def zoom_in(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("zoom_in")
        del self.list_state[0]
        img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.update_img(img)   
        self.list_image_for_undo.append(img)

    #------- tab effect

    def effect_sepia(self):
        image = ImageQt.fromqimage(self.will_change_img)
        self.list_state.append("effect_sepia")
        del self.list_state[0]
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                red = int(r * 0.393 + g * 0.769 + b * 0.189)
                green = int(r * 0.349 + g * 0.686 + b * 0.168)
                blue = int(r * 0.272 + g * 0.534 + b * 0.131)
                image.putpixel((x, y), (red, green, blue))
        image = self.img_to_cv(image)
        self.update_img(image)   
        self.list_image_for_undo.append(image)

    def effect_black(self):
        image = ImageQt.fromqimage(self.will_change_img)
        self.list_state.append("effect_black")
        del self.list_state[0]
        if image.mode != "RGBA":
            image.convert("RGBA")

        width, height = image.size
        pixels = image.load()

        separator = 255 / 1.2 / 2 * 3
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                total = r + g + b
                if total > separator:
                    image.putpixel((x, y), (255, 255, 255))
                else:
                    image.putpixel((x, y), (0, 0, 0))

        image = self.img_to_cv(image)
        self.update_img(image) 
        self.list_image_for_undo.append(image)

    def effect_negative(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("effect_negative")
        del self.list_state[0]

        k = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                k.append(img[i, j])
        maxi = np.max(k)
        self.temp = img.copy()

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                self.temp[i, j] = maxi - self.temp[i, j]
        self.update_img(self.temp)     
        self.list_image_for_undo.append(self.temp)  

    def effect_cartoon(self):
        samp = 2
        filternum = 50
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        self.list_state.append("effect_cartoon")
        del self.list_state[0]

        for _ in range(samp):
            img = cv2.pyrDown(img)

        for _ in range(filternum):
            img = cv2.bilateralFilter(img, 9, 9, 7)

        for _ in range(samp):
            img = cv2.pyrUp(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        (x, y, z) = img.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        self.temp = cv2.bitwise_and(img, img_edge)
        self.update_img(self.temp)
        self.list_image_for_undo.append(self.temp)

    def effect_handDraw(self):
        image = ImageQt.fromqimage(self.will_change_img)
        imga = self.img_to_cv(image)
        self.list_state.append("effect_handDraw")
        del self.list_state[0]
        gray = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(gray)
        smooth = cv2.GaussianBlur(invert, (21,21),sigmaX=0, sigmaY=0)

        def dodge(x, y):
            return cv2.divide(x, 255 - y, scale=256)
        img = dodge(gray, smooth)
        self.update_img(img)
        self.list_image_for_undo.append(img)

    def effect_gray(self):
        image = ImageQt.fromqimage(self.will_change_img)
        self.list_state.append("effect_gray")
        del self.list_state[0]
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                gray = int(r * 0.2126 + g * 0.7152 + b * 0.0722)
                image.putpixel((x, y), (gray, gray, gray))
        image = self.img_to_cv(image)
        self.update_img(image)
        self.list_image_for_undo.append(image)

    def effect_contrast(self):
        image = ImageQt.fromqimage(self.will_change_img)
        self.list_state.append("effect_contrast")
        del self.list_state[0]
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        avg = 0
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                avg += r * 0.299 + g * 0.587 + b * 0.114
        avg /= image.size[0] * image.size[1]

        palette = []
        for i in range(256):
            temp = int(avg + 2 * (i - avg))
            if temp < 0:
                temp = 0
            elif temp > 255:
                temp = 255
            palette.append(temp)

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                image.putpixel((x, y), (palette[r], palette[g], palette[b]))

        image = self.img_to_cv(image)
        self.update_img(image)
        self.list_image_for_undo.append(image)

    def filter_focus(self):
        self.list_state.append("filter_focus")
        del self.list_state[0]
        image = ImageQt.fromqimage(self.will_change_img)
        image = self.img_to_cv(image)
        rows, cols = image.shape[:2]
        X_kernel = cv2.getGaussianKernel(cols, 200)
        Y_kernel = cv2.getGaussianKernel(rows, 200)
        kernel = Y_kernel * X_kernel.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        output = np.copy(image)
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask
        self.update_img(output)   
        self.list_image_for_undo.append(output) 

    def filter_blue_curacao(self):
        self.list_state.append("filter_blue_curacao")
        del self.list_state[0]
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        image.point(lambda i: i ^ 0x8B if i < 128 else i)
        image.point(lambda i: i ^ 0x3D if i < 256 else i)
        width, height = image.size
        pixels = image.load()
        for i in range(width):
            for j in range(height):
                red, green, blue = pixels[i, j]
                pixels[i, j] = min(100, int(green * blue / 255)), \
                               min(100, int(blue * red / 255)), \
                               min(255, int(red * green / 255))

        image = self.img_to_cv(image)
        self.update_img(image)   
        self.list_image_for_undo.append(image)  

    def filter_dream(self):
        self.list_state.append("filter_dream")
        del self.list_state[0]
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        b, g, r = cv2.split(img)
        rbr_img = cv2.merge((r, b, g))
        morphology = cv2.morphologyEx(rbr_img, cv2.MORPH_OPEN, element)
        canvas = cv2.normalize(morphology, None, 20, 255, cv2.NORM_MINMAX)
        new_image = cv2.stylization(canvas, sigma_s=60, sigma_r=0.6)
        self.update_img(new_image)
        self.list_image_for_undo.append(new_image) 

app = QApplication(sys.argv)
win = image_editor()
win.show()
sys.exit(app.exec())

