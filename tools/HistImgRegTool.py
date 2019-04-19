import sys, os
import openslide
import math
import numpy as np
import imreg_dft as ird
import scipy as sp
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import QDir, Qt, pyqtSlot, QCoreApplication, QMetaObject, QObject, QPoint, QEvent
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QPicture, QCursor, QPen
from PyQt5.QtWidgets import (QAction, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QWidget, QApplication,
                             QFileDialog, QLabel,
                             QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QTextEdit, QLineEdit, QLayout,
                             QComboBox, QSpinBox, QDoubleSpinBox)


class ImgRegistration(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.OpenTemplateButton = QPushButton("Template")  # load template image
        self.OpenFloatButton = QPushButton("Floating")  # load floating image
        self.EditImgNameTemplate = QLineEdit()  # edit line to show template image name
        self.EditImgNameFloating = QLineEdit()  # edit line to show floating image name
        self.ImgSizeLabel = QLabel("Patch Size:")
        self.EditImgPatchSize = QLineEdit("800")  # editbox to show image patch size
        self.ImgPatchSize_display = QLabel("*800")
        self.AutoRegButton = QPushButton("AutoReg")  # automatically co-registrate two images
        self.HelpButton = QPushButton("Help")
        self.ImgOffsetLabel_x = QLabel("Offset_X:")
        self.BoxOffsetX = QSpinBox()  # editbox to show offsets x
        self.ImgOffsetLabel_y = QLabel("Offset_Y:")
        self.BoxOffsetY = QSpinBox()  # editbox to show offsets y
        self.ImgAngleLabel = QLabel("Angle:")
        self.BoxAngle = QDoubleSpinBox()  # editbox to show registration angles
        self.TemplateImageLabel = QLabel()  # show image patch from template WSI
        self.FloatImageLabel = QLabel()  # show image patches from floating WSI
        self.AbsCoordinateTemplate = QLabel("Template Coordinate[x,y]:")
        self.AbsCoordinateFloat = QLabel("Absolut Coordinate[x,y]:")
        self.EditAbsCoordinateTemplateX = QLineEdit()
        self.EditAbsCoordinateTemplateY = QLineEdit()
        self.EditAbsCoordinateFloatX = QLineEdit()
        self.EditAbsCoordinateFloatY = QLineEdit()
        self.CoordinateTemplate = QLabel()
        self.CoordinateFloat = QLabel()

        vbox_main = QVBoxLayout()
        row1_hbox = QHBoxLayout()
        row2_hbox = QHBoxLayout()
        row3_hbox = QHBoxLayout()
        row4_hbox = QHBoxLayout()
        row5_hbox = QHBoxLayout()
        row1_hbox.addWidget(self.EditImgNameTemplate)
        row1_hbox.addWidget(self.OpenTemplateButton)
        row1_hbox.addWidget(self.EditImgNameFloating)
        row1_hbox.addWidget(self.OpenFloatButton)
        row2_hbox.addWidget(self.ImgSizeLabel)
        row2_hbox.addWidget(self.EditImgPatchSize)
        row2_hbox.addWidget(self.ImgPatchSize_display)
        row2_hbox.addStretch()
        row2_hbox.addWidget(self.ImgOffsetLabel_x)
        row2_hbox.addWidget(self.BoxOffsetX)
        row2_hbox.addWidget(self.ImgOffsetLabel_y)
        row2_hbox.addWidget(self.BoxOffsetY)
        row2_hbox.addWidget(self.ImgAngleLabel)
        row2_hbox.addWidget(self.BoxAngle)
        row2_hbox.addWidget(self.AutoRegButton)
        row2_hbox.addWidget(self.HelpButton)
        row3_hbox.addWidget(self.AbsCoordinateTemplate)
        row3_hbox.addWidget(self.EditAbsCoordinateTemplateX)
        row3_hbox.addWidget(self.EditAbsCoordinateTemplateY)
        row3_hbox.addStretch()
        row3_hbox.addWidget(self.AbsCoordinateFloat)
        row3_hbox.addWidget(self.EditAbsCoordinateFloatX)
        row3_hbox.addWidget(self.EditAbsCoordinateFloatY)
        row3_hbox.addStretch()
        row4_hbox.addWidget(self.TemplateImageLabel)
        row4_hbox.addWidget(self.FloatImageLabel)
        row5_hbox.addWidget(self.CoordinateTemplate)
        row5_hbox.addWidget(self.CoordinateFloat)
        vbox_main.addLayout(row1_hbox)
        vbox_main.addLayout(row2_hbox)
        vbox_main.addLayout(row3_hbox)
        vbox_main.addLayout(row4_hbox)
        vbox_main.addLayout(row5_hbox)
        vbox_main.setSizeConstraint(QLayout.SetMinimumSize)

        self.setLayout(vbox_main)
        self.TemplateImageLabel.setMouseTracking(True)
        self.FloatImageLabel.setMouseTracking(True)
        self.title = 'File'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480

        self.T_Orig_X_Coord = 21000
        self.T_Orig_Y_Coord = 21000
        self.F_Orig_X_Coord = 21000  # 21249 + 42
        self.F_Orig_Y_Coord = 21000
        self.init_offset = [0, 0]
        self.F_Angle = 0

        self.F_Spinbox_X = 0
        self.F_Spinbox_Y = 0
        self.F_Abs_X_Coord = self.F_Orig_X_Coord + self.init_offset[0] + self.F_Spinbox_X
        self.F_Abs_Y_Coord = self.F_Orig_Y_Coord + self.init_offset[1] + self.F_Spinbox_Y

        self.sd_fix = None
        self.sd_float = None

        pixmap_t = QPixmap(800, 800)
        pixmap_t.fill(Qt.white)
        pixmap_f = QPixmap(800, 800)
        pixmap_f.fill(Qt.white)
        self.TemplatePixmap = pixmap_t
        self.FloatPixmap = pixmap_f
        self.Template_Load = False
        self.Float_Load = False
        self.ImgPatchSize = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Registration")
        self.setGeometry(self.left + 30, self.top + 30, self.width, self.height)
        self.EditImgPatchSize.setFixedWidth(50)
        pixmap_it = QPixmap(800, 800)
        pixmap_it.fill(Qt.white)
        pixmap_if = QPixmap(800, 800)
        pixmap_if.fill(Qt.white)
        self.TemplateImageLabel.setPixmap(pixmap_it)
        self.TemplateImageLabel.setCursor(QCursor(Qt.CrossCursor))
        self.FloatImageLabel.setPixmap(pixmap_if)
        self.FloatImageLabel.setCursor(QCursor(Qt.CrossCursor))
        self.BoxOffsetY.setFixedWidth(80)
        self.BoxOffsetY.setRange(-2000, 2000)
        self.BoxOffsetY.setSingleStep(1)
        self.BoxOffsetX.setFixedWidth(80)
        self.BoxOffsetX.setRange(-2000, 2000)
        self.BoxOffsetX.setSingleStep(1)
        self.EditAbsCoordinateTemplateX.setFixedWidth(100)
        self.EditAbsCoordinateTemplateY.setFixedWidth(100)
        self.EditAbsCoordinateFloatX.setFixedWidth(100)
        self.EditAbsCoordinateFloatY.setFixedWidth(100)
        self.BoxOffsetX.setEnabled(False)
        self.BoxOffsetY.setEnabled(False)
        self.BoxAngle.setEnabled(False)
        self.EditImgPatchSize.setEnabled(False)
        self.EditAbsCoordinateTemplateX.setEnabled(False)
        self.EditAbsCoordinateTemplateY.setEnabled(False)
        self.EditAbsCoordinateFloatX.setEnabled(False)
        self.EditAbsCoordinateFloatY.setEnabled(False)
        self.BoxAngle.setFixedWidth(50)
        # self.BoxAngle.setRange(-math.pi, math.pi)  # rotate as radians
        # self.BoxAngle.setSingleStep(0.1)
        self.BoxAngle.setRange(-90, 90)  # rotate as degree
        self.BoxAngle.setSingleStep(1)
        self.AutoRegButton.setStyleSheet("background-color: green;")
        # self.EditAbsCoordinateTemplateX.editingFinished.connect(self.ROITemplateX_Change)
        # self.EditAbsCoordinateTemplateY.editingFinished.connect(self.ROITemplateY_Change)
        self.EditAbsCoordinateTemplateX.textChanged.connect(self.ROITemplateX_Change)
        self.EditAbsCoordinateTemplateY.textChanged.connect(self.ROITemplateY_Change)

        self.EditImgPatchSize.editingFinished.connect(self.ImgPatchSize_Change)
        self.BoxOffsetX.valueChanged.connect(self.OffsetX_Change)
        self.BoxOffsetY.valueChanged.connect(self.OffsetY_Change)
        self.BoxAngle.valueChanged.connect(self.Angle_Change)
        self.OpenTemplateButton.clicked.connect(self.openTemplate)
        self.OpenFloatButton.clicked.connect(self.openFloating)
        self.HelpButton.clicked.connect(self.HelpClicked)
        self.AutoRegButton.clicked.connect(self.AutoReg)

    def HelpClicked(self):
        QMessageBox.information(self, 'Help', "\
        Developed by Jun Jiang (smujiang@gmail.com)\n \
        Visit my Github(https://github.com/smujiang) for more information", QMessageBox.Ok)

    def AutoReg(self):
        # QMessageBox.information(self, 'Message', "Pending", QMessageBox.Ok)
        WSI_Width, WSI_Height = self.sd_fix.dimensions
        thumb_size_x = round(WSI_Width / 100)
        thumb_size_y = round(WSI_Height / 100)
        thumbnail_fixed = self.sd_fix.get_thumbnail([thumb_size_x, thumb_size_y])
        thumbnail_fixed = thumbnail_fixed.convert('L')  # get grayscale image
        WSI_Width, WSI_Height = self.sd_float.dimensions
        thumb_size_x = round(WSI_Width / 100)
        thumb_size_y = round(WSI_Height / 100)
        thumbnail_float = self.sd_float.get_thumbnail([thumb_size_x, thumb_size_y])
        thumbnail_float = thumbnail_float.convert('L')  # get grayscale image
        img_w = min(thumbnail_fixed.width, thumbnail_float.width)
        img_h = min(thumbnail_fixed.height, thumbnail_float.height)
        img_size = min(img_w, img_h)
        fixed_array = np.array(thumbnail_fixed)
        float_array = np.array(thumbnail_float)
        fixed_img = Image.fromarray(fixed_array[0:img_size, 0:img_size])
        float_img = Image.fromarray(float_array[0:img_size, 0:img_size])
        Img_fix = sp.misc.fromimage(fixed_img, True)  # flatten is True, means we convert images into graylevel images.
        Img_float = sp.misc.fromimage(float_img, True)
        con_s = dict(angle=[0, 0], scale=[1, 1])
        sim = ird.similarity(Img_fix, Img_float, constraints=con_s)
        tvec = sim["tvec"].round(4)
        offset = [-tvec[1], -tvec[0]]
        raw_offset = [offset[0] * 100, offset[1] * 100]
        self.BoxOffsetX.setValue(raw_offset[0])
        self.BoxOffsetY.setValue(raw_offset[1])
        self.F_Abs_X_Coord = self.F_Orig_X_Coord + self.init_offset[0] + self.BoxOffsetX.value()
        self.F_Abs_Y_Coord = self.F_Orig_Y_Coord + self.init_offset[1] + self.BoxOffsetY.value()
        print(raw_offset)
        self.updateFloatImg()


    def ImgPatchSize_Change(self):
        x = int(self.EditImgPatchSize.text())
        if (x > 50) & self.Template_Load & self.Float_Load:
            Img_fix_col = self.sd_fix.read_region((self.T_Orig_X_Coord, self.T_Orig_Y_Coord), 0, (x, x))
            pixmap_fix = ImageQt(Img_fix_col)
            self.TemplatePixmap = QPixmap.fromImage(pixmap_fix)
            self.TemplateImageLabel.setPixmap(self.TemplatePixmap)
            Img_float_col = self.sd_float.read_region((self.F_Orig_X_Coord, self.F_Orig_Y_Coord), 0, (x, x))
            pixmap_float = ImageQt(Img_float_col)
            self.FloatPixmap = QPixmap.fromImage(pixmap_float)
            self.FloatImageLabel.setPixmap(self.FloatPixmap)
            self.ImgPatchSize = x
            self.ImgPatchSize_display.setText("*" + str(x))

    def ROITemplateX_Change(self):
        if self.Template_Load:
            self.T_Orig_X_Coord = int(self.EditAbsCoordinateTemplateX.text())
            self.updateTemplateImg()
            self.ROIFloatX_Change()
        print("Change Template X:" + str(self.T_Orig_X_Coord))

    def ROITemplateY_Change(self):
        if self.Template_Load:
            self.T_Orig_Y_Coord = int(self.EditAbsCoordinateTemplateY.text())
            self.updateTemplateImg()
            self.ROIFloatY_Change()
        print("Change Template Y:" + str(self.T_Orig_Y_Coord))

    def ROIFloatX_Change(self):
        if self.Float_Load:
            self.F_Orig_X_Coord = int(self.EditAbsCoordinateTemplateX.text())
            self.F_Abs_X_Coord = self.F_Orig_X_Coord+self.init_offset[0]+self.F_Spinbox_X
            self.EditAbsCoordinateFloatX.setText(str(self.F_Abs_X_Coord))
            self.updateFloatImg()
        print("Change Float X:" + str(self.F_Orig_X_Coord))

    def ROIFloatY_Change(self):
        if self.Float_Load:
            self.F_Orig_Y_Coord = int(self.EditAbsCoordinateTemplateY.text())
            self.F_Abs_Y_Coord = self.F_Orig_Y_Coord+self.init_offset[1]+self.F_Spinbox_Y
            self.EditAbsCoordinateFloatY.setText(str(self.F_Abs_Y_Coord))
            self.updateFloatImg()
        print("Change Float Y:" + str(self.F_Orig_Y_Coord))

    def OffsetX_Change(self):
        self.F_Abs_X_Coord = self.F_Orig_X_Coord + self.init_offset[0] + self.BoxOffsetX.value()
        self.EditAbsCoordinateFloatX.setText(str(self.F_Abs_X_Coord))
        self.updateFloatImg()

    def OffsetY_Change(self):
        self.F_Abs_Y_Coord = self.F_Orig_Y_Coord + self.init_offset[1] + self.BoxOffsetY.value()
        self.EditAbsCoordinateFloatY.setText(str(self.F_Abs_Y_Coord))
        self.updateFloatImg()

    def updateFloatImg(self):
        if self.F_Angle == 0:
            rel_x = self.F_Abs_X_Coord
            rel_y = self.F_Abs_Y_Coord
            Img_temp = self.sd_float.read_region((rel_x, rel_y), 0, (self.ImgPatchSize, self.ImgPatchSize))
            Img_temp = np.array(Img_temp)
            pixmap_temp = ImageQt(Image.fromarray(Img_temp))
        else:
            center_X = int(self.EditAbsCoordinateFloatX.text()) + int(self.FloatPixmap.size().width() / 2)
            center_Y = int(self.EditAbsCoordinateFloatY.text()) + int(self.FloatPixmap.size().height() / 2)
            temp_org_X = int(center_X - 1.414 * self.FloatPixmap.size().width() / 2)
            temp_org_Y = int(center_Y - 1.414 * self.FloatPixmap.size().height() / 2)
            Img_temp = self.sd_float.read_region((temp_org_X, temp_org_Y), 0, (
                int(1.414 * self.FloatPixmap.size().width()), int(1.414 * self.FloatPixmap.size().height())))
            Img_temp1 = np.array(Img_temp.rotate(self.F_Angle))
            # Img_temp1 = np.array(Img_temp.rotate(math.degrees(self.F_Angle)))  # if Angle is radians
            index_start_x = int(0.414 * self.FloatPixmap.size().width() / 2)
            index_start_y = int(0.414 * self.FloatPixmap.size().height() / 2)
            Img_rot = Img_temp1[index_start_x:index_start_x + self.FloatPixmap.size().width(),
                      index_start_y:index_start_y + self.FloatPixmap.size().height(), :]
            pixmap_temp = ImageQt(Image.fromarray(Img_rot))
        self.FloatImageLabel.setPixmap(QPixmap.fromImage(pixmap_temp))
        self.FloatPixmap = QPixmap.fromImage(pixmap_temp)

    def updateTemplateImg(self):
        x = int(self.EditAbsCoordinateTemplateX.text())
        y = int(self.EditAbsCoordinateTemplateY.text())
        Img_col = self.sd_fix.read_region((x, y), 0, (self.ImgPatchSize, self.ImgPatchSize))
        pixmap_fix = ImageQt(Img_col)
        self.TemplatePixmap = QPixmap.fromImage(pixmap_fix)
        self.TemplateImageLabel.setPixmap(self.TemplatePixmap)

    def Angle_Change(self):
        self.F_Angle = float(self.BoxAngle.text())
        if self.Float_Load:
            self.updateFloatImg()

    # def mousePressEvent(self, event):
    def mouseMoveEvent(self, event):
        if self.Template_Load & self.Float_Load:
            x = event.x()
            y = event.y()
            t_pos = self.TemplateImageLabel.pos()
            t_size = self.TemplateImageLabel.size()
            f_pos = self.FloatImageLabel.pos()
            f_size = self.FloatImageLabel.size()
            # Mouse moving in Template image
            if (t_pos.x() < x < (t_pos.x() + t_size.width())) & (t_pos.y() < y < (t_pos.y() + t_size.height())):
                rel_pos_x = x - t_pos.x() - 1
                rel_pos_y = y - t_pos.y() - 1
                self.CoordinateTemplate.setText('Mouse coords: ( %d : %d )' % (rel_pos_x, rel_pos_y))
                self.CoordinateFloat.setText('Mouse coords: ( %d : %d )' % (rel_pos_x, rel_pos_y))
                painter = QPainter(self)
                # temp_pixmap_t = self.TemplatePixmap
                self.updateTemplateImg()
                temp_pixmap_t = self.TemplateImageLabel.pixmap()
                painter.begin(temp_pixmap_t)
                pen = QPen(Qt.green, 1)
                painter.setPen(pen)
                painter.drawLine(rel_pos_x - 50, rel_pos_y, rel_pos_x + 50, rel_pos_y)
                painter.drawLine(rel_pos_x, rel_pos_y - 50, rel_pos_x, rel_pos_y + 50)
                painter.end()
                self.TemplateImageLabel.setPixmap(temp_pixmap_t)

                painter = QPainter(self)
                # temp_pixmap_f = self.FloatPixmap
                self.updateFloatImg()
                temp_pixmap_f = self.FloatImageLabel.pixmap()
                painter.begin(temp_pixmap_f)
                pen = QPen(Qt.green, 1)
                painter.setPen(pen)
                painter.drawLine(rel_pos_x - 50, rel_pos_y, rel_pos_x + 50, rel_pos_y)
                painter.drawLine(rel_pos_x, rel_pos_y - 50, rel_pos_x, rel_pos_y + 50)
                painter.end()
                self.FloatImageLabel.setPixmap(temp_pixmap_f)
            # Mouse moving in Floating image
            if (f_pos.x() < x < (f_pos.x() + f_size.width())) & (f_pos.y() < y < (f_pos.y() + f_size.height())):
                rel_pos_x = x - f_pos.x() - 1
                rel_pos_y = y - f_pos.y() - 1
                self.CoordinateTemplate.setText('Mouse coords: ( %d : %d )' % (rel_pos_x, rel_pos_y))
                self.CoordinateFloat.setText('Mouse coords: ( %d : %d )' % (rel_pos_x, rel_pos_y))

                self.F_Angle = float(self.BoxAngle.text())
                painter = QPainter(self)
                # temp_pixmap_f = self.FloatPixmap
                self.updateFloatImg()
                temp_pixmap_f = self.FloatImageLabel.pixmap()
                painter.begin(temp_pixmap_f)
                pen = QPen(Qt.green, 1)
                painter.setPen(pen)
                painter.drawLine(rel_pos_x - 50, rel_pos_y, rel_pos_x + 50, rel_pos_y)
                painter.drawLine(rel_pos_x, rel_pos_y - 50, rel_pos_x, rel_pos_y + 50)
                painter.end()
                self.FloatImageLabel.setPixmap(temp_pixmap_f)

                painter = QPainter(self)
                # temp_pixmap_t = self.TemplatePixmap
                self.updateTemplateImg()
                temp_pixmap_t = self.TemplateImageLabel.pixmap()
                painter.begin(temp_pixmap_t)
                pen = QPen(Qt.green, 1)
                painter.setPen(pen)
                painter.drawLine(rel_pos_x - 50, rel_pos_y, rel_pos_x + 50, rel_pos_y)
                painter.drawLine(rel_pos_x, rel_pos_y - 50, rel_pos_x, rel_pos_y + 50)
                painter.end()
                self.TemplateImageLabel.setPixmap(temp_pixmap_t)
        else:
            QMessageBox.warning(self, 'Message', "You must open both images first.", QMessageBox.Ok)

    @pyqtSlot()  # if click on the next button, jump to the next index of image
    def openTemplate(self):
        filename_fix = self.openFileNameDialog()
        if filename_fix:
            self.EditImgNameTemplate.setText(filename_fix)
            self.sd_fix = openslide.OpenSlide(filename_fix)
            Img_fix_col = self.sd_fix.read_region((self.T_Orig_X_Coord, self.T_Orig_Y_Coord), 0,
                                                  (self.ImgPatchSize, self.ImgPatchSize))
            pixmap_fix = ImageQt(Img_fix_col)
            self.TemplatePixmap = QPixmap.fromImage(pixmap_fix)
            self.TemplateImageLabel.setPixmap(self.TemplatePixmap)
            self.EditAbsCoordinateTemplateX.setText(str(self.T_Orig_X_Coord))
            self.EditAbsCoordinateTemplateY.setText(str(self.T_Orig_Y_Coord))
            self.Template_Load = True
            if self.Float_Load:
                self.BoxOffsetX.setEnabled(True)
                self.BoxOffsetY.setEnabled(True)
                self.BoxAngle.setEnabled(True)
                self.EditImgPatchSize.setEnabled(True)
                self.EditAbsCoordinateTemplateX.setEnabled(True)
                self.EditAbsCoordinateTemplateY.setEnabled(True)
                # self.EditAbsCoordinateFloatX.setEnabled(True)
                # self.EditAbsCoordinateFloatY.setEnabled(True)

    def openFloating(self):
        filename_float = self.openFileNameDialog()
        if filename_float:
            self.EditImgNameFloating.setText(filename_float)
            self.sd_float = openslide.OpenSlide(filename_float)
            Img_float_col = self.sd_float.read_region((self.F_Abs_X_Coord, self.F_Abs_Y_Coord), 0,
                                                      (self.ImgPatchSize, self.ImgPatchSize))
            pixmap_float = ImageQt(Img_float_col)
            self.FloatPixmap = QPixmap.fromImage(pixmap_float)
            self.FloatImageLabel.setPixmap(self.FloatPixmap)
            self.EditAbsCoordinateFloatX.setText(str(self.F_Abs_X_Coord))
            self.EditAbsCoordinateFloatY.setText(str(self.F_Abs_Y_Coord))
            self.Float_Load = True
            if self.Template_Load:
                self.BoxOffsetX.setEnabled(True)
                self.BoxOffsetY.setEnabled(True)
                self.BoxAngle.setEnabled(True)
                self.EditImgPatchSize.setEnabled(True)
                self.EditAbsCoordinateTemplateX.setEnabled(True)
                self.EditAbsCoordinateTemplateY.setEnabled(True)
                # self.EditAbsCoordinateFloatX.setEnabled(True)
                # self.EditAbsCoordinateFloatY.setEnabled(True)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        default_dir = "H:\\HE_IHC_Stains"
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", default_dir,
                                                  "All Files (*)", options=options)
        return fileName


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Main_Window = ImgRegistration()
    Main_Window.show()
    sys.exit(app.exec_())
