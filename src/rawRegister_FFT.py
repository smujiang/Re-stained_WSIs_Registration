import numpy as np
import os,sys
import cv2
import scipy as sp
import scipy.misc
import imreg_dft as ird
from PIL import Image
import time
import openslide
import matplotlib.pyplot as plt

methd = "FFT"
data_in_dir = "H:\\HE_IHC_Registration"
data_out_dir = "H:\\HE_IHC_Registration_result\\FFT"
sub_dir = ['HE_Caspase','HE_Ki67','HE_PHH3']
sub_dir_IHC = ['Caspase','Ki67','PHH3']
img_level = 4
img_cnt = 100
ground_truth_offset = [[146, 98], [-713, -85], [-295, -55]] #[x,y]
ground_truth_angles = [0, 0, 0]
tolerance_offset = 6  # registration error exceed this will be treat as failure
tolerance_angle = 1  # registration error exceed this will be treat as failure
for ihc_idx in range(len(sub_dir_IHC)):
    for lv_idx in range(img_level):
        failure_cnt = img_cnt
        offset_list = []
        angle_list = []
        score_list = []
        start_time = time.time()
        for img_idx in range(img_cnt):
            # img_idx = 8
            fix_img_name = os.path.join(data_in_dir,sub_dir[ihc_idx],"HE","level"+str(lv_idx),str(img_idx)+".jpg")
            float_img_name = os.path.join(data_in_dir, sub_dir[ihc_idx], sub_dir_IHC[ihc_idx], "level" + str(lv_idx),str(img_idx) + ".jpg")
            print(float_img_name)
            print(fix_img_name)
            Img_fix_col = Image.open(fix_img_name)
            Img_float_col = Image.open(float_img_name)
            Img_fix = sp.misc.fromimage(Img_fix_col,True)  # flatten is True, means we convert images into graylevel images.
            Img_float = sp.misc.fromimage(Img_float_col, True)
            # sim = ird.similarity(Img_fix, Img_float)
            con_s = dict(angle=[0, 0], scale=[1, 1])
            sim = ird.similarity(Img_fix, Img_float,constraints=con_s)
            # con_s= {"angle": 0, "scale": 0}
            # sim = ird.translation(Img_fix,Img_float,constraints=con_s)
            tvec = sim["tvec"].round(4)
            angle = sim["angle"]
            score = sim["success"]
            offset_list.append([tvec[1],tvec[0]])
            angle_list.append(angle)
            score_list.append(score)
            print("Translation is {}, angle is {}, success rate {:.4g}".format(tuple(tvec), angle, sim["success"]))
            TImg = sp.misc.imrotate(Img_float_col, sim["angle"])
            fix_img = np.array(Img_fix_col)
            M = np.float32([[1, 0, tvec[0]], [0, 1, tvec[1]]])
            dst = cv2.warpAffine(TImg, M, (Img_fix.shape[0], Img_fix.shape[1]))
            t_dst = np.array(dst)
            t_dst[:,0:int(Img_fix.shape[0]/2),:] = fix_img[:,0:int(Img_fix.shape[0]/2),:]
            #sp.misc.imsave(os.path.join(data_out_dir, "1col_fixed.png"), Img_fix_col)
            #sp.misc.imsave(os.path.join(data_out_dir, "3col_float.png"), Img_float_col)
            #sp.misc.imsave(os.path.join(data_out_dir, "2col_timg.png"), dst)
            os.makedirs(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx)),exist_ok=True)
            sp.misc.imsave(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx), str(img_idx)+".jpg"), t_dst)
            # if lv_idx == 0:  # only compare failure on leve 0
            angle_fit = (ground_truth_angles[ihc_idx]-tolerance_angle)< angle <(ground_truth_angles[ihc_idx]+tolerance_angle)
            offset_x_fit = ((ground_truth_offset[ihc_idx][0] - tolerance_offset) < tvec[0] < (ground_truth_offset[ihc_idx][0] + tolerance_offset))
            offset_y_fit = ((ground_truth_offset[ihc_idx][1]-tolerance_offset)< tvec[1]< (ground_truth_offset[ihc_idx][1]+tolerance_offset))
            if angle_fit&offset_x_fit&offset_y_fit:
                failure_cnt -= 1
        # save running log rate
        end_time = time.time()
        failure_rate = failure_cnt / img_cnt
        log_txt = os.path.join(data_out_dir, sub_dir[ihc_idx], "RunLog_" + methd + ".txt")
        with open(log_txt, "a", encoding="utf8") as log_fp:
            log_fp.write("level" + str(lv_idx)+"\n")
            log_fp.write("ExecutionTime(s):" + str(end_time-start_time)+"\n")
            log_fp.write("FailureRate:" + str(failure_rate) + "\n")
        # save results
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_offset_list.npz"), offset_list)
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_score_list.npz"),score_list)
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_angle_list.npz"),angle_list)

