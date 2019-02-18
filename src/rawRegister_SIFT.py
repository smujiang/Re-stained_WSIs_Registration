import numpy as np
import os,sys
import cv2
import scipy as sp
import scipy.misc
import imreg_dft as ird
from imreg import register, sampler, model, metric
from PIL import Image
import math
from panorama import Stitcher
from collections import Counter
import time
import openslide
import matplotlib.pyplot as plt

# https://github.com/vishwa91/pyimreg
def histogram(iterable, low, high, bins):
    '''Count elements from the iterable into evenly spaced bins
        scores = [82, 85, 90, 91, 70, 87, 45]
        histogram(scores, 0, 100, 10)
        >>[0, 0, 0, 0, 1, 0, 0, 1, 3, 2]
    '''
    step = (high - low + 0.0) / bins
    dist = Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)]


methd = "SIFT"
data_in_dir = "H:\\HE_IHC_Registration"
data_out_dir = "H:\\HE_IHC_Registration_result\\SIFT"
sub_dir = ['HE_Caspase','HE_Ki67','HE_PHH3']
sub_dir_IHC = ['Caspase','Ki67','PHH3']
img_level = 4
img_cnt = 100
ground_truth_offset = [[146, 98], [-713, -85], [-295, -55]]  #[x,y]
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
            fix_img_name = os.path.join(data_in_dir,sub_dir[ihc_idx],"HE","level"+str(lv_idx),str(img_idx)+".jpg")
            float_img_name = os.path.join(data_in_dir, sub_dir[ihc_idx], sub_dir_IHC[ihc_idx], "level" + str(lv_idx),str(img_idx) + ".jpg")
            print(float_img_name)
            print(fix_img_name)
            Img_fix_col = Image.open(fix_img_name)
            Img_float_col = Image.open(float_img_name)
            # Img_fix = sp.misc.fromimage(Img_fix_col,True)  # flatten is True, means we convert images into graylevel images.
            # Img_float = sp.misc.fromimage(Img_float_col,True)
            Img_fix = np.array(Img_fix_col)  # flatten is True, means we convert images into graylevel images.
            Img_float = np.array(Img_float_col)
            stitcher = Stitcher()

            # (result, vis) = stitcher.stitch([Img_fix, Img_float], showMatches=True)
            # cv2.imwrite("ImageA.jpg", cv2.cvtColor(Img_fix, cv2.COLOR_BGR2RGB))
            # cv2.imwrite("ImageB.jpg", cv2.cvtColor(Img_float, cv2.COLOR_BGR2RGB))
            # cv2.imwrite("matches.jpg", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            # cv2.imwrite("Result.jpg", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            # print("OK")

            '''
            if the rotation angle is confirmed to be 0, we can use below method to distill offset.
            '''
            matches, ptsA, ptsB, H, status = stitcher.returnMatchCoord([Img_fix, Img_float]) # here, already with RANSAC algorithm to match key points
            matched_ptsA = []
            matched_ptsB = []
            slops = []
            offsets = []
            for m_idx, m in enumerate(ptsA):
                if status[m_idx] == 1:
                    matched_ptsA.append(m)
                    matched_ptsB.append(ptsB[m_idx])
                    s = (ptsB[m_idx][1]-m[1])/(ptsB[m_idx][0]-m[0])
                    offsetY = ptsB[m_idx][1]-m[1]
                    offsetX = ptsB[m_idx][0]-m[0]
                    slops.append(s)
                    offsets.append([offsetX, offsetY])
            # # an alternative way is just use the mean
            # mean_slop = np.mean(slops)
            angle = 0  # no rotation expected
            tvec = np.mean(offsets, 0)   # use mean offset as offset
            score = np.mean(np.std(offsets, 0))  # use std as score

            offset_list.append(tvec)
            score_list.append(score)
            angle_list.append(angle)

            angle_fit = (ground_truth_angles[ihc_idx] - tolerance_angle) < angle < (ground_truth_angles[ihc_idx] + tolerance_angle)
            offset_x_fit = ((ground_truth_offset[ihc_idx][0] - tolerance_offset) < tvec[0] < (ground_truth_offset[ihc_idx][0] + tolerance_offset))
            offset_y_fit = ((ground_truth_offset[ihc_idx][1] - tolerance_offset) < tvec[1] < (ground_truth_offset[ihc_idx][1] + tolerance_offset))
            if angle_fit & offset_x_fit & offset_y_fit:
                failure_cnt -= 1
        # save running log rate
        end_time = time.time()
        failure_rate = failure_cnt / img_cnt
        log_txt = os.path.join(data_out_dir, sub_dir[ihc_idx], "RunLog_" + methd + ".txt")
        with open(log_txt, "a", encoding="utf8") as log_fp:
            log_fp.write("level" + str(lv_idx) + "\n")
            log_fp.write("ExecutionTime(s):" + str(end_time-start_time) + "\n")
            log_fp.write("FailureRate:" + str(failure_rate) + "\n")
        # save results
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_offset_list.npz"),offset_list)
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_score_list.npz"),score_list)
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_angle_list.npz"),angle_list)

        # (result, vis) = stitcher.stitch([Img_fix, Img_float], showMatches=True)
        # cv2.imwrite("ImageA.jpg", cv2.cvtColor(Img_fix, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("ImageB.jpg", cv2.cvtColor(Img_float, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("matches.jpg", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("Result.jpg", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # print("OK")

