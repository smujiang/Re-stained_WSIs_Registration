'''
The ECC image alignment algorithm introduced in OpenCV 3 is based on a 2008 paper
titled Parametric Image Alignment using Enhanced Correlation Coefficient Maximization
by Georgios D. Evangelidis and Emmanouil Z. Psarakis. They propose using a new 
similarity measure called Enhanced Correlation Coefficient (ECC) for estimating 
the parameters of the motion model. 
'''

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


methd = "ECC"
data_in_dir = "H:\\HE_IHC_Registration"
data_out_dir = "H:\\HE_IHC_Registration_result\\ECC"
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
            fix_img_name = os.path.join(data_in_dir,sub_dir[ihc_idx],"HE","level"+str(lv_idx),str(img_idx)+".jpg")
            float_img_name = os.path.join(data_in_dir, sub_dir[ihc_idx], sub_dir_IHC[ihc_idx], "level" + str(lv_idx),str(img_idx) + ".jpg")
            print(float_img_name)
            print(fix_img_name)
            Img_fix_col = Image.open(fix_img_name)
            Img_float_col = Image.open(float_img_name)
            im1 = np.array(Img_fix_col)  # flatten is True, means we convert images into graylevel images.
            im2 = np.array(Img_float_col)
            im1_gray = sp.misc.fromimage(Img_fix_col,True)  # flatten is True, means we convert images into graylevel images.
            im2_gray = sp.misc.fromimage(Img_float_col, True)

            # Find size of image1
            sz = im1.shape

            # Define the motion model
            # warp_mode = cv2.MOTION_EUCLIDEAN
            warp_mode = cv2.MOTION_TRANSLATION
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            # Specify the number of iterations.
            number_of_iterations = 50000
            # Specify the threshold of the increment
            # in the correlation coefficient between two iterations
            termination_eps = 1e-8

            # Define termination criteria
            criteria = (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, number_of_iterations,termination_eps)
            print("processing " + sub_dir[ihc_idx] + " " + str(img_idx) + " " + "level" + str(lv_idx))
            # Run the ECC algorithm. The results are stored in warp_matrix.
            offsetY = 0
            offsetX = 0
            score = 0
            try:
                (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray, warp_matrix, warp_mode, criteria)
                offsetY = warp_matrix[1,2]
                offsetX = warp_matrix[0,2]
                score = cc
            except:
                print("unaligned")
                offsetY = 0
                offsetX = 0
                score = 0

            angle = 0  # no rotation expected
            tvec = [offsetX,offsetY]  # use mean offset as offset
            offset_list.append(tvec)
            score_list.append(score)
            angle_list.append(angle)
            print("Translation is {}, angle is {}, success rate {:.4g}".format(tuple(tvec), angle, score))
            # Use warpAffine for Translation, Euclidean and Affine
            # im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            # Show final results
            # cv2.imshow("Image 1", cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
            # cv2.imshow("Image 2", cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
            # cv2.imshow("Aligned Image 2", cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)
        # save running log rate
        end_time = time.time()
        failure_rate = failure_cnt / img_cnt
        log_txt = os.path.join(data_out_dir, sub_dir[ihc_idx], "RunLog_" + methd + ".txt")
        with open(log_txt, "a", encoding="utf8") as log_fp:
            log_fp.write("level" + str(lv_idx) + "\n")
            log_fp.write("ExecutionTime(s):" + str(end_time - start_time) + "\n")
            log_fp.write("FailureRate:" + str(failure_rate) + "\n")
        # save results
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_offset_list.npz"),offset_list)
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_score_list.npz"),score_list)
        np.save(os.path.join(data_out_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methd + "_angle_list.npz"),angle_list)

