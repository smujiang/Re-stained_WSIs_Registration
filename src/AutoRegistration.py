import openslide
import scipy as sp
import cv2
import scipy.misc
import imreg_dft as ird
import numpy as np
import imutils
from collections import Counter
from sklearn import linear_model
import random
import os
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.stats import gaussian_kde
from skimage.color import rgb2hsv
import time
import logging
from panorama import Stitcher


layer_patch_num = [100, 64, 16, 4]  # patch numbers per image level
patch_size = [1200, 800, 300, 200]

t0 = time.time()

def validate_WSI_info(WSI_fixed, WSI_float):
    if not WSI_fixed.level_count == WSI_float.level_count:
        return False
    fix_down = WSI_fixed.level_downsamples
    float_down = WSI_float.level_downsamples
    if not len(fix_down) == len(float_down):
        return False
    for idx, r in enumerate(fix_down):
        if not int(r) == int(float_down[idx]):
            return False
    return True


def histogram(iterable, low, high, bins):
    '''Count elements from the iterable into evenly spaced bins
        scores = [82, 85, 90, 91, 70, 87, 45]
        histogram(scores, 0, 100, 10)
        >>[0, 0, 0, 0, 1, 0, 0, 1, 3, 2]
    '''
    step = (high - low + 0.0) / bins
    dist = Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)]


def raw_reg(fixed_img, float_img, init_offset, down_rate, lv, method="FFT"):
    if type(fixed_img) == Image.Image:
        Img_fix = sp.misc.fromimage(fixed_img, True)  # flatten is True, means we convert images into graylevel images.
        Img_float = sp.misc.fromimage(float_img, True)
    else:
        Img_fix = fixed_img
        Img_float = float_img
    if method == "FFT":
        con_s = dict(angle=[0, 0], scale=[1, 1])
        sim = ird.similarity(Img_fix, Img_float, constraints=con_s)
        tvec = sim["tvec"].round(4)
        score = sim["success"]
        offset = [tvec[1], tvec[0]]
    elif method == "ECC":
        # warp_mode = cv2.MOTION_EUCLIDEAN
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        number_of_iterations = 50000  # Specify the number of iterations.
        termination_eps = 1e-8 # in the correlation coefficient between two iterations
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, number_of_iterations, termination_eps)  # Define termination criteria
        try:
            (score, warp_matrix) = cv2.findTransformECC(Img_fix, Img_float, warp_matrix, warp_mode, criteria)
            offset = [warp_matrix[0, 2], warp_matrix[1, 2]]
        except:
            print("unaligned")
            offset = [0, 0]
            score = 0
    elif method == "SIFT":
        Img_fix = np.array(fixed_img)
        Img_float = np.array(fixed_img)
        stitcher = Stitcher()
        matches, ptsA, ptsB, H, status = stitcher.returnMatchCoord([Img_fix, Img_float])  # here, already with RANSAC algorithm to match key points
        matched_ptsA = []
        matched_ptsB = []
        slops = []
        offsets = []
        for m_idx, m in enumerate(ptsA):
            if status[m_idx] == 1:
                matched_ptsA.append(m)
                matched_ptsB.append(ptsB[m_idx])
                s = (ptsB[m_idx][1] - m[1]) / (ptsB[m_idx][0] - m[0])
                offsetY = ptsB[m_idx][1] - m[1]
                offsetX = ptsB[m_idx][0] - m[0]
                slops.append(s)
                offsets.append([offsetX, offsetY])
        offset = np.mean(offsets, 0)  # use mean offset as offset
        score = np.mean(np.std(offsets, 0))  # use std as score
    elif method == "SIFT_ENH":
        stitcher = Stitcher()
        matches, ptsA, ptsB, H, status = stitcher.returnMatchCoord([Img_fix, Img_float])  # here, already with RANSAC algorithm to match key points
        matched_ptsA = []
        matched_ptsB = []
        slops = []
        offsets = []
        for m_idx, m in enumerate(ptsA):
            if status[m_idx] == 1:
                matched_ptsA.append(m)
                matched_ptsB.append(ptsB[m_idx])
                s = (ptsB[m_idx][1] - m[1]) / (ptsB[m_idx][0] - m[0])
                offsetY = ptsB[m_idx][1] - m[1]
                offsetX = ptsB[m_idx][0] - m[0]
                slops.append(s)
                offsets.append([offsetX, offsetY])
        # use a complicate way to distill matched key points
        max_slop = np.amax(slops)
        min_slop = np.amin(slops)
        bins = int(len(slops) / 2)
        slop_hist = histogram(slops, min_slop, max_slop, bins)
        step = (max_slop - min_slop) / bins
        idx_max_count = slop_hist.index(max(slop_hist))
        if type(idx_max_count) == list:
            idx_max_count = idx_max_count[0]
        low_range = min_slop + idx_max_count * step
        high_range = min_slop + (idx_max_count + 1) * step
        idx_s_list = []
        for idx_s, s in enumerate(slops):
            if low_range <= s <= high_range:
                idx_s_list.append(idx_s)
        offset = np.mean([offsets[i] for i in idx_s_list], 0)
        score = 1 / (np.mean(np.std([offsets[i] for i in idx_s_list], 0)) + 0.00000001)
    else:
        return [0, 0], 0
    offset = [offset[0] + init_offset[0]/down_rate[lv], offset[1] + init_offset[1]/down_rate[lv]]
    return offset, score


def get_initial_pos(WSI_fixed, WSI_float):
    print("Getting a raw initial position on thumbnail")
    WSI_Width, WSI_Height = WSI_fixed.dimensions
    thumb_size_x = round(WSI_Width / 100)
    thumb_size_y = round(WSI_Height / 100)
    thumbnail_fixed = WSI_fixed.get_thumbnail([thumb_size_x, thumb_size_y])
    thumbnail_fixed = thumbnail_fixed.convert('L')  # get grayscale image
    WSI_Width, WSI_Height = WSI_float.dimensions
    thumb_size_x = round(WSI_Width / 100)
    thumb_size_y = round(WSI_Height / 100)
    thumbnail_float = WSI_float.get_thumbnail([thumb_size_x, thumb_size_y])
    thumbnail_float = thumbnail_float.convert('L')  # get grayscale image
    img_w = min(thumbnail_fixed.width, thumbnail_float.width)
    img_h = min(thumbnail_fixed.height, thumbnail_float.height)
    img_size = min(img_w, img_h)
    fixed_array = np.array(thumbnail_fixed)
    float_array = np.array(thumbnail_float)
    fixed_img = Image.fromarray(fixed_array[0:img_size,0:img_size])
    float_img = Image.fromarray(float_array[0:img_size,0:img_size])
    init_offset, _ = raw_reg(fixed_img, float_img, [0, 0],[1,1],0, method="FFT")
    print("Thumbnail offset: [%f, %f]" % (init_offset[0], init_offset[1]))
    init_offset = [init_offset[0]*100, init_offset[1]*100]
    return init_offset


def getROIs(WSI_Img, init_offset, level=0):
    locations = []
    WSI_Width, WSI_Height = WSI_Img.dimensions
    thumb_size_x = int(WSI_Width / 100)
    thumb_size_y = int(WSI_Height / 100)
    thumbnail = WSI_Img.get_thumbnail([thumb_size_x, thumb_size_y])
    rgb_image_array = np.array(thumbnail)
    hsv_img = rgb2hsv(rgb_image_array)
    value_img = hsv_img[:, :, 2]
    binary_img = value_img < 0.8  # brightness higher than 0.8
    binary_img = ndimage.binary_erosion(binary_img, structure=np.ones((5,5)))
    cnt = 0
    while cnt < layer_patch_num[level]:
        x = random.randint(2, thumb_size_x - 2)
        y = random.randint(2, thumb_size_y - 2)
        if binary_img[y, x]:
            loc_x = int((x * 100))
            loc_y = int((y * 100))
            locations.append([loc_y, loc_x])
            cnt += 1
    return locations


def change_offset(HE_offsets, init_offset):
    IHC_offsets = []
    for HEoff in HE_offsets:
        IHC_offsets.append([int(HEoff[0]-init_offset[1]), int(HEoff[1]-init_offset[0])])
    return IHC_offsets


def getROIs_debug(WSI_Img, init_offset, debug_out_file, level=0):
    colors = ["black", "red", "green", "yellow"]
    locations = []
    WSI_Width, WSI_Height = WSI_Img.dimensions
    thumb_size_x = int(WSI_Width / 100)
    thumb_size_y = int(WSI_Height / 100)
    thumbnail = WSI_Img.get_thumbnail([thumb_size_x, thumb_size_y])
    rgb_image_array = np.array(thumbnail)
    hsv_img = rgb2hsv(rgb_image_array)
    value_img = hsv_img[:, :, 2]
    binary_img = value_img < 0.8  # brightness higher than 0.8
    draw = ImageDraw.Draw(thumbnail)
    cnt = 0
    while cnt < layer_patch_num[level]:
        x = random.randint(0, thumb_size_x-1)
        y = random.randint(0, thumb_size_y-1)
        #draw.line((x - 15, y, x + 15, y), fill="green", width=10)
        if binary_img[y, x]:
            draw.line((x - 8, y, x + 8, y), fill=colors[level], width=17)
            loc_x = int((x * 100) - init_offset[0])
            loc_y = int((y * 100) - init_offset[1])
            locations.append([loc_y, loc_x])
            cnt += 1
    del draw
    thumbnail.save(debug_out_file, "png")
    return locations

def get_img_patches(WSI_Img, locations, p_size, level=0, mod="IHC"):
    Imgs = []
    cnt = 0
    for y,x in locations:
        Img_patch = WSI_Img.read_region((x, y), level, (p_size, p_size))
        Img_patch = Img_patch.convert("RGB")
        #print("Location: %d, %d; Patch size: %d" % (y, x, p_size))
        # if level == 2:
        #     path_save = "H:\\HE_IHC_Stains\\patches"
        #     Img_patch.save(os.path.join(path_save, mod+"_"+str(cnt)+"_"+str(x)+"_"+str(y)+"_lv"+str(level)+".jpg"))
        Imgs.append(Img_patch)
        cnt += 1
    return Imgs


def raw_reg_batch(fixed_imgs, float_imgs, init_offset, down_rate, level=0, method="m"):
    offsets = []
    scores = []
    for idx in range(len(fixed_imgs)):
        offset, score = raw_reg(fixed_imgs[idx], float_imgs[idx], init_offset, down_rate, level, method=method)
        offsets.append(offset)
        scores.append(score)
    return offsets, scores


def offset_kde(offsets):
    a = np.array(offsets)[:, 0]
    b = np.array(offsets)[:, 1]
    xy = np.vstack([a, b])
    kde_scores = gaussian_kde(xy)(xy)
    return kde_scores


def norm(rvalue, newmin, newmax):
    oldmin = min(rvalue)
    oldmax = max(rvalue)
    oldrange = oldmax - oldmin
    # newmin = 0.
    # newmax = 1.
    newrange = newmax - newmin
    if oldrange == 0:  # Deal with the case where rvalue is constant:
        if oldmin < newmin:  # If rvalue < newmin, set all rvalue values to newmin
            newval = newmin
        elif oldmin > newmax:  # If rvalue > newmax, set all rvalue values to newmax
            newval = newmax
        else:  # If newmin <= rvalue <= newmax, keep rvalue the same
            newval = oldmin
        normal = [newval for v in rvalue]
    else:
        scale = newrange / oldrange
        normal = [(v - oldmin) * scale + newmin for v in rvalue]

    return normal


# get the most likely points,and calculate the mean and std
# parameters:
# X,Y: x,y coordinate
# cv: confidence value
# Thr: the confidence threshold (eg. 0.65)
def getEstimation(X, Y, cv, Thr):
    if (not len(X) == len(Y)) or (not len(X) == len(cv)):
        print("X dim: %d, Y dim: %d, Score dim: %d" % (len(X), len(Y), len(cv)))
        raise Exception("Offset and score dimension not match")
    else:
        ncv = norm(cv, 0.0, 1.0)
        idx = [n for n, i in enumerate(ncv) if i > Thr]
        newX = X[idx]
        newY = Y[idx]
        avgX = np.mean(newX)
        avgY = np.mean(newY)
        stdX = np.std(newX)
        stdY = np.std(newY)
        return newX, newY, avgX, avgY, stdX, stdY


def HL_fit(offsets, scores,level_ratio):
    all_offsets = np.concatenate((offsets[0],offsets[1],offsets[2]))
    x_np = all_offsets[:, 0].reshape(-1, 1)
    y_np = all_offsets[:, 1].reshape(-1, 1)
    all_scores = np.concatenate((scores[0],scores[1],scores[2]))
    w_np = all_scores.flatten()
    regr_w = linear_model.LinearRegression(fit_intercept=False)
    k_s_w = regr_w.fit(x_np, y_np, sample_weight=w_np * 10)
    slop_s_w = k_s_w.coef_[0][0]
    # get final estimation
    key_layer_x = np.array(offsets[2])[:, 0]
    key_layer_y = np.array(offsets[2])[:, 1]
    newX, newY, avgX, avgY, stdX, stdY = getEstimation(key_layer_x, key_layer_y, scores[2], 0.65)
    est_x_lv0_k_a = avgX * level_ratio[1]
    est_y_lv0_k_b = est_x_lv0_k_a * slop_s_w
    est_y_lv0_k_a = avgY * level_ratio[1]
    est_x_lv0_k_b = est_y_lv0_k_a / slop_s_w
    k_est_x = round((est_x_lv0_k_a + est_x_lv0_k_b) / 2)
    k_est_y = round((est_y_lv0_k_a + est_y_lv0_k_b) / 2)
    refined_offsets = [k_est_x,k_est_y]
    return refined_offsets


RELEASE = False
VERBOSE = False
DRAW_FIG = False


################################################
# HE_Img_name = "H:\\HE_IHC_Stains\\Merkel-CC_CR16-1790-A2_HE-Cleaved-Caspase3.svs"
# IHC_Img_name = "H:\\HE_IHC_Stains\\Merkel-CC_CR16-1790-A2_Cleaved-Caspase3.svs"

# HE_Img_name = "H:\\HE_IHC_Stains\\Merkel-CC_CR16-1790-A2_HE-PHH3.svs"
# IHC_Img_name = "H:\\HE_IHC_Stains\\Merkel-CC_CR16-1790-A2_PHH3.svs"

def match_WSI(HE_Img_name, IHC_Img_name, save_to_txt):
    HE_n = os.path.split(HE_Img_name)[1]
    IHC_n = os.path.split(IHC_Img_name)[1]
    print("Processing %s & %s" % (HE_n, IHC_n))

    HE_Img = openslide.open_slide(HE_Img_name)
    IHC_Img = openslide.open_slide(IHC_Img_name)

    print("Checking image information")
    # check image information
    if not validate_WSI_info(HE_Img, IHC_Img):
        raise Exception("Float image and fixed image don't share some essential properties")

    # get initial position, just in case the initial offset is too large
    init_offset = get_initial_pos(HE_Img, IHC_Img)

    locations_lv3 = getROIs(HE_Img, [0,0], level=3)
    locations_lv2 = getROIs(HE_Img, [0,0], level=2)
    locations_lv1 = getROIs(HE_Img, [0,0], level=1)
    if VERBOSE:
        locations_lv0 = getROIs(HE_Img, [0,0], level=0)

    if patch_size[0] < abs(init_offset[0]) or patch_size[0] < abs(init_offset[1]):
        logging.warning("Patch size smaller than offset")
        locations_lv3_IHC = change_offset(locations_lv3, init_offset)
        locations_lv2_IHC = change_offset(locations_lv2, init_offset)
        locations_lv1_IHC = change_offset(locations_lv1, init_offset)
        if VERBOSE:
            locations_lv0_IHC = change_offset(locations_lv0, init_offset)
    else:
        init_offset = [0, 0]
        locations_lv3_IHC = locations_lv3
        locations_lv2_IHC = locations_lv2
        locations_lv1_IHC = locations_lv1
        if VERBOSE:
            locations_lv0_IHC = locations_lv0
    # if not RELEASE:
    #     out_file = os.path.join("H://HE_IHC_Stains/thumbnails_rois", HE_n[:-4]+"lv3.png")
    #     locations_lv3 = getROIs_debug(HE_Img, init_offset, out_file, level=3)
    #     out_file = os.path.join("H://HE_IHC_Stains/thumbnails_rois", HE_n[:-4] + "lv2.png")
    #     locations_lv2 = getROIs_debug(HE_Img, init_offset, out_file, level=2)
    #     out_file = os.path.join("H://HE_IHC_Stains/thumbnails_rois", HE_n[:-4] + "lv1.png")
    #     locations_lv1 = getROIs_debug(HE_Img, init_offset, out_file, level=1)
    #     if VERBOSE:
    #         locations_lv0 = getROIs_debug(HE_Img, init_offset, out_file, level=0)

    print("Spend %s s" % str(time.time()-t0))

    print("Getting image patches from WSI")
    #######################################################
    # extract patches from image levels
    HE_Imgs_lv3 = get_img_patches(HE_Img, locations_lv3, patch_size[3], level=3, mod="HE")
    HE_Imgs_lv2 = get_img_patches(HE_Img, locations_lv2, patch_size[2], level=2, mod="HE")
    HE_Imgs_lv1 = get_img_patches(HE_Img, locations_lv1, patch_size[1], level=1, mod="HE")
    if VERBOSE:
        HE_Imgs_lv0 = get_img_patches(HE_Img, locations_lv0, patch_size[0], level=0, mod="HE")

    IHC_Imgs_lv3 = get_img_patches(IHC_Img, locations_lv3_IHC, patch_size[3], level=3, mod="IHC")
    IHC_Imgs_lv2 = get_img_patches(IHC_Img, locations_lv2_IHC, patch_size[2], level=2, mod="IHC")
    IHC_Imgs_lv1 = get_img_patches(IHC_Img, locations_lv1_IHC, patch_size[1], level=1, mod="IHC")
    if VERBOSE:
        IHC_Imgs_lv0 = get_img_patches(IHC_Img, locations_lv0_IHC, patch_size[0], level=0, mod="IHC")
    #######################################################
    print("Spend %s s" % str(time.time()-t0))
    print("Getting raw registration")
    downsample_rate = HE_Img.level_downsamples
    # FFT
    print("FFT method")
    Raw_FFT_lv3, scores_FFT_lv3 = raw_reg_batch(HE_Imgs_lv3, IHC_Imgs_lv3, init_offset, downsample_rate, level=3, method="FFT")
    Raw_FFT_lv2, scores_FFT_lv2 = raw_reg_batch(HE_Imgs_lv2, IHC_Imgs_lv2, init_offset, downsample_rate, level=2, method="FFT")
    Raw_FFT_lv1, scores_FFT_lv1 = raw_reg_batch(HE_Imgs_lv1, IHC_Imgs_lv1, init_offset, downsample_rate, level=1, method="FFT")
    if VERBOSE:
        # Prove that it's impossible to directly match at highest resolution
        Raw_FFT_lv0, scores_FFT_lv0 = raw_reg_batch(HE_Imgs_lv0, IHC_Imgs_lv0,init_offset, downsample_rate, level=0, method="FFT")

    if not RELEASE:
        # ECC
        print("ECC method")
        Raw_ECC_lv3, scores_ECC_lv3 = raw_reg_batch(HE_Imgs_lv3, IHC_Imgs_lv3,init_offset, downsample_rate, level=3, method="ECC")
        Raw_ECC_lv2, scores_ECC_lv2 = raw_reg_batch(HE_Imgs_lv2, IHC_Imgs_lv2,init_offset, downsample_rate, level=2, method="ECC")
        Raw_ECC_lv1, scores_ECC_lv1 = raw_reg_batch(HE_Imgs_lv1, IHC_Imgs_lv1,init_offset, downsample_rate, level=1, method="ECC")
        if VERBOSE:
            Raw_ECC_lv0, scores_ECC_lv0 = raw_reg_batch(HE_Imgs_lv0, IHC_Imgs_lv0,init_offset, downsample_rate, level=0, method="ECC")
        # SIFT
        print("SIFT method")
        Raw_SIFT_lv3, scores_SIFT_lv3 = raw_reg_batch(HE_Imgs_lv3, IHC_Imgs_lv3,init_offset, downsample_rate, level=3, method="SIFT")
        Raw_SIFT_lv2, scores_SIFT_lv2 = raw_reg_batch(HE_Imgs_lv2, IHC_Imgs_lv2,init_offset, downsample_rate, level=2, method="SIFT")
        Raw_SIFT_lv1, scores_SIFT_lv1 = raw_reg_batch(HE_Imgs_lv1, IHC_Imgs_lv1,init_offset, downsample_rate, level=1, method="SIFT")
        if VERBOSE:
            Raw_SIFT_lv0, scores_SIFT_lv0 = raw_reg_batch(HE_Imgs_lv0, IHC_Imgs_lv0,init_offset, downsample_rate, level=0, method="SIFT")
        # SIFT_ENH
        print("SIFT_ENH method")
        Raw_SIFT_ENH_lv3, scores_SIFT_ENH_lv3 = raw_reg_batch(HE_Imgs_lv3, IHC_Imgs_lv3,init_offset, downsample_rate, level=3, method="SIFT_ENH")
        Raw_SIFT_ENH_lv2, scores_SIFT_ENH_lv2 = raw_reg_batch(HE_Imgs_lv2, IHC_Imgs_lv2,init_offset, downsample_rate, level=2, method="SIFT_ENH")
        Raw_SIFT_ENH_lv1, scores_SIFT_ENH_lv1 = raw_reg_batch(HE_Imgs_lv1, IHC_Imgs_lv1,init_offset, downsample_rate, level=1, method="SIFT_ENH")
        if VERBOSE:
            Raw_SIFT_ENH_lv0, scores_SIFT_ENH_lv0 = raw_reg_batch(HE_Imgs_lv0, IHC_Imgs_lv0,init_offset, downsample_rate, level=0, method="SIFT_ENH")
    #######################################################
    # KDE
    KDE_weights_FFT_lv3 = offset_kde(Raw_FFT_lv3)
    KDE_weights_FFT_lv2 = offset_kde(Raw_FFT_lv2)
    KDE_weights_FFT_lv1 = offset_kde(Raw_FFT_lv1)
    if not RELEASE:
        KDE_weights_ECC_lv3 = offset_kde(Raw_ECC_lv3)
        KDE_weights_ECC_lv2 = offset_kde(Raw_ECC_lv2)
        KDE_weights_ECC_lv1 = offset_kde(Raw_ECC_lv1)

        KDE_weights_SIFT_lv3 = offset_kde(Raw_SIFT_lv3)
        KDE_weights_SIFT_lv2 = offset_kde(Raw_SIFT_lv2)
        KDE_weights_SIFT_lv1 = offset_kde(Raw_SIFT_lv1)

        KDE_weights_SIFT_ENH_lv3 = offset_kde(Raw_SIFT_ENH_lv3)
        KDE_weights_SIFT_ENH_lv2 = offset_kde(Raw_SIFT_ENH_lv2)
        KDE_weights_SIFT_ENH_lv1 = offset_kde(Raw_SIFT_ENH_lv1)

        # draw figures
        print("Draw evaluation figures")
    print("Spend %s s" % str(time.time()-t0))
    print("Getting refined offset")
    #######################################################

    # regression
    KDE_offset_FFT = HL_fit([Raw_FFT_lv3, Raw_FFT_lv2, Raw_FFT_lv1], [KDE_weights_FFT_lv3, KDE_weights_FFT_lv2, KDE_weights_FFT_lv1],downsample_rate)
    Score_offset_FFT = HL_fit([Raw_FFT_lv3, Raw_FFT_lv2, Raw_FFT_lv1], [scores_FFT_lv3, scores_FFT_lv2, scores_FFT_lv1],downsample_rate)
    if not RELEASE:
        KDE_offset_ECC = HL_fit([Raw_ECC_lv3, Raw_ECC_lv2, Raw_ECC_lv1], [KDE_weights_ECC_lv3, KDE_weights_ECC_lv2, KDE_weights_ECC_lv1])
        Score_offset_ECC = HL_fit([Raw_ECC_lv3, Raw_ECC_lv2, Raw_ECC_lv1], [scores_ECC_lv3, scores_ECC_lv2, scores_ECC_lv1])
        KDE_offset_SIFT = HL_fit([Raw_SIFT_lv3, Raw_SIFT_lv2, Raw_SIFT_lv1], [KDE_weights_SIFT_lv3, KDE_weights_SIFT_lv2, KDE_weights_SIFT_lv1])
        Score_offset_SIFT = HL_fit([Raw_SIFT_lv3, Raw_SIFT_lv2, Raw_SIFT_lv1], [scores_SIFT_lv3, scores_SIFT_lv2, scores_SIFT_lv1])
        KDE_offset_SIFT_ENH = HL_fit([Raw_SIFT_ENH_lv3, Raw_SIFT_ENH_lv2, Raw_SIFT_ENH_lv1],[KDE_weights_SIFT_ENH_lv3, KDE_weights_SIFT_ENH_lv2, KDE_weights_SIFT_ENH_lv1])
        Score_offset_SIFT_ENH = HL_fit([Raw_SIFT_ENH_lv3, Raw_SIFT_ENH_lv2, Raw_SIFT_ENH_lv1], [scores_SIFT_ENH_lv3, scores_SIFT_ENH_lv2, scores_SIFT_ENH_lv1])
    if VERBOSE:
        # draw figures
        print("Draw evaluation figures")
    print("KDE result: %d, %d" % (int(KDE_offset_FFT[0]), int(KDE_offset_FFT[1])))
    print("Similarity result: %d, %d" % (int(Score_offset_FFT[0]), int(Score_offset_FFT[1])))
    print("Spend %s s" % str(time.time()-t0))

    f_ele = os.path.split(save_to_txt)
    sv_fn = os.path.splitext(f_ele[1])
    save_to_txt_fft = os.path.join(f_ele[0], sv_fn[0]+"_fft.csv")
    fp_sv = open(save_to_txt_fft, 'a')
    wrt_str = ",".join([HE_n, IHC_n, str(KDE_offset_FFT[0]),str(KDE_offset_FFT[1]), str(Score_offset_FFT[0]), str(Score_offset_FFT[1])])
    fp_sv.write(wrt_str+"\n")
    if VERBOSE:
        wrt_str = ",".join([HE_n, IHC_n, str(KDE_offset_ECC[0]),str(KDE_offset_ECC[1]), str(Score_offset_ECC[0]), str(Score_offset_ECC[1])])
        save_to_txt_ecc = os.path.join(f_ele[0], sv_fn[0] + "_ecc.csv")
        fp_sv = open(save_to_txt_ecc, 'a')
        fp_sv.write(wrt_str+"\n")

        wrt_str = ",".join([HE_n, IHC_n, str(KDE_offset_SIFT[0]),str(KDE_offset_SIFT[1]), str(Score_offset_SIFT[0]), str(Score_offset_SIFT[1])])
        save_to_txt_sift = os.path.join(f_ele[0], sv_fn[0] + "_sift.csv")
        fp_sv = open(save_to_txt_sift, 'a')
        fp_sv.write(wrt_str+"\n")

        wrt_str = ",".join([HE_n, IHC_n, str(KDE_offset_SIFT_ENH[0]),str(KDE_offset_SIFT_ENH[1]), str(Score_offset_SIFT_ENH[0]), str(Score_offset_SIFT_ENH[1])])
        save_to_txt_sift_enh = os.path.join(f_ele[0], sv_fn[0] + "_sift_enh.csv")
        fp_sv = open(save_to_txt_sift_enh, 'a')
        fp_sv.write(wrt_str+"\n")
    fp_sv.close()
















