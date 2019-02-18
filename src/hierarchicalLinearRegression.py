import numpy as np
import os, sys
from scipy.stats import gaussian_kde
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

methods = ["FFT", "SIFT_ENH","SIFT","ECC"]
data_in_dir = "H:\\HE_IHC_Registration_result\\out_data"
figure_out_dir = "H:\\HE_IHC_Registration_result\\final_figures"
sub_dir = ['HE_Caspase', 'HE_Ki67', 'HE_PHH3']
sub_dir_IHC = ['Caspase', 'Ki67', 'PHH3']
plot_marker = ['.', 'o', '>', '+']
marker_color = ['r', 'g', 'b', 'y']
img_level = 4
img_cnt = 100
ground_truth_offset = [[146, 98], [-713, -85], [-295, -55]]  # [x,y]
level_ratio = [1, 4, 16, 32]
ground_truth_angles = [0, 0, 0]
tolerance_offset = 6  # registration error exceed this will be treat as failure
tolerance_angle = 1  # registration error exceed this will be treat as failure


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
    ncv = norm(cv, 0.0, 1.0)
    idx = [n for n, i in enumerate(ncv) if i > Thr]
    newX = X[idx]
    newY = Y[idx]
    avgX = np.mean(newX)
    avgY = np.mean(newY)
    stdX = np.std(newX)
    stdY = np.std(newY)
    return newX, newY, avgX, avgY, stdX, stdY

Thr = 0.75
for m_idx in range(len(methods)):
    # m_idx =1
    print(methods[m_idx])
    for ihc_idx in range(len(sub_dir_IHC)):
        # ihc_idx = 0
        # use similarity score
        # X_train_list_score = []
        # Y_train_list_score = []
        # Score_list_score = []
        plt.figure(1,[5,4])
        plt.title("%s_based Cross Level Estimation(%s)" % (methods[m_idx],sub_dir[ihc_idx]))

        '''
        use similarity as weight
        '''
        similar_offset_XY_l3 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level3_Score_" + methods[m_idx] + "_offset_list.npz.npy"))
        similar_score_XY_l3 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level3_Score_" + methods[m_idx] + "_score_list.npz.npy"))
        X_train_l3_list_score = list(similar_offset_XY_l3[1, :]*level_ratio[3]/level_ratio[2])
        Y_train_l3_list_score = list(similar_offset_XY_l3[0, :]*level_ratio[3]/level_ratio[2])
        plt.scatter(similar_offset_XY_l3[1, :], similar_offset_XY_l3[0, :], marker=plot_marker[3], c=marker_color[3])
        Score_list_score_l3 = list(similar_score_XY_l3)

        similar_offset_XY_l2 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level2_Score_" + methods[m_idx] + "_offset_list.npz.npy"))
        similar_score_XY_l2 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level2_Score_" + methods[m_idx] + "_score_list.npz.npy"))
        X_train_l2_list_score = list(similar_offset_XY_l2[1, :])
        Y_train_l2_list_score = list(similar_offset_XY_l2[0, :])
        plt.scatter(similar_offset_XY_l2[1, :], similar_offset_XY_l2[0, :], marker=plot_marker[2], c=marker_color[2])
        Score_list_score_l2 = list(similar_score_XY_l2)

        similar_offset_XY_l1 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level1_Score_" + methods[m_idx] + "_offset_list.npz.npy"))
        similar_score_XY_l1 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level1_Score_" + methods[m_idx] + "_score_list.npz.npy"))
        X_train_l1_list_score = list(similar_offset_XY_l1[1, :]*level_ratio[1]/level_ratio[2])
        Y_train_l1_list_score = list(similar_offset_XY_l1[0, :]*level_ratio[1]/level_ratio[2])
        plt.scatter(similar_offset_XY_l1[1, :], similar_offset_XY_l1[0, :], marker=plot_marker[1], c=marker_color[1])
        Score_list_score_l1 = list(similar_score_XY_l1)

        x_np = np.concatenate([np.array(xi) for xi in [X_train_l1_list_score, X_train_l2_list_score, X_train_l3_list_score]]).reshape(-1, 1)
        y_np = np.concatenate([np.array(xi) for xi in [Y_train_l1_list_score, Y_train_l2_list_score, Y_train_l3_list_score]]).reshape(-1, 1)
        w_np = np.concatenate([np.array(xi) for xi in [Score_list_score_l1, Score_list_score_l2, Score_list_score_l3]])

        regr_w = linear_model.LinearRegression(fit_intercept=False)
        k_s_w = regr_w.fit(x_np, y_np, sample_weight=w_np * 10)
        slop_s_w = k_s_w.coef_[0][0]
        g_k = ground_truth_offset[ihc_idx][0] / ground_truth_offset[ihc_idx][1]
        # draw figure
        if ground_truth_offset[ihc_idx][1] >= 0:
            x = range(ground_truth_offset[ihc_idx][1]+10)
        else:
            x = range(ground_truth_offset[ihc_idx][1]-10, -1)
        y_s_w = slop_s_w*x
        plt.plot(x, y_s_w, 'r', linewidth=0.5)

        # get final estimation
        score_norm = norm(Score_list_score_l2, 0, 1)
        newX, newY, avgX, avgY, stdX, stdY = getEstimation(similar_offset_XY_l2[1, :], similar_offset_XY_l2[0, :], score_norm, Thr)
        est_x_lv0_s_a = avgX * level_ratio[2]
        est_y_lv0_s_b = est_x_lv0_s_a * slop_s_w

        est_y_lv0_s_a = avgY * level_ratio[2]
        est_x_lv0_s_b = est_y_lv0_s_a / slop_s_w

        s_est_x = round((est_x_lv0_s_a + est_x_lv0_s_b) / 2)
        s_est_y = round((est_y_lv0_s_a + est_y_lv0_s_b) / 2)

        print("OK")
        '''
        # use KDE as weight
        '''
        KDE_offset_XY_l3 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level3_KDE_" + methods[m_idx] + "_offset_list.npz.npy"))
        KDE_score_XY_l3 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level3_KDE_" + methods[m_idx] + "_score_list.npz.npy"))
        X_train_l3_list_score = list(KDE_offset_XY_l3[1, :] * level_ratio[3] / level_ratio[2])
        Y_train_l3_list_score = list(KDE_offset_XY_l3[0, :] * level_ratio[3] / level_ratio[2])
        plt.scatter(KDE_offset_XY_l3[1, :], KDE_offset_XY_l3[0, :], marker=plot_marker[3], c=marker_color[3])
        Score_list_score_l3 = list(KDE_score_XY_l3)

        KDE_offset_XY_l2 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level2_KDE_" + methods[m_idx] + "_offset_list.npz.npy"))
        KDE_score_XY_l2 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level2_KDE_" + methods[m_idx] + "_score_list.npz.npy"))
        X_train_l2_list_score = list(KDE_offset_XY_l2[1, :])
        Y_train_l2_list_score = list(KDE_offset_XY_l2[0, :])
        plt.scatter(KDE_offset_XY_l2[1, :], KDE_offset_XY_l2[0, :], marker=plot_marker[2], c=marker_color[2])
        Score_list_score_l2 = list(KDE_score_XY_l2)

        KDE_offset_XY_l1 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level1_KDE_" + methods[m_idx] + "_offset_list.npz.npy"))
        KDE_score_XY_l1 = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level1_KDE_" + methods[m_idx] + "_score_list.npz.npy"))
        X_train_l1_list_score = list(KDE_offset_XY_l1[1, :] * level_ratio[1] / level_ratio[2])
        Y_train_l1_list_score = list(KDE_offset_XY_l1[0, :] * level_ratio[1] / level_ratio[2])
        plt.scatter(KDE_offset_XY_l1[1, :], KDE_offset_XY_l1[0, :], marker=plot_marker[1], c=marker_color[1])
        Score_list_score_l1 = list(KDE_score_XY_l1)

        x_np = np.concatenate([np.array(xi) for xi in [X_train_l1_list_score, X_train_l2_list_score, X_train_l3_list_score]]).reshape(-1, 1)
        y_np = np.concatenate([np.array(xi) for xi in [Y_train_l1_list_score, Y_train_l2_list_score, Y_train_l3_list_score]]).reshape(-1, 1)
        w_np = np.concatenate([np.array(xi) for xi in [Score_list_score_l1, Score_list_score_l2, Score_list_score_l3]])

        regr_w = linear_model.LinearRegression(fit_intercept=False)
        k_s_w = regr_w.fit(x_np, y_np, sample_weight=w_np * 10)
        slop_s_w = k_s_w.coef_[0][0]
        g_k = ground_truth_offset[ihc_idx][0] / ground_truth_offset[ihc_idx][1]
        # draw figure
        if ground_truth_offset[ihc_idx][1] >= 0:
            x = range(ground_truth_offset[ihc_idx][1] + 10)
        else:
            x = range(ground_truth_offset[ihc_idx][1] - 10, -1)
        y_s_w = slop_s_w * x
        plt.plot(x, y_s_w, 'b', linewidth=0.5)

        # get final estimation
        score_norm = norm(Score_list_score_l2, 0, 1)
        newX, newY, avgX, avgY, stdX, stdY = getEstimation(KDE_offset_XY_l2[1, :], KDE_offset_XY_l2[0, :], score_norm, Thr)
        est_x_lv0_k_a = avgX * level_ratio[2]
        est_y_lv0_k_b = est_x_lv0_k_a * slop_s_w

        est_y_lv0_k_a = avgY * level_ratio[2]
        est_x_lv0_k_b = est_y_lv0_k_a / slop_s_w

        k_est_x = round((est_x_lv0_k_a + est_x_lv0_k_b) / 2)
        k_est_y = round((est_y_lv0_k_a + est_y_lv0_k_b) / 2)

        print("OK")

        plt.plot(ground_truth_offset[ihc_idx][1], ground_truth_offset[ihc_idx][0], 'm+')
        plt.plot(s_est_x, s_est_y, 'm^')
        plt.plot(k_est_x, k_est_y, 'm*')

        # plt.plot(x, y_k, 'r-')
        plt.grid()
        plt.legend(["Score Weighted Regression", "KDE Weighted Regression", "Ground Truth", "Score", "KDE"])
        plt.xlim([min(x) - 1, max(x) + 10])
        # plt.ylim([min(y_k), max(y_k)])
        # plt.savefig(os.path.join(figure_out_dir, sub_dir[ihc_idx] + "_" + methods[m_idx] + '.png'), dpi=300)

        plt.show()
        plt.close()

        #
        # # for lv_idx in [1,2,3]:
        # #     similar_offset_XY = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_Score_" + methods[m_idx] + "_offset_list.npz.npy"))
        # #     similar_score_XY = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx], "level" + str(lv_idx) + "_Score_" + methods[m_idx] + "_score_list.npz.npy"))
        # #     X_train_list_score.append(list(similar_offset_XY[1,:]))
        # #     Y_train_list_score.append(list(similar_offset_XY[0,:]))
        # #     plt.scatter(similar_offset_XY[1, :], similar_offset_XY[0,:], marker=plot_marker[lv_idx], c=marker_color[lv_idx])
        # #     Score_list_score.append(list(similar_score_XY))
        # # x_np = np.concatenate([np.array(xi) for xi in X_train_list_score]).reshape(-1,1)
        # # y_np = np.concatenate([np.array(xi) for xi in Y_train_list_score]).reshape(-1,1)
        # # w_np = np.concatenate([np.array(xi) for xi in Score_list_score])
        # '''
        # y= ax +b . in our case, b should be 0. So, set fit_intercept to False
        # '''
        # # regr = linear_model.LinearRegression(fit_intercept=False)
        # # k_s = regr.fit(x_np, y_np)
        # # slop_s = k_s.coef_[0][0]
        # regr_w = linear_model.LinearRegression(fit_intercept=False)
        # k_s_w = regr_w.fit(x_np, y_np, sample_weight=w_np*10)
        # slop_s_w = k_s_w.coef_[0][0]
        # g_k = ground_truth_offset[ihc_idx][0]/ground_truth_offset[ihc_idx][1]
        # print("Estimate slope of %s with %s similarity score" % (sub_dir[ihc_idx], methods[m_idx]))
        # # print("Estimate with/without weight: %.4f / %.4f, Ground Truth K: %.4f" % (slop_s_w, slop_s, g_k))
        #
        # # x = np.linspace(-1+np.amin(x_np), 1+np.amax(x_np), 10)
        # if ground_truth_offset[ihc_idx][1] >= 0:
        #     x = range(ground_truth_offset[ihc_idx][1]+10)
        # else:
        #     x = range(ground_truth_offset[ihc_idx][1]-10, -1)
        # y_s_w = slop_s_w*x
        # # y_s = slop_s * x
        # plt.plot(x, y_s_w, 'r', linewidth=0.5)
        # # plt.plot(x, y_s, 'r-')
        # # plt.grid()
        # # plt.xlim([min(x), max(x)])
        # # plt.ylim([min(y_s), max(y_s)])
        # # plt.show()
        #
        # '''
        # # use KDE as weight
        # '''
        #
        # X_train_list_kde = []
        # Y_train_list_kde = []
        # Score_list_kde = []
        # # plt.figure(2,[5,4])
        # # plt.title("KDE confidence-based cross level estimation(%s)" % sub_dir[ihc_idx])
        # # for lv_idx in range(img_level):
        # for lv_idx in [1,2,3]:
        #     KDE_offset_XY = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx],"level" + str(lv_idx) + "_KDE_" + methods[m_idx] + "_offset_list.npz.npy"))
        #     KDE_Score_XY = np.load(os.path.join(data_in_dir, sub_dir[ihc_idx],"level" + str(lv_idx) + "_KDE_" + methods[m_idx] + "_score_list.npz.npy"))
        #     X_train_list_kde.append(list(KDE_offset_XY[1,:]))
        #     Y_train_list_kde.append(list(KDE_offset_XY[0,:]))
        #     # plt.scatter(KDE_offset_XY[1, :], KDE_offset_XY[0, :], marker=plot_marker[lv_idx],c=marker_color[lv_idx])
        #     Score_list_kde.append(list(KDE_Score_XY))
        # x_np = np.concatenate([np.array(xi) for xi in X_train_list_kde]).reshape(-1, 1)
        # y_np = np.concatenate([np.array(xi) for xi in Y_train_list_kde]).reshape(-1, 1)
        # w_np = np.concatenate([np.array(xi) for xi in Score_list_kde])
        # '''
        # y= ax +b . in our case, b should be 0. So, set fit_intercept to False
        # '''
        # # regr = linear_model.LinearRegression(fit_intercept=False)
        # # k_k = regr.fit(x_np, y_np)
        # # slop_k = k_k.coef_[0][0]
        # regr_k = linear_model.LinearRegression(fit_intercept=False)
        # k_k_w = regr_k.fit(x_np, y_np,sample_weight=w_np)
        # slop_k_w = k_k_w.coef_[0][0]
        # g_k = ground_truth_offset[ihc_idx][0] / ground_truth_offset[ihc_idx][1]
        # print("Estimate slope of %s with %s KDE score" % (sub_dir[ihc_idx], methods[m_idx]))
        # # print("Estimate with/without weight: %.4f / %.4f, Ground Truth K: %.4f" % (slop_k_w, slop_k, g_k))
        # # x = range(0,ground_truth_offset[ihc_idx][1])
        # if ground_truth_offset[ihc_idx][1] >= 0:
        #     x = range(ground_truth_offset[ihc_idx][1]+10)
        # else:
        #     x = range(ground_truth_offset[ihc_idx][1]-10, -1)
        # y_k_w = slop_k_w * x
        # # y_k = slop_k * x
        # plt.plot(x, y_k_w, 'b', linewidth=0.5)
        # plt.plot(ground_truth_offset[ihc_idx][1], ground_truth_offset[ihc_idx][0], 'm+')
        # # plt.plot(x, y_k, 'r-')
        # plt.grid()
        # plt.legend(["Score weighted", "KDE weighted", "Ground Truth"])
        # plt.xlim([min(x)-1,max(x)+10])
        # # plt.ylim([min(y_k), max(y_k)])
        # plt.savefig(os.path.join(figure_out_dir, sub_dir[ihc_idx] + "_" + methods[m_idx]+ '.png'),dpi=300)
        #
        # # plt.show()
        # plt.close()
        print("OK")

