import numpy as np
import os,sys
from scipy.stats import gaussian_kde
import math
import matplotlib.pyplot as plt

methods = ["FFT", "SIFT_ENH","SIFT","ECC"]
data_in_dir = "H:\\HE_IHC_Registration_result\\"
data_out_dir = "H:\\HE_IHC_Registration_result\\figures"
data_out_dir_data = "H:\\HE_IHC_Registration_result\\out_data"
sub_dir = ['HE_Caspase','HE_Ki67','HE_PHH3']
sub_dir_IHC = ['Caspase','Ki67','PHH3']
img_level = 4
img_cnt = 100
ground_truth_offset = [[146, 98], [-713, -85], [-295, -55]] #[x,y]
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
def getEstimation(X,Y,cv,Thr):
    ncv=norm(cv,0.0,1.0)
    idx = [n-1 for n, i in enumerate(ncv) if i > Thr]
    newX=X[idx]
    newY=Y[idx]
    avgX=np.mean(newX)
    avgY=np.mean(newY)
    stdX=np.std(newX)
    stdY=np.std(newY)
    return newX,newY,avgX,avgY,stdX,stdY

def getEstimation_KDE(X,Y,cv,Thr):
    ncv=norm(cv,0.0,1.0)
    idx = [n-1 for n, i in enumerate(ncv) if i > Thr]
    newX=X[idx]
    newY=Y[idx]
    avgX=np.mean(newX)
    avgY=np.mean(newY)
    stdX=np.std(newX)
    stdY=np.std(newY)
    return newX,newY,avgX,avgY,stdX,stdY

for m_idx in range(len(methods)):
    # m_idx =1
    for ihc_idx in range(len(sub_dir_IHC)):
        # ihc_idx = 0
        for lv_idx in range(img_level):
            # lv_idx = 1
            offset_list = np.load(os.path.join(data_in_dir, methods[m_idx], sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methods[m_idx] + "_offset_list.npz.npy"))
            score_list = np.load(os.path.join(data_in_dir, methods[m_idx], sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methods[m_idx] + "_score_list.npz.npy"))
            angle_list = np.load(os.path.join(data_in_dir, methods[m_idx], sub_dir[ihc_idx], "level" + str(lv_idx) + "_" + methods[m_idx] + "_angle_list.npz.npy"))
            # Plot the KDE labeled result
            plt.figure(0,[4,3])
            plt.title(sub_dir[ihc_idx]+"(KDE)_"+methods[m_idx]+"_level" + str(lv_idx), size=12)
            a = np.asarray(offset_list[:,0]).flatten()
            b = np.array(offset_list[:,1]).flatten()
            xy = np.vstack([a, b])
            level0_z = gaussian_kde(xy)(xy)
            plt.scatter(a, b, c=level0_z, marker=".", cmap='jet')
            # newX, newY, avgX, avgY, stdX, stdY = getEstimation(a, b, level0_z, 0.95)
            newX, newY, avgX, avgY, stdX, stdY = getEstimation(a, b, level0_z, 0)
            print(sub_dir[ihc_idx]+" at level"+ str(lv_idx))
            print("avgX:" + str(avgX) + "  stdX:" + str(stdX))
            print("avgY:" + str(avgY) + "  stdY:" + str(stdY))
            # plt.plot(avgX, avgY, 'm*')
            print("Ground truth: %f.4, %f.4" % (ground_truth_offset[ihc_idx][0] / level_ratio[lv_idx], ground_truth_offset[ihc_idx][1] / level_ratio[lv_idx]))
            plt.plot(ground_truth_offset[ihc_idx][0]/level_ratio[lv_idx], ground_truth_offset[ihc_idx][1]/level_ratio[lv_idx], 'r+')
            plt.grid()
            plt.axis('equal')
            # plt.legend(["Estimation","Ground Truth","Offsets"])
            plt.legend(["Ground Truth","Offsets"])
            plt.xlabel('offset_x', size=12)
            plt.ylabel('offset_y', size=12)
            # plt.xlim(ground_truth_offset[ihc_idx][0] - 25, ground_truth_offset[ihc_idx][0] + 25)
            # plt.ylim(ground_truth_offset[ihc_idx][1] - 25, ground_truth_offset[ihc_idx][1] + 25)
            # plt.show()
            plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.01)
            plt.tight_layout()
            plt.savefig(os.path.join(data_out_dir,sub_dir[ihc_idx] + "_KDE_" + methods[m_idx] + "_level" + str(lv_idx) + '.png'),dpi=300)
            plt.close()
            # save results
            np.save(os.path.join(data_out_dir_data, sub_dir[ihc_idx], "level" + str(lv_idx) + "_KDE_" + methods[m_idx]  + "_offset_list.npz"),[a, b])
            # np.save(os.path.join(data_out_dir_data, sub_dir[ihc_idx], "level" + str(lv_idx) + "_KDE_" + methods[m_idx]  + "_offset_list.npz"),[newX, newY])
            np.save(os.path.join(data_out_dir_data, sub_dir[ihc_idx], "level" + str(lv_idx) + "_KDE_" + methods[m_idx]  + "_score_list.npz"),level0_z)


            # Plot the Score labeled result
            plt.figure(1, [4, 3])
            plt.title(sub_dir[ihc_idx]+"(Score)_"+methods[m_idx]+"_level" + str(lv_idx), size=12)
            plt.scatter(a, b, c=score_list, marker=".", cmap='jet')
            # newX, newY, avgX, avgY, stdX, stdY = getEstimation(a, b, score_list, 0.85)
            newX, newY, avgX, avgY, stdX, stdY = getEstimation(a, b, score_list, 0)
            print(sub_dir[ihc_idx] + " at level" + str(lv_idx))
            print("avgX:" + str(avgX) + "  stdX:" + str(stdX))
            print("avgY:" + str(avgY) + "  stdY:" + str(stdY))
            # plt.plot(avgX, avgY, 'm*')
            print("Ground truth: %f.4, %f.4" % (ground_truth_offset[ihc_idx][0] / level_ratio[lv_idx], ground_truth_offset[ihc_idx][1] / level_ratio[lv_idx]))
            plt.plot(ground_truth_offset[ihc_idx][0]/level_ratio[lv_idx], ground_truth_offset[ihc_idx][1]/level_ratio[lv_idx], 'r+')
            plt.grid()
            plt.axis('equal')
            # plt.legend(["Estimation", "Ground Truth","Offsets"])
            plt.legend(["Ground Truth","Offsets"])
            plt.xlabel('offset_x', size=12)
            plt.ylabel('offset_y', size=12)
            # plt.xlim(ground_truth_offset[ihc_idx][0] - 25, ground_truth_offset[ihc_idx][0] + 25)
            # plt.ylim(ground_truth_offset[ihc_idx][1] - 25, ground_truth_offset[ihc_idx][1] + 25)
            # plt.show()
            plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.01)
            plt.tight_layout()
            plt.savefig(os.path.join(data_out_dir, sub_dir[ihc_idx]+"_Score_"+methods[m_idx]+"_level" + str(lv_idx)+'.png'), dpi=300)
            plt.close()
            # save results
            np.save(os.path.join(data_out_dir_data, sub_dir[ihc_idx],"level" + str(lv_idx) + "_Score_" + methods[m_idx] + "_offset_list.npz"), [a, b])
            # np.save(os.path.join(data_out_dir_data, sub_dir[ihc_idx],"level" + str(lv_idx) + "_Score_" + methods[m_idx] + "_offset_list.npz"), [newX, newY])
            np.save(os.path.join(data_out_dir_data, sub_dir[ihc_idx],"level" + str(lv_idx) + "_Score_" + methods[m_idx] + "_score_list.npz"), score_list)
            print("OK")


