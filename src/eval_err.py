import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, scatter

#
def load_from_csvs(csv_file):
    eval_data = np.loadtxt(csv_file, dtype=str, delimiter=',')
    return eval_data


#
def calculate_err(ground_tr, test_data, Type="dist"):
    err_x_list = []
    err_y_list = []
    err_x_list_score = []
    err_y_list_score = []
    err_dist_list = []
    err_dist_list_score = []
    for idx, gt in enumerate(ground_tr):
        if gt[0] == test_data[idx][0] and gt[1] == test_data[idx][1]:
            err_x = int(gt[2]) + int(test_data[idx][2])
            err_y = int(gt[3]) + int(test_data[idx][3])
            err_x_score = int(gt[2]) + int(test_data[idx][4])
            err_y_score = int(gt[3]) + int(test_data[idx][5])
            err_dist = sqrt(err_x ** 2 + err_y ** 2)
            err_dist_score = sqrt(err_x_score ** 2 + err_y_score ** 2)
            err_x_list.append(err_x)
            err_y_list.append(err_y)
            err_x_list_score.append(err_x_score)
            err_y_list_score.append(err_y_score)
            err_dist_list.append(err_dist)
            err_dist_list_score.append(err_dist_score)
        else:
            raise Exception("Evaluation data is not in correct order")
    if Type == "dist":
        return err_dist_list, err_dist_list_score
    else:
        return err_x_list, err_y_list, err_x_list_score, err_y_list_score


ground_truth_file = "H:\\HE_IHC_Stains\\log\\groundTruth_Jiang.csv"
FFT_eval_file = "H:\\HE_IHC_Stains\\log\\log_fft.csv"
ECC_eval_file = "H:\\HE_IHC_Stains\\log\\log_ecc.csv"
SIFT_eval_file = "H:\\HE_IHC_Stains\\log\\log_sift.csv"
SIFT_ENH_eval_file = "H:\\HE_IHC_Stains\\log\\log_sift_enh.csv"

ground_truth = load_from_csvs(ground_truth_file)
FFT_eval = load_from_csvs(FFT_eval_file)
ECC_eval = load_from_csvs(ECC_eval_file)
SIFT_eval = load_from_csvs(SIFT_eval_file)
SIFT_ENH_eval = load_from_csvs(SIFT_ENH_eval_file)

FFT_err_x_list, FFT_err_y_list, FFT_err_x_list_score, FFT_err_y_list_score = calculate_err(ground_truth, FFT_eval, Type="xy")
ECC_err_x_list, ECC_err_y_list, ECC_err_x_list_score, ECC_err_y_list_score = calculate_err(ground_truth, ECC_eval, Type="xy")
SIFT_err_x_list, SIFT_err_y_list, SIFT_err_x_list_score, SIFT_err_y_list_score = calculate_err(ground_truth, SIFT_eval, Type="xy")
SIFT_ENH_err_x_list, SIFT_ENH_err_y_list, SIFT_ENH_err_x_list_score, SIFT_ENH_err_y_list_score = calculate_err(ground_truth, SIFT_ENH_eval, Type="xy")

sample_size = 30

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 6))
axes[0, 0].scatter(FFT_err_x_list, FFT_err_y_list, marker=".", color='red')
axes[0, 0].set_title('FFT errors')
axes[0, 0].set_ylabel('KDE weighted')

axes[0, 1].scatter(ECC_err_x_list, ECC_err_y_list, marker=".", color='red')
axes[0, 1].set_title('ECC errors')

axes[0, 2].scatter(SIFT_err_x_list, SIFT_err_y_list, marker=".", color='red')
axes[0, 2].set_title('SIFT errors')

axes[0, 3].scatter(SIFT_ENH_err_x_list, SIFT_ENH_err_y_list, marker=".", color='red')
axes[0, 3].set_title('SIFT_ENH errors')

axes[1, 0].scatter(FFT_err_x_list, FFT_err_y_list_score, marker=".", color='blue')
axes[1, 0].set_ylabel('Score weighted')
axes[1, 1].scatter(ECC_err_x_list, ECC_err_y_list_score, marker=".", color='blue')
axes[1, 2].scatter(SIFT_err_x_list, SIFT_err_y_list_score, marker=".", color='blue')
axes[1, 3].scatter(SIFT_ENH_err_x_list, SIFT_ENH_err_y_list_score, marker=".", color='blue')


for ax_r in axes:
    for ax in ax_r:
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])

plt.show()


FFT_err_dist_list, FFT_err_dist_list_score = calculate_err(ground_truth, FFT_eval, Type="dist")
ECC_err_dist_list, ECC_err_dist_list_score = calculate_err(ground_truth, ECC_eval, Type="dist")
SIFT_err_dist_list, SIFT_err_dist_list_score = calculate_err(ground_truth, SIFT_eval, Type="dist")
SIFT_ENH_err_dist_list, SIFT_ENH_err_dist_list_score = calculate_err(ground_truth, SIFT_ENH_eval, Type="dist")

all_err_dist = [FFT_err_dist_list, ECC_err_dist_list, SIFT_err_dist_list, SIFT_ENH_err_dist_list]
all_err_dist_score = [FFT_err_dist_list_score, ECC_err_dist_list_score, SIFT_err_dist_list_score, SIFT_ENH_err_dist_list_score]
labels = ["FFT", "ECC", "SIFT", "SIFT_ENH"]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
bplot1 = axes[0].boxplot(all_err_dist,  vert=True,  patch_artist=True, labels=labels)
axes[0].set_title('KDE weighted')
axes[0].set_ylim([-20, 350])

bplot2 = axes[1].boxplot(all_err_dist_score,  vert=True, patch_artist=True, labels=labels)
axes[1].set_title('Score weighted')
axes[1].set_ylim([-20, 350])

colors = ['pink', 'lightgreen', 'lightblue', 'green']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xlabel('Methods')
    ax.set_ylabel('Error')


plt.show()

print(ECC_err_x_list)

