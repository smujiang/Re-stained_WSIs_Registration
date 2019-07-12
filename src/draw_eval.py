import matplotlib.pyplot as plt
import os
import numpy as np
from getWSICases import get_co_registration_pairs
import pickle

def draw_cross_level_regression(ground_truth_offset,slop_kde,slop_score,offset_lvs_dict, save_name, method, he_ihcs):
    plt.figure(1, [5, 4])
    plt.title("%s_based Cross Level Estimation(%s)" % (method, he_ihcs))
    plot_marker = ['.', 'o', '>', '+']
    marker_color = ['r', 'g', 'b', 'y']
    if ground_truth_offset[0] >= 0:
        x = range(ground_truth_offset[0] + 10)
    else:
        x = range(ground_truth_offset[0] - 10, -1)
    y_k_w = slop_kde * x
    y_s_w = slop_score * x
    for idx,lv in enumerate(["lv3", "lv2", "lv1"]):
        offset_XY = np.array(offset_lvs_dict.get(lv))
        plt.scatter(offset_XY[:, 0], offset_XY[:, 1], marker=plot_marker[idx], c=marker_color[idx])
    plt.plot(x, y_s_w, 'r', linewidth=0.5)
    plt.plot(x, y_k_w, 'b', linewidth=0.5)
    plt.plot(ground_truth_offset[0], ground_truth_offset[1], 'm+')
    # plt.plot(x, y_k, 'r-')
    plt.grid()
    plt.legend(["Score weighted", "KDE weighted", "Ground Truth"])
    # plt.xlim([min(x) - 1, max(x) + 10])
    # plt.ylim([min(y_k), max(y_k)])
    if not os.path.exists(os.path.split(save_name)[0]):
        os.makedirs(os.path.split(save_name)[0], exist_ok=True)
    plt.savefig(save_name, dpi=300)
    plt.show()
    plt.close()
    print("OK")

def get_ground_truth(ground_truth_csv, WSI_name):
    fp = open(ground_truth_csv, 'r')
    lines = fp.readlines()
    ground_truth = []
    for l in lines:
        if WSI_name in l:
            ele = l.split(",")
            ground_truth = [-int(ele[2]), -int(ele[3])]
    fp.close()
    return ground_truth


if __name__ == '__main__':
    methods = ["FFT", "ECC", "SIFT", "SIFT_ENH"]
    # IDX = 2
    HE_IHC = ['HE_Caspase', 'HE_Ki67', 'HE_PHH3']
    # WSI_IDX = 0
    data_dir = "Z:\\Mitosis_Deep Learning"
    result_root_dir = "H:\\HE_IHC\\HE_IHC_Stains"
    cases_save_to = "H:\\HE_IHC\\HE_IHC_Stains\\case_names.npy"
    ground_truth_csv = "H:\\HE_IHC\\HE_IHC_Stains\\log\\groundTruth_Jiang.csv"
    if not os.path.exists(cases_save_to):
        WSI_pairs = get_co_registration_pairs(data_dir)
        np.save(cases_save_to, WSI_pairs)
    else:
        WSI_pairs = np.load(cases_save_to)
    for case_idx, case in enumerate(WSI_pairs):
        # case = WSI_pairs[1]
        # case_idx = 1
        for p_idx, p in enumerate(case):
            print("Processing case: %d pair: %d" % (case_idx, p_idx))
            HE_Img_name = os.path.join(data_dir, p[0])
            HE_n = os.path.split(HE_Img_name)[1]
            ground_truth_offset = get_ground_truth(ground_truth_csv, HE_n)
            print(HE_n)
            for IDX, method in enumerate(methods):
                for WSI_IDX, he_ihc in enumerate(HE_IHC):
                    result_npy_file = os.path.join(result_root_dir, HE_n, methods[IDX]+"_results.pickle")
                    if os.path.exists(result_npy_file):
                        # result_np = np.load(result_npy_file, allow_pickle=True)
                        pickle_in = open(result_npy_file, "rb")
                        result_dict = pickle.load(pickle_in)
                        slop_kde = result_dict.get("kde_slp")
                        slop_score = result_dict.get("score_slp")
                        offset_lvs_dict = result_dict
                        save_name = os.path.join(result_root_dir, HE_n, methods[IDX]+"_cross_levels.png")
                        draw_cross_level_regression(ground_truth_offset,slop_kde,slop_score,offset_lvs_dict, save_name, methods[IDX], HE_IHC[WSI_IDX])











