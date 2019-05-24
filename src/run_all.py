import os
import numpy as np
from AutoRegistration import match_WSI
from getWSICases import get_co_registration_pairs

RUN_SINGLE_PAIR = False
################################################
data_dir = "Z:\\Mitosis_Deep Learning"
save_to = "H:\\HE_IHC_Stains\\log.txt"
cases_save_to = "H:\\HE_IHC_Stains\\case_names.npy"
if not os.path.exists(cases_save_to):
    WSI_pairs = get_co_registration_pairs(data_dir)
    np.save(cases_save_to, WSI_pairs)
else:
    WSI_pairs = np.load(cases_save_to)

if RUN_SINGLE_PAIR:
    case_idx = 0
    p_idx = 1
    case = WSI_pairs[case_idx]
    pair = case[p_idx]
    HE = os.path.join(data_dir, pair[0])
    IHC = os.path.join(data_dir, pair[1])
    match_WSI(HE, IHC, ["FFT","ECC","SIFT_ENH","SIFT_ENH"], save_to)
else:
    for case_idx, case in enumerate(WSI_pairs):
        for p_idx, p in enumerate(case):
            print("Processing case: %d pair: %d" % (case_idx, p_idx))
            HE = os.path.join(data_dir, p[0])
            IHC = os.path.join(data_dir, p[1])
            # if the pair of images has been matched before, skip them
            if os.path.exists(save_to):
                fp = open(save_to, 'r')
                lines = fp.readlines()
                ALREADY_DONE = False
                for l in lines:
                    if p[0] in l:
                        ALREADY_DONE = True
                        break
                if not ALREADY_DONE:
                    # match_WSI(HE, IHC, ["FFT"], save_to)
                    # match_WSI(HE, IHC, ["ECC"], save_to)
                    # match_WSI(HE, IHC, ["SIFT"], save_to)
                    # match_WSI(HE, IHC, ["SIFT_ENH"], save_to)
                    match_WSI(HE, IHC, ["FFT","ECC","SIFT","SIFT_ENH"], save_to)
                    # match_WSI(HE, IHC, ["ECC", "SIFT"], save_to)
                    # write to log
                    fp = open(save_to, 'a')
                    fp.write(HE + "\n")
                    fp.close()



