from getWSICases import get_co_registration_pairs, get_img_patches

RELEASE = False
VERBOSE = True

data_dir = "\\\\mfad.mfroot.org\\researchmn\\DLMP-MACHINE-LEARNING\\Mitosis_Deep Learning"
WSI_pairs = get_co_registration_pairs(data_dir)

HE_Img_name = ""
IHC_Img_name = ""
size = [800, 400, 300, 200]

def getROIs(Img_name, level=0):
    print("Get ROI")

def raw_reg(fixed_img, float_img, method="FFT"):
    offsets = []
    sim_scores = []
    return offsets, sim_scores

def offset_kde(offsets):
    kde_scores = []
    return kde_scores

def HL_fit(offsets, scores):
    refined_offsets = []
    return refined_offsets

# get ROI from WSI
locations_lv3 = getROIs(HE_Img_name, level=3)
locations_lv2 = getROIs(HE_Img_name, level=2)
locations_lv1 = getROIs(HE_Img_name, level=1)
if VERBOSE:
    locations_lv0 = getROIs(HE_Img_name, level=0)
#######################################################
# extract patches from image levels
HE_Imgs_lv3 = get_img_patches(HE_Img_name, locations[3], size[3], level=3)
HE_Imgs_lv2 = get_img_patches(HE_Img_name, locations[2], size[2], level=2)
HE_Imgs_lv1 = get_img_patches(HE_Img_name, locations[1], size[1], level=1)
if VERBOSE:
    HE_Imgs_lv0 = get_img_patches(HE_Img_name, locations[0], size[0], level=0)
IHC_Imgs_lv3 = get_img_patches(IHC_Img_name, locations[3], size[3], level=3)
IHC_Imgs_lv2 = get_img_patches(IHC_Img_name, locations[2], size[2], level=2)
IHC_Imgs_lv1 = get_img_patches(IHC_Img_name, locations[1], size[1], level=1)
if VERBOSE:
    IHC_Imgs_lv0 = get_img_patches(IHC_Img_name, locations[0], size[0], level=0)
#######################################################
# get raw registration
# FFT
Raw_FFT_lv3, scores_FFT_lv3 = raw_reg(HE_Imgs_lv3, IHC_Imgs_lv3, method="FFT")
Raw_FFT_lv2, scores_FFT_lv2 = raw_reg(HE_Imgs_lv2, IHC_Imgs_lv2, method="FFT")
Raw_FFT_lv1, scores_FFT_lv1 = raw_reg(HE_Imgs_lv1, IHC_Imgs_lv1, method="FFT")
if VERBOSE:
    Raw_FFT_lv0, scores_FFT_lv0 = raw_reg(HE_Imgs_lv0, IHC_Imgs_lv0, method="FFT")

if not RELEASE:
    # ECC
    Raw_ECC_lv3, scores_ECC_lv3 = raw_reg(HE_Imgs_lv3, IHC_Imgs_lv3, method="ECC")
    Raw_ECC_lv2, scores_ECC_lv2 = raw_reg(HE_Imgs_lv2, IHC_Imgs_lv2, method="ECC")
    Raw_ECC_lv1, scores_ECC_lv1 = raw_reg(HE_Imgs_lv1, IHC_Imgs_lv1, method="ECC")
    if VERBOSE:
        Raw_ECC_lv0, scores_ECC_lv0 = raw_reg(HE_Imgs_lv0, IHC_Imgs_lv0, method="ECC")
    # SIFT
    Raw_SIFT_lv3, scores_SIFT_lv3 = raw_reg(HE_Imgs_lv3, IHC_Imgs_lv3, method="SIFT")
    Raw_SIFT_lv2, scores_SIFT_lv2 = raw_reg(HE_Imgs_lv2, IHC_Imgs_lv2, method="SIFT")
    Raw_SIFT_lv1, scores_SIFT_lv1 = raw_reg(HE_Imgs_lv1, IHC_Imgs_lv1, method="SIFT")
    if VERBOSE:
        Raw_SIFT_lv0, scores_SIFT_lv0 = raw_reg(HE_Imgs_lv0, IHC_Imgs_lv0, method="SIFT")
    # SIFT_ENH
    Raw_SIFT_ENH_lv3, scores_SIFT_ENH_lv3 = raw_reg(HE_Imgs_lv3, IHC_Imgs_lv3, method="SIFT_ENH")
    Raw_SIFT_ENH_lv2, scores_SIFT_ENH_lv2 = raw_reg(HE_Imgs_lv2, IHC_Imgs_lv2, method="SIFT_ENH")
    Raw_SIFT_ENH_lv1, scores_SIFT_ENH_lv1 = raw_reg(HE_Imgs_lv1, IHC_Imgs_lv1, method="SIFT_ENH")
    if VERBOSE:
        Raw_SIFT_ENH_lv0, scores_SIFT_ENH_lv0 = raw_reg(HE_Imgs_lv0, IHC_Imgs_lv0, method="SIFT_ENH")
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
#######################################################
# regression
KDE_offset_FFT = HL_fit([Raw_FFT_lv3, Raw_FFT_lv2, Raw_FFT_lv1], [KDE_weights_FFT_lv3, KDE_weights_FFT_lv2, KDE_weights_FFT_lv1])
Score_offset_FFT = HL_fit([Raw_FFT_lv3, Raw_FFT_lv2, Raw_FFT_lv1], [scores_FFT_lv3, scores_FFT_lv2, scores_FFT_lv1])
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
























