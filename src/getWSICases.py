import os

data_dir = "\\\\mfad.mfroot.org\\researchmn\\DLMP-MACHINE-LEARNING\\Mitosis_Deep Learning"
pre_fix = "Merkel-CC_"

HEs = ["HE-Cleaved-Caspase3", "HE-Ki-67", "HE-PHH3"]
IHCs = ["Cleaved-Caspase3", "Ki-67", "PHH3"]


def get_case_id(wsi_name):
    sp_wsi_name = wsi_name.replace(pre_fix, "")
    wsi_name_ele = sp_wsi_name.split("_")
    case_id = wsi_name_ele[0]
    return case_id


def get_co_registration_pairs(data_dir):
    WSI_pairs = []
    case_ids = set()
    wsi_names = os.listdir(data_dir)
    for f_name in wsi_names:
        if ".svs" in f_name:
            c_id = get_case_id(f_name)
            case_ids.add(c_id)
    for c_id in case_ids:
        HE_Caspase = pre_fix + c_id + "_" + HEs[0] + ".svs"
        Caspase = pre_fix + c_id + "_" + IHCs[0] + ".svs"
        HE_Ki67 = pre_fix + c_id + "_" + HEs[1] + ".svs"
        Ki67 = pre_fix + c_id + "_" + IHCs[1] + ".svs"
        HE_PHH3 = pre_fix + c_id + "_" + HEs[2] + ".svs"
        PHH3 = pre_fix + c_id + "_" + IHCs[2] + ".svs"
        WSI_pairs.append([[HE_Caspase,Caspase],[HE_Ki67,Ki67],[HE_PHH3, PHH3]])
    return WSI_pairs

if __name__ == '__main__':
    WSI_pairs = get_co_registration_pairs(data_dir)
    for case in WSI_pairs:
        for he_ihc in case:
            for img_name in he_ihc:
                full_img_name = os.path.join(data_dir,img_name)
                if not os.path.exists(full_img_name):
                    raise Exception("File not exist")
                else:
                    print("Test")
    print(WSI_pairs)











