import os
from AutoRegistration import match_WSI
from getWSICases import get_co_registration_pairs

################################################
data_dir = "\\\\mfad.mfroot.org\\researchmn\\DLMP-MACHINE-LEARNING\\Mitosis_Deep Learning"
save_to = "H:\\HE_IHC_Stains\\log.txt"
WSI_pairs = get_co_registration_pairs(data_dir)
for pairs in WSI_pairs:
    for p in pairs:
        HE = os.path.join(data_dir, p[0])
        IHC = os.path.join(data_dir, p[1])
        match_WSI(HE, IHC, save_to)




