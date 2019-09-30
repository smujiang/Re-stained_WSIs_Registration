'''
Extract image patches from WSIs and encode the non-blank patches into npy files.
'''


import openslide
import os
import fnmatch
data_dir = "//mfad.mfroot.org/researchmn/DLMP-MACHINE-LEARNING/Mitosis_Deep Learning"
data_out_dir = "H://HE_IHC_Stains/thumbnails"
level = 0
thumbnail_max_size = (512,512)

if __name__ == '__main__':
    WSI_names = fnmatch.filter(os.listdir(data_dir), 'Merkel-CC*')
    for wsi_name in WSI_names:
        slide_path = os.path.join(data_dir, wsi_name)
        # out_path = os.path.join(data_out_dir, wsi_name[0:-4])
        if not os.path.exists(data_out_dir):
            os.makedirs(data_out_dir)
        WSI = openslide.OpenSlide(slide_path)
        thumbnail = WSI.get_thumbnail(thumbnail_max_size)
        thumbnail.save(os.path.join(data_out_dir, wsi_name[0:-4] + ".jpg"))












