import numpy as np
import os,sys
from PIL import Image
import openslide
import matplotlib.pyplot as plt

org_data_dir = "H:\\HE_IHC_Stains"
out_dir = "H:\\HE_IHC_Registration"

sub_dir = ['HE_Caspase','HE_Ki67','HE_PHH3']
sub_dir_IHC = ['Caspase','Ki67','PHH3']
fixed_img_names = ['Merkel-CC_CR16-1790-A2_HE-Cleaved-Caspase3.svs','Merkel-CC_CR16-1790-A2_HE-Ki-67.svs','Merkel-CC_CR16-1790-A2_HE-PHH3.svs']
float_img_names = ['Merkel-CC_CR16-1790-A2_Cleaved-Caspase3.svs','Merkel-CC_CR16-1790-A2_Ki-67.svs','Merkel-CC_CR16-1790-A2_PHH3.svs']

level_patch_num = [10,6,4,3]
level_patch_size = [800,700,600,500]
# level_patch_size = [800,800,800,800]
level_0_size = 800
for idx, name in enumerate(float_img_names):
    fix_filename = os.path.join(org_data_dir, fixed_img_names[idx])
    sd_fix = openslide.OpenSlide(fix_filename)
    float_filename = os.path.join(org_data_dir,float_img_names[idx])
    sd_float = openslide.OpenSlide(float_filename)

    fix_levels = sd_fix.level_count
    fix_dim = sd_fix.dimensions
    fix_level_dim = sd_fix.level_dimensions
    fix_level_downsample = sd_fix.level_downsamples

    float_levels = sd_float.level_count
    float_dim = sd_float.dimensions
    float_level_dim = sd_float.level_dimensions
    float_level_downsample = sd_float.level_downsamples

    if fix_levels == float_levels:
        lds_equ = True
        for ld_idx, ldv in enumerate(fix_level_downsample):
            if not int(float_level_downsample[ld_idx]) == int(ldv):
                lds_equ = False
                print("Down sample rate not equal")
                sys.exit(1)
    else:
        print("Image level not equal")
        sys.exit(1)
    # get ROI coordinate for each level
    ROISize = min(fix_dim[0],fix_dim[1],float_dim[0],float_dim[1])
    if fix_dim[0] > fix_dim[1]:
        steps_x = range(0,ROISize,level_0_size)
        org_y_fix = int((fix_dim[0]-ROISize)/2)
        steps_y = range(org_y_fix, org_y_fix + ROISize, level_0_size)
    else:
        steps_y = range(0,ROISize,level_0_size)
        org_x_fix = int((fix_dim[0] - ROISize) / 2)
        steps_x = range(org_x_fix,org_x_fix+ROISize,level_0_size)
    x = len(steps_x)
    y = len(steps_y)
    level3_x = range(steps_x[0], steps_x[0] + level_0_size*40, level_0_size*4)
    level3_y = range(steps_y[0], steps_y[0] + level_0_size*40, level_0_size*4)
    level2_x = range(steps_x[13], steps_x[13] + level_0_size*40, level_0_size*4)
    level2_y = range(steps_y[13], steps_y[13] + level_0_size*40, level_0_size*4)
    level1_x = range(steps_x[23], steps_x[23] + level_0_size*20, level_0_size*2)
    level1_y = range(steps_y[23], steps_y[23] + level_0_size*20, level_0_size*2)
    level0_x = range(steps_x[28], steps_x[28] + level_0_size*10, level_0_size)
    level0_y = range(steps_y[28], steps_y[28] + level_0_size*10, level_0_size)
    level_cord = [[level0_x,level0_y], [level1_x,level1_y], [level2_x,level2_y], [level3_x,level3_y]]

    for lv in range(0,fix_levels):
        cnt = 0
        print("LEVEL %d" % lv)
        cord_XY = level_cord[lv]
        for pos_x in cord_XY[0]:
            for pos_y in cord_XY[1]:
                print("Get Patch from [%d,%d], image size: [%d,%d]" % (pos_x, pos_y, level_patch_size[lv], level_patch_size[lv]))
                Img_fix_col = sd_fix.read_region((pos_y, pos_x), lv, (level_patch_size[lv], level_patch_size[lv]))
                Img_float_col = sd_float.read_region((pos_y, pos_x), lv, (level_patch_size[lv], level_patch_size[lv]))
                float_patch_name = os.path.join(out_dir,sub_dir[idx],sub_dir_IHC[idx],"level"+str(lv),str(cnt)+".jpg")
                fix_patch_name= os.path.join(out_dir, sub_dir[idx], "HE", "level" + str(lv), str(cnt)+ ".jpg")
                Img_fix_col.convert("RGB").save(fix_patch_name)
                Img_float_col.convert("RGB").save(float_patch_name)
                cnt += 1
        # for m in range(level_patch_num[lv]):
        #     for n in range(level_patch_num[lv]):
        #         pos_y = int((init_y*float_level_downsample[lv] + m * level_patch_size[lv]))
        #         pos_x = int((init_x*float_level_downsample[lv] + n * level_patch_size[lv]))
        #         print("Get Patch from [%d,%d], image size: [%d,%d]" %(pos_x,pos_y,level_patch_size[lv],level_patch_size[lv]))
        #         Img_fix_col = sd_fix.read_region((pos_y, pos_x), lv, (level_patch_size[lv], level_patch_size[lv]))
        #         Img_float_col = sd_float.read_region((pos_y, pos_x), lv, (level_patch_size[lv], level_patch_size[lv]))
        #         fix_patch_name = os.path.join(out_dir,sub_dir[idx],sub_dir_IHC[idx],"level"+str(lv),str(m)+"_"+str(n)+".jpg")
        #         float_patch_name = os.path.join(out_dir, sub_dir[idx], "HE", "level" + str(lv), str(m) + "_" + str(n) + ".jpg")
        #         Img_fix_col.convert("RGB").save(fix_patch_name)
        #         Img_float_col.convert("RGB").save(float_patch_name)

    #
    # for lv in range(0,fix_levels):
    #     print("LEVEL %d" %lv)
    #     # fix_level_dim_l = fix_level_dim[lv]
    #     # init_x = int(fix_level_dim_l[1]/2 - level_patch_size[lv]*level_patch_num[lv]/2)
    #     # init_y = int(fix_level_dim_l[0] / 2 - level_patch_size[lv] * level_patch_num[lv] / 2)
    #     # pos_x = init_x
    #     # pos_y = init_y
    #     for m in range(level_patch_num[lv]):
    #         for n in range(level_patch_num[lv]):
    #             pos_y = int((init_y*float_level_downsample[lv] + m * level_patch_size[lv]))
    #             pos_x = int((init_x*float_level_downsample[lv] + n * level_patch_size[lv]))
    #             print("Get Patch from [%d,%d], image size: [%d,%d]" %(pos_x,pos_y,level_patch_size[lv],level_patch_size[lv]))
    #             Img_fix_col = sd_fix.read_region((pos_y, pos_x), lv, (level_patch_size[lv], level_patch_size[lv]))
    #             Img_float_col = sd_float.read_region((pos_y, pos_x), lv, (level_patch_size[lv], level_patch_size[lv]))
    #             fix_patch_name = os.path.join(out_dir,sub_dir[idx],sub_dir_IHC[idx],"level"+str(lv),str(m)+"_"+str(n)+".jpg")
    #             float_patch_name = os.path.join(out_dir, sub_dir[idx], "HE", "level" + str(lv), str(m) + "_" + str(n) + ".jpg")
    #             Img_fix_col.convert("RGB").save(fix_patch_name)
    #             Img_float_col.convert("RGB").save(float_patch_name)









