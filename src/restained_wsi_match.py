import argparse
import logging
import os
import sys
from wsi_registration import TissueDetector, MatcherParameters, WSI_Matcher


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--fixed_wsi",
                        required=True,
                        dest='fixed_wsi',
                        help="fixed/template WSI file name")

    parser.add_argument("-m", "--float_wsi",
                        required=True,
                        dest='float_wsi',
                        help="float/moving WSI file name")

    parser.add_argument("-d", "--det_type",
                        default="LAB_Threshold",
                        dest='det_type',
                        help="Could be GNB or LAB_Threshold")

    parser.add_argument("-v", "--threshold",
                        default=80,
                        dest='threshold',
                        help="if detector is LAB_Threshold, threshold should be in [0,100];"
                             "if detector is GNB, threshold should be in [0,1]")

    parser.add_argument("-r", "--rescale_rate",
                        default=100,
                        dest='rescale_rate',
                        help="rescale to get the thumbnail")

    parser.add_argument("-lpn", "--layer_patch_num",
                        default=[6, 6, 6],
                        dest='layer_patch_num',
                        help="patch numbers per image level")

    parser.add_argument("-lpm", "--layer_patch_max_num",
                        default=[20, 50, 50],
                        dest='layer_patch_max_num',
                        help="maximum try at each image level")

    parser.add_argument("-lps", "--layer_patch_size",
                        default=[2000, 800, 500],
                        dest='layer_patch_size',
                        help="patch size at each image level for registration")

    args = parser.parse_args()

    fixed_wsi = args.fixed_wsi  # file name of your fixed (template) whole slide image
    float_wsi = args.float_wsi  # file name of your float (moving) whole slide image
    layer_patch_num = args.layer_patch_num
    layer_patch_max_try = args.layer_patch_max_try
    layer_patch_size = args.layer_patch_size
    rescale_rate = args.rescale_rate

    # define the tissue detector, so the patches can be sampled
    tissue_detector = TissueDetector(args.det_type, args.threshold)
    matcher_parameters = MatcherParameters(layer_patch_num, layer_patch_max_try, layer_patch_size, rescale_rate)
    matcher = WSI_Matcher(tissue_detector, matcher_parameters)
    offset = matcher.match(fixed_wsi, float_wsi)
    logging.info("Shifting offset: %d %d" % offset)
    print("Shifting offset: %d %d" % offset)


if __name__ == "__main__":
    print("Example:")
    print("python wsi_match.py -t /fixed_wsi_file_name -m /float_wsi_file_name")
    sys.exit(main())


