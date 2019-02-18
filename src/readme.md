workflow
1. get image patches(getImgPatches.py)
2. introduce image registration methods to get the raw result
   a) rawRegister_FFT.py  FFT
   b) rawRegister_SIFT.py SIFT
   c) rawRegister_SIFT_ENH.py SIFT_ENH
   d) rawRegister_ECC.py  ECC
3. use KDE filter to get KDE confidence(kde_weighting.py)
4. use hierarchical weighted linear regression to make result more robust(linear_fitting.py)


