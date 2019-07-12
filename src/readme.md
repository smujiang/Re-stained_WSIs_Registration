### You can go though the workflow step by step
1. get image patches(getImgPatches.py)
2. introduce image registration methods to get the raw result
   a) rawRegister_FFT.py  FFT
   b) rawRegister_SIFT.py SIFT
   c) rawRegister_SIFT_ENH.py SIFT_ENH
   d) rawRegister_ECC.py  ECC
3. use KDE filter to get KDE confidence(kde_weighting.py)
4. use hierarchical weighted linear regression to make result more robust(hierarchicalLinearRegression.py)
5. draw evaluation chart and calculate evaluation metrics(draw_eval.py and eval_err.py)
### You can also refer to an end-to-end solution for multiple pairs of WSIs
modify the data path in run_all.py and run.

