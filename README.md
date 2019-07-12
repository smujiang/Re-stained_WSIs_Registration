# Re-stained whole slide image registration

This is a repo about whole slide image(WSI) registration for re-stained slides.
If you find this repo helpful to your project, please cite the paper below:

    -- Robust Hierarchical Density Estimation and Regression for Re-stained Whole Slide Image Co-registration

#### Description of directories  
1. tools: WSI matching tools for manually registration and validation   
2. src: implementation of method presented in the paper, you can have more details in the readme.md in this folder. You can:  
    * go though the workflow step by step;   
    * also refer to an end-to-end solution for multiple pairs of WSIs by just modifying the data path in run_all.py and run.  
    Basically, you just need to call match_WSI(HE_Img_name, IHC_Img_name, methods, save_to) in your project. The first
     two parameters are the file names of the fixed and the float WSIs; parameter "methods" can be one/all of the method(s) mentioned in
     our paper; parameter 'save_to' defines where to save your result, your results will be saved as csv files in the folder you pointed to.   
3. data: data for replicating the figures in the paper.   
    Because each WSI takes up more than 4GB, we are not able to upload the original WSIs for demonstration. Some intermediate data in our experiments are provided to replicate our results.






