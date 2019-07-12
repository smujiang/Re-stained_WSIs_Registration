This is a histological image registration tool. With the dependent libraries (listed in environment.yml), you can    
1. run in your python environment;
2. or compile your binary executable file, so that you can run this tool independently.
* Note: It will take around 20s to show the main window of the UI, if you compile this tool into binary executable file.
#### Guideline of this tool
```bash
1. Load template(fixed) image, should be an H&E whole slide image(.svs or other format supported by openslide)
2. Load testing(float)image, should be an IHC whole slide image(.svs or other format supported by openslide)
3. Click “AutoReg” button to get the raw registration result
4. Adjust the spin box to get the your best alignment, during this step, you can:   
    * click and move your mouse on the image, a customized green mouse cursor will help you to locate and compare local details;
    * shift to another field of view for annotation by modify the coordinate shown in edit box to the up left of the fixed image;
5. Click "Save2CSV" button, your annotation will be append to the CSV file you specified in the edit box.
```

![alt text](./HistImgtool.png)