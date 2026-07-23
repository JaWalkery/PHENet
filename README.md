# PHENet
This project provides the code and results for 'Pathology-Guided Heterogeneous Expert Network with Optimal Transport Alignment for Brain Tumor Segmentation',
# Requirements
Python 3.10, Pytorch 1.13+, Cuda 10.2+,  <br>
If anything goes wrong with the environment, please check requirements for details.

# Architecture and Details
   ![image](https://github.com/JaWalkery/PHENet/blob/main/PHENet/tu1_clean_render-1.png)
   ![image](https://github.com/JaWalkery/PHENet/blob/main/PHENet/tu2_clean_render-1.png)
   ![image](https://github.com/JaWalkery/PHENet/blob/main/PHENet/tu3_clean_render-1.png)


# Results
<img src="(https://github.com/JaWalkery/PHENet/blob/main/PHENet/%E5%9B%BE%E7%89%87.png
)"/>


# Data Preparation
    + downloading BraTS 2020 dataset
    which can be found from [Here](https://www.med.upenn.edu/cbica/brats2020/data.html).
Note that the depth maps of the raw data above are foreground is white.
# Training & Testing
modify the `train_root` `train_root` `save_path` path in `config.py` according to your own data path.

    
modify the `test_path` path in `config.py` according to your own data path.



# Evaluate tools
- You can select one of toolboxes to get the metrics

Note that we resize the testing data to the size of 224 * 224 for quicky evaluate. <br>

                    

