# PHENet
This project provides the code and results for 'Pathology-Guided Heterogeneous Expert Network with Optimal Transport Alignment for Brain Tumor Segmentation',
# Requirements
Python 3.7+, Pytorch 1.13+, Cuda 10.2+,  <br>
If anything goes wrong with the environment, please check requirements.txt for details.

# Architecture and Details
   ![image](https://github.com/JaWalkery/CCSNet-BrainTumor-Segmentation/blob/60f9756c4b12f5969aaa6777968254ab6e3e2a06/%E5%9B%BE%E7%89%874.png)
   ![image](https://github.com/JaWalkery/CCSNet-BrainTumor-Segmentation/blob/60f9756c4b12f5969aaa6777968254ab6e3e2a06/%E5%9B%BE%E7%89%873.png)
   ![image](https://github.com/JaWalkery/CCSNet-BrainTumor-Segmentation/blob/cb11c972874f90a3cfd3bba0b22bc7898582c44f/%E5%9B%BE%E7%89%87.png)

# Results
<img src="https://github.com/JaWalkery/CCSNet-BrainTumor-Segmentation/blob/59d701eb4c84c285e523b46864978cfdc1861438/%E5%9B%BE%E7%89%871.png"/>
<img src="https://github.com/JaWalkery/CCSNet-BrainTumor-Segmentation/blob/59d701eb4c84c285e523b46864978cfdc1861438/%E5%9B%BE%E7%89%872.png"/>


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

                    

