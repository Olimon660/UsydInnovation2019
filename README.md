# UsydInnovation2019

## Highlights
- Based on the recently published [WSL ResNext101](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/) model, I followed an optimised and highly efficient tuning process, which leverages the power of both GCP and Artemis. 
  - Models are saved at each epoch, so the best model can be selected without overfitting.
- I have collected as much external data as possible, as the amount of data is really the key to boost deep learning algorithms' performance.
- Many other models are also explored in the competition, including EfficientNet, DenseNet, InceptionV3 etc. 
- The preprocessing with gaussian blur is also pivotal for enhancing the performance.
- Preprocessed images are stored and then loaded for training. Compared to preprocessing images everytime in training, this dramatically saves the training time needed.
  - The shorter dimension of the image is scaled to 350
  - Gaussian Blur
- One novel approach is to treat this as an Ordinal Regression (OR) problem. Ideas are brought from [this](https://ieeexplore.ieee.org/document/7780901) study, with a better implementation of the architecture. However, this architecture makes the training a lot slower and the improvement is almost negligible, therefore not much tuning effort is spent on it.
  - the code is in `script/train_orcnn.py`
- The final best model is an ensemble of 5 ResNext101 models, treating the problem as **regression** with **SmoothL1Loss**. The best single model has the following parameters:
  - Batch size = 2^6
  - Num of epoches = 9
  - Learning rate = 5e-5, decreasing with a factor of 0.2 at every 2 steps.
  - Adam optimizer with weight decay = 5e-4
  - Data augmentation and TTA are also used

## Prediction Usage
Sample usage:
```shell
python predict_norm.py ./input/SampleSubmission.csv ./model/final.ptm ./submission.csv
```
- `predict_EN.py` is used for generating submission files from EfficientNet models.
- `predict_norm.py` is used for generating submission files with ImageNet color normalisation
- `predict.py` is used for generating submission files with models without color normalisation

**Note:**
 - This script will not properly run unless all required python packages are installed. The easiest is to install the latest version of `Anaconda` and `PyTorch`.
 - Currently test images are saved after resize and gaussian blur, and these saved processed images are loaded for prediction. Therefore, running this script on new test images will not directly work. I have included the processed images under `Test` folder. The code for resizing and gaussian blur is also included in `process.py` under `input`

## Folder Structure
- `script` contains the scripts that are used to tune various models. Please note these scripts are not polished and they serve as a quick-and-dirty role for this competition.
  - sample training commands `python train_densenet_reg.py densenet_5data > ../logs/densenet_5data.log`
- `logs` contains all the log files from the stdout of the training scripts. With these logs we also have a good version controls of the models.
- `model` contains the models trained by the training scripts. Models are saved at each epoch for finer tuning. Only one of the models is included for this submission due to space limitation.
- `input` contains all the data used in this competition. In this submission I only included the processed test images and the preprocessing script.
- `artemis` includes sample pbs and singularity build files for using Artemis.
- `submissions` includes all historical submission csv files to kaggle. The final two selected submissions are from `SampleSubmission_pooled_5model.csv` and `final_resnet_5data_6.csv`

## Cloud Platform and HPC
Google Cloud Platform (GCP) virtual machines are used for this competition. A VM instance with 12 CPUs, 48G RAM and 1-4 T4 GPUs (dynamically adjusting) is set up for most of the training work.
The number of GPUs attached to the VM is dynamically edited so that no credit is wasted. Also I added a shutdown command at the end of the training script, so the VM is automatically shut down after the training is finished.

In addition to GCP, and thanks to Nathaniel, Artemis is also heavily used for the training work via a singularity container. Sample config files are included under `artemis`. The V100 GPUs are really powerful!

## Ensemble
One of the final submission is based on a weighted voting from 5 models. The calculation is done in `./submission/SampleSubmission_pooled.xlsx` for simplicity due to time constraints. The voting is comprised of `final_cv_resnet_5e5_reg_5data_hflip_vflip_hvflip_tta`*2 + `final_cv_resnet_5e5_reg_5data_mse_hflip_vflip_hvflip_tta` + `final_cv_resnet_5e5_reg_5data_WD7e4_10` + `final_resnet_5data_6` + `resnet5e5_5data_lr2_norm_9.csv`

## Datasets

No one can deny the importance of the amount of the data needed by any deep learning algorithm. Therefore I explored as many as possible datasets that are publicly available. The additional data include:
- [APTOS](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [diabetic-retinopathy-detection (both training and testing data)](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- [IDRID](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
- [messidor](http://www.adcis.net/en/third-party/messidor/)
