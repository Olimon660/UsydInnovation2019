# UsydInnovation2019

## Prediction Usage
Sample usage:
```shell
python predict.py ./input/SampleSubmission.csv ./model/final.ptm ./submission.csv
```
**Note:**
 - This script will not properly run unless all required python packages are installed. The easiest is to install the latest version of `Anaconda` and `PyTorch`.
 - Currently test images are saved after resize and gaussian blur, and these saved processed images are loaded for prediction. Therefore, running this script on new test images will not directly work. I have included the processed images under `Test` folder. The code for resizing and gaussian blur is also included in `process.py` under `input`

## Folder Structure
- `script` contains the scripts that are used to tune various models. Please note these scripts are not polished and the serve as a quick-and-dirty role for this competition.
- `logs` contains all the log files from the stdout of the training scripts. With these logs we also have a good version controls of the models. Only sample logs are included.
- `model` contains the models trained by the training scripts. Models are saved at each epoch for finer tuning. Only the final model is included for this submission.
- `input` contains all the data used in this competition. In this submission I only included the test images and the preprocessing script.
- `artemis` includes sample pbs and singularity build files for using Artemis.
- `submission` includes all historical submission csv files to kaggle. The final two selected submissions are from `SampleSubmission_pooled_5model.csv` and `final_resnet_5data_6.csv`

## Cloud Platform and HPC
Google Cloud Platform (GCP) virtual machines are used for this competition. A VM instance with 12 CPUs, 48G RAM and 1-4 T4 GPUs (dynamically adjusting) is set up for most of the training work.
The number of GPUs attached to the VM is dynamically edited so that no credit is wasted.

In addition to GCP, and thanks to Nathaniel, Artemis is also heavily used for the training work via a singularity container. Sample config files are included under `artemis`. The V100 GPUs are really powerful!

## Ensemble
One of the final submission is based on a weighted voting from 5 models. The calculation is done in `./submission/SampleSubmission_pooled.xlsx` for simplicity due to time constraints. The voting is comprised of `final_cv_resnet_5e5_reg_5data_hflip_vflip_hvflip_tta`*2 + `final_cv_resnet_5e5_reg_5data_mse_hflip_vflip_hvflip_tta` + `final_cv_resnet_5e5_reg_5data_WD7e4_10` + `final_resnet_5data_6` + `resnet5e5_5data_lr2_norm_9.csv`
