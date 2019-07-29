# UsydInnovation2019

## Prediction Usage
Sample usage:
```shell
python predict.py ./input/SampleSubmission.csv ./model/final.ptm ./submission.csv
```
**Note:**
 - This script will not properly run unless all required python packages are installed. The easiest is to install latest version of `Anaconda` and `PyTorch`.
 - Currently test images are saved after resize and gaussian blur, and these saved processed images are loaded for prediction. Therefore, running this script on new test images will not directly work. I have included the processed images under `Test` folder. The code for resizing and gaussian blur is also included in `process.py` under `input`

## Folder Structure
- `scripts` contains the scripts that are used to tune various models. Please note these scripts are not polished and the serve as a quick-and-dirty role for this competition.
- `logs` contains all the log files from the stdout of the training scripts. With these logs we also have a good version controls of the models. Only sample logs are included.
- `model` contains the models trained by the training scripts. Models are saved at each epoch for finer tuning. Only the final model is included for this submission.
- `input` contains all the data used in this competition. In this submission I only included the test images and the preprocessing script.
- `artemis` include sample pbs and singularity build files for using Artemis.

## Cloud Platform and HPC
Google Cloud Platform (GCP) virtual machines are used for this competition. A VM instance with 12 CPUs, 48G RAM and 1-4 T4 GPUs (dynamically adjusting) is set up for most of the training work.

In addition to GCP, and thanks to Nathaniel, Artemis is also heavily used for some of the training work. Sample config files are included under `artemis`. The V100 GPUs are really powerful!
