# UsydInnovation2019

## Prediction Usage
Sample usage:
```shell
python predict.py ./input/SampleSubmission.csv ./model/final.ptm ./submission.csv
```
**Note:**
 - This will not properly run unless all required python packages are installed.
 - Currently test images are saved after resize and gaussian blur, and these saved processed images are loaded for prediction. Therefore, running this script on new test images will not directly work. I have included the processed images under `Test` folder
 
