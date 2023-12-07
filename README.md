[![built with Codeium](https://codeium.com/badges/main)](https://codeium.com)

# MCP data analysis
This repo consists of code to analyse the data collected from the Motion Capture Pillow (MCP) (see *other repo* for how this was done). 

## Libraries
Use `requirements.txt` file to install libraries required. 
Numpy also needs to be installed. 

### Virtual env set-up steps
1. Create new virtual env using: `python -m venv yourvenvname`
2. Activate environment: `source yourvenvname/bin/activate`
3. Install libraies using: `pip install requirements.txt`
4. To deactivate venv: `deactivate`

## Instructions to run `opticalflow.py`
1. Activate virtual env. 
2. Run the file by `python opticalflow.py <s> <path>`. 
3. s = `f` for fibrescope or `w` for webcam. 
4. path = path to video that is to be processed. 
5. See results! 

## Instructions to run `trial_img_process_calib.py`
1. Activate venv
2. Run the file by `python trial_img_process_calib.py`
3. It requires user input to enter video path, if it requires calibration (Y/N), and selection of 'webcam' or 'fibrescope'. Follow prompts. 
   * Note: the calibration is only required for fibrescope. If N selected, it is assumed that the calibration files already exist. The path for these may need updating in the code.

## Instrucitons to run `ml_pickle_results.py`
* Command format: `python ml_pickle_results.py <whichmodel>`
* In whichmodel, pass one of the following options: 
  1. bd_bin
  2. bd_gray
  3. lk_bin
  4. lk_gray


## Folder structure
* Debugging with vscode file: `launch.json `
* Get data from: `opticalflow.py`
* Machine Learning implement and results: `ml_pickle_results.py` 
* Calibrating paramters tuning: `trial_img_process_calib.py`

## Current issues (self-notes): 
* Gunnar farneback uses up a lot of the ram, need to rethink data processing. 
  * It works with a video that is few seconds long and low resolution. 
  * Or scrap using this altogether. 
* ML: 
  * SVM will no longer be used since it is deterministic. 
  * Naive Bayes GaussianNB to be used.
  * Coding still needs to happen... 
* Corner detection: 
  * Works fine on grayscale for webcam, NOT on binary webcam. 
    * This works now. Changed ref_frame from 0 to 7. 
  * Done for fibrescope videos: gray, bright, and binary.
* Blob detection on fibrescope keeps failing. Needs some parameters adjusting...  