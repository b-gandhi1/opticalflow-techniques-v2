# Libraries
Use `requirements.txt` file to install libraries required. 
Numpy also needs to be installed. 

## Virtual env set-up steps
1. Create new virtual env using: `python -m venv yourvenvname`
2. Activate environment: `source yourvenvname/bin/activate`
3. Install libraies using: `pip install requirements.txt`
4. To deactivate venv: `deactivate`

# Instructions to run `opticalflow.py`
1. Activate virtual env. 
2. Run the file by `python opticalflow.py`. 
3. It requires user input to enter video path, and selection of 'webcam' or 'fibrescope'. Follow prompts. 
4. See results! 

# Instructions to run `trial_img_process_calib.py`
1. Activate venv
2. Run the file by `python trial_img_process_calib.py`
3. It requires user input to enter video path, if it requires calibration (Y/N), and selection of 'webcam' or 'fibrescope'. Follow prompts. 
   * Note: the calibration is only required for fibrescope. If N selected, it is assumed that the calibration files already exist. The path for these may need updating in the code.
# Folder structure
* Debugging with vscode file: `launch.json `
* Get data from: `opticalflow.py`
* Machine Learning implement and results: `ml_pickle_results.py` 
* Calibrating paramters tuning: `trial_img_process_calib.py`

# Current issues (self-notes): 
* fisheye undistortion does not work properly. 
  * the filtered frame is not showing anything, it appears to be blank. 
* Gunnar farneback uses up a lot of the ram, need to rethink data processing. It works with a video that is few seconds long and low resolution. 