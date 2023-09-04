close all, clear all
%% load files
% parameters for calibration
calParams = load("calibrationParametersFibrescope.mat","params");

% fibrescope image
fibreVid = VideoReader('Basler_acA2440-20gm__23898138__20221019_163606385.mp4');
fibreInp = read(fibreVid,50);

figure(1)
imshow(fibreInp)

%% apply calibration parameters

fibreCalibrated = undistortFisheyeImage(fibreInp,calParams.params.Intrinsics,'OutputView','same','ScaleFactor',0.1);

figure(2)
imshow(fibreCalibrated)