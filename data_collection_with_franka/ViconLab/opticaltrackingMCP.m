%% load videos 
close all, clear all
addpath("fibrescope\")
addpath("webcam\")
addpath("viconTrials\vicon\")
params = load("calibrationParametersFibrescope2.mat");
webcam = VideoReader('webcamMannequinXY.mp4');
% fibrescope = VideoReader('ballLtoRthenRtoLfibrescope.avi');
fibrescope = VideoReader('trial1.mp4');

% define output variables
webOUT = VideoWriter('webOUT.mp4');
fibreOUT = VideoWriter('fibreOUT.mp4');

%% Webcam LK OF
tic
webOF = opticalFlowLK('NoiseThreshold',0.0015);
% webOF = opticalFlowLKDoG('NoiseThreshold',0.001);
% webOF = opticalFlowHS("MaxIteration",1,'Smoothness',1);
% webOF = opticalFlowFarneback('NeighborhoodSize',15);

web_h = figure;
movegui(web_h);
web_hViewPanel = uipanel(web_h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
web_hPlot = axes(web_hViewPanel);

open(webOUT)
i=1;
while hasFrame(webcam)
    webframeRGB = readFrame(webcam);
    webframeGray = im2gray(webframeRGB);
    webflow = estimateFlow(webOF,webframeGray);
    web_store(i)=webflow;
    i=i+1;
    imshow(webframeRGB)
    hold on
    plot(webflow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',web_hPlot);
    drawnow;
    writeVideo(webOUT,getframe(web_h))
    hold off
    pause(10^-6)
end
webToc = toc; % 302.02s
close(webOUT)
%% Fibrescope LK OF
tic
fibreOF = opticalFlowLK('NoiseThreshold',0.0003);
% fibreOF = opticalFlowLKDoG('NoiseThreshold',0.0001);
% fibreOF = opticalFlowHS("MaxIteration",1,'Smoothness',1);
% fibreOF = opticalFlowFarneback('NeighborhoodSize',10);

fibre_h = figure;
movegui(fibre_h);
fibre_hViewPanel = uipanel(fibre_h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
fibre_hPlot = axes(fibre_hViewPanel);
i=1;
% intrinsics extraction
% calib = readFrame(fibrescope);
% [imagePoints,boardSize] = detectCheckerboardPoints(images.Files, 'HighDistortion', true);
% worldPoints = generateCheckerboardPoints(boardSize,size(calib)); 
% params = estimateFisheyeParameters(imagePoints,worldPoints,imageSize); 

open(fibreOUT)

while hasFrame(fibrescope)
    fibreframeRGB = readFrame(fibrescope);
%     fish_cor = undistortFisheyeImage(fibreframeRGB,params.params.Intrinsics); %fish eye lens correction

%     fibrecrop = imcrop(fibreframeRGB, )
    fibreframeGray = im2gray(fibreframeRGB);
    fibreflow = estimateFlow(fibreOF,fibreframeGray);
    figre_store(i)=fibreflow;
    i=i+1;
    imshow(fibreframeGray)
    hold on
    plot(fibreflow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',fibre_hPlot);
    drawnow;
    writeVideo(fibreOUT,getframe(fibre_h))
    hold off
    pause(10^-6)
end
fibreToc = toc; % 213.87s
close(fibreOUT)
%% Max mags web and fibre

%% Mean shift optical tracking method 

%% SIFT ? 


