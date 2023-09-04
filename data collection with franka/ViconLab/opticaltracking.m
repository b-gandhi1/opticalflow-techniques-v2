%% load videos 
close all, clear all
webcam = VideoReader('ballLtoRthenRtoLwebcam.mp4');
% fibrescope = VideoReader('ballLtoRthenRtoLfibrescope.avi');
fibrescope = VideoReader('fibrescopeV2.mp4');
control = VideoReader('control.mp4');

% define output variables
controlOUT = VideoWriter('controlOUT.mp4','MPEG-4');
webOUT = VideoWriter('webOUT.mp4');
fibreOUT = VideoWriter('fibreOUT.mp4');

%% optical flow control - to test if code works
tic
controlOF = opticalFlowLK('NoiseThreshold',0.008);
% controlOF = opticalFlowLKDoG('NoiseThreshold',0.008);
% controlOF = opticalFlowHS("MaxIteration",1,'Smoothness',1);
% controlOF = opticalFlowFarneback('NeighborhoodSize',20);

control_h = figure;
movegui(control_h);
control_hViewPanel = uipanel(control_h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
control_hPlot = axes(control_hViewPanel);

open(controlOUT)
i=1;
while hasFrame(control)
    controlframeRGB = readFrame(control);
    controlframeGray = im2gray(controlframeRGB);
    controlflow = estimateFlow(controlOF,controlframeGray);

    store_control(i) = controlflow; % vector format, my initial method
    i=i+1;

    imshow(controlframeGray) 
    hold on
    plot(controlflow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',control_hPlot);
    drawnow;
    writeVideo(controlOUT,getframe(control_h))
    hold off
    pause(10^-3)
end
maxMagControl = max(max(controlflow.Magnitude));
controlToc = toc; % 106.65s
close(controlOUT)
%% Optical flow LK webcam
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
    
    store_web(i) = webflow; % vector format, my initial method
    i=i+1; 

    imshow(webframeRGB)
    hold on
    plot(webflow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',web_hPlot);
    drawnow;
    writeVideo(webOUT,getframe(web_h))
    hold off
    pause(10^-3)
end
webToc = toc; % 302.02s
close(webOUT)
%% Optical traciking LK fibrescope
tic
fibreOF = opticalFlowLK('NoiseThreshold',0.0003);
% fibreOF = opticalFlowLKDoG('NoiseThreshold',0.0001);
% fibreOF = opticalFlowHS("MaxIteration",1,'Smoothness',1);
% fibreOF = opticalFlowFarneback('NeighborhoodSize',10);

fibre_h = figure;
movegui(fibre_h);
fibre_hViewPanel = uipanel(fibre_h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
fibre_hPlot = axes(fibre_hViewPanel);

% intrinsics extraction
% calib = readFrame(fibrescope);
% [imagePoints,boardSize] = detectCheckerboardPoints(images.Files, 'HighDistortion', true);
% worldPoints = generateCheckerboardPoints(boardSize,size(calib)); 
% params = estimateFisheyeParameters(imagePoints,worldPoints,imageSize); 

open(fibreOUT)
i=1;
while hasFrame(fibrescope)
    fibreframeRGB = readFrame(fibrescope);
%     fish_cor = undistortFisheyeImage(fibreframeRGB,1); %fish eye lens correction

%     fibrecrop = imcrop(fibreframeRGB, )
    fibreframeGray = im2gray(fibreframeRGB);
    fibreflow = estimateFlow(fibreOF,fibreframeGray);

    store_fibre(i) = fibreflow; % vector format, my initial method
    i=i+1;

    imshow(fibreframeGray)
    hold on
    plot(fibreflow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',fibre_hPlot);
    drawnow;
    writeVideo(fibreOUT,getframe(fibre_h))
    hold off
    pause(10^-3)
end
fibreToc = toc; % 213.87s
close(fibreOUT)
%% plot means for mag, Vx and Vy on all three: control, webcam, fibrescope

% Control mean 
for i=1:length(store_control)
    meanControlMag(i) = mean(mean([store_control(i).Magnitude]));
    meanControlVx(i) = mean(mean([store_control(i).Vx]));
    meanControlVy(i) = mean(mean([store_control(i).Vy]));
end
controlmean = figure;
plot(meanControlMag)
hold on
plot(meanControlVx)
plot(meanControlVy)
legend('Magnitude','Vx','Vy')
title('Control Experiment')
hold off

% Webcam mean
for i=1:length(store_web)
    meanWebMag(i) = mean(mean([store_web(i).Magnitude]));
    meanWebVx(i) = mean(mean([store_web(i).Vx]));
    meanWebVy(i) = mean(mean([store_web(i).Vy]));
end
webcammean = figure;
plot(meanWebMag)
hold on
plot(meanWebVx)
plot(meanWebVy)
legend('Magnitude','Vx','Vy')
hold off
title('Webcam Experiment')

% Fibrescope mean 
for i=1:length(store_control)
    meanFibreMag(i) = mean(mean([store_fibre(i).Magnitude]));
    meanFibreVx(i) = mean(mean([store_fibre(i).Vx]));
    meanFibreVy(i) = mean(mean([store_fibre(i).Vy]));
end
fibrescopemean = figure;
plot(meanFibreMag)
hold on
plot(meanFibreVx)
plot(meanFibreVy)
legend('Magnitude','Vx','Vy')
hold off
title('Fibrescope Experiment')