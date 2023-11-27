function [OFout, frameGray, flow, h] = myOF(video,method)
videoRead = VideoReader(video);

switch method
    case 'LK'
        OFout = opticalFlowLK('NoiseThreshold',0.008);
    case 'LKDoG'
        OFout = opticalFlowLKDoG('NoiseThreshold',0.008);
    case 'Farneback'
        OFout = opticalFlowFarneback('Neighorhoodsize',20);
    case 'HS'
        OFout = opticalFlowHS("MaxIteration",1,'Smoothness',1);
    otherwise
        disp('Error in inputs, try again')
end

h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

while hasFrame(videoRead)
    frameRGB = readFrame(videoRead);
    frameGray = im2gray(frameRGB);
    flow = estimateFlow(OFout,frameGray);
    imshow(frameGray) 
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',hPlot);
    hold off
    pause(10^-3)
end