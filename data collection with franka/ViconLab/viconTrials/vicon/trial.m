close all, clear all
%% load data
data = readmatrix('mannequinXY 1.csv');
data = data(4:end,1:8);

% spliting cols to variables
frame = data(:,1);
RX = data(:,3); % rad
RY = data(:,4); % rad
RZ = data(:,5); % rad
TX = data(:,6); % mm
TY = data(:,7); % mm
TZ = data(:,8); % mm 

%% plot data
figure(1)

subplot(2,1,1)
plot(frame,RX)
hold on
plot(frame,RY)
plot(frame,RZ)
legend('RX','RY','RZ')
hold off 
xlabel('Frame number')
ylabel('Magnitude (rad)')

subplot(2,1,2)
plot(frame,TX)
hold on
plot(frame,TY)
plot(frame,TZ)
legend('TX','TY','TZ')
hold off
xlabel('Frame number')
ylabel('Magnitude (mm)')
