% script_PDPR_main.m
% Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
% A main script for pedestrian detection and pose estimation
% 

close all; clear; clc;

%% settings 
addpath('~/Downloads/convolutional-pose-machines-release/testing/');


%% load test data
selLoc = 1;
YX = [...
    138 167;
    160 260;
    170 570;
    262 544;
    130 460;
    235 325;
    169 92;
    91 354;
    230 438;
    105 245];
y = YX(selLoc,1);
x = YX(selLoc,2);
quality = 'LQ';
testdata = load_dataset(x,y,quality);


%% Procedure1: detection (0: no detection, 1: sliding window, 2: gt-box)
% As an output, it needs a box or rectangle which will be used as an input
% to a pose estimation algorithm such as CPM
% NO, DPM_INRIA, DPM_VOC, GT
detectionmethod = 'DPM_INRIA'; 
testdata = run_detection(testdata, detectionmethod);


%% evaluate detection performance
if strcmp(detectionmethod,'DPM_VOC') || strcmp(detectionmethod,'CPM_INRIA')
    eval_detection(testdata);
end


%% Procedure2: pose estimation (CPM, IEF)
posemethod = 'CPM';
run_poseestimation(testdata, posemethod);

