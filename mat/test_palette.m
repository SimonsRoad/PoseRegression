% test_palette.m
% Test temporary .mat files and stuff.
clear;clc;close all;

input = '../src/testchannel.mat';
load(input);

figure(1); imagesc(x);axis image;