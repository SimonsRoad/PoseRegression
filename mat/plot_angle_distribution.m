% plot_angle_distribution.m
% Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
% 

clc; clear; close all;

angles_rand = dlmread('angdist/angles_rand.txt');
angles_gaus = dlmread('angdist/angles_gaus.txt');

figure(1); set(gcf, 'Position', [100,100, 1000, 400]);
title('angle distribution (random)');
subplot(1,2,1); rose(degtorad(angles_rand),30);
subplot(1,2,2); hist(angles_rand,30);

figure(2); set(gcf, 'Position', [500,100, 1000, 400]);
title('angle distribution (Gaussian)');
subplot(1,2,1); rose(degtorad(angles_gaus),30);
subplot(1,2,2); hist(angles_gaus,30);


