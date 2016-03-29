#!/bin/sh

echo 'Set GPU Fan Speed to'$1'%'

nvidia-settings -a "[gpu:0]/GPUFanControlState=1" -a "[fan:0]/GPUTargetFanSpeed=$1"
nvidia-settings -a "[gpu:1]/GPUFanControlState=1" -a "[fan:1]/GPUTargetFanSpeed=$1"
nvidia-settings -a "[gpu:2]/GPUFanControlState=1" -a "[fan:2]/GPUTargetFanSpeed=$1"
nvidia-settings -a "[gpu:3]/GPUFanControlState=1" -a "[fan:3]/GPUTargetFanSpeed=$1"
