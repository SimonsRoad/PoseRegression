#!/bin/sh

# paths
path_aug='/home/namhoon/develop/towncenter/data/frames_y144_x256_sel'
path_img_pos=$path_aug'/pos'

path_list=$path_aug'/lists'
mkdir $path_list

# pos.txt
outFile_img_pos=$path_list'/img.txt'
find $path_img_pos -name "*.jpg" ! -type d > $outFile_img_pos

# jsdc_pos.txt
outFile_jsdc_pos=$path_list'/jsdc.txt'
cp $outFile_img_pos $outFile_jsdc_pos
sed -i -e 's/pos/jsdc/g' $outFile_jsdc_pos
sed -i -e 's/jpg/mat/g' $outFile_jsdc_pos

