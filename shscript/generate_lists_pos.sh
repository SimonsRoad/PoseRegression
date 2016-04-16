#!/bin/sh

# paths
path='/home/namhoon/develop/PoseRegression/data/rendout/anc_y138_x167'
path_img_pos=$path'/pos'

path_list=$path'/lists'
mkdir $path_list

# pos.txt
outFile_img_pos=$path_list'/img_pos.txt'
find $path_img_pos -name "*.jpg" ! -type d > $outFile_img_pos

# jsc_pos.txt
outFile_jsc_pos=$path_list'/jsc_pos.txt'
cp $outFile_img_pos $outFile_jsc_pos
sed -i -e 's/\/pos\//\/jsc\//g' $outFile_jsc_pos
sed -i -e 's/pos0000/jsc/g' $outFile_jsc_pos
sed -i -e 's/jpg/mat/g' $outFile_jsc_pos

