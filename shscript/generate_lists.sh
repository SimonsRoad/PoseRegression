#!/bin/sh

# paths
path_aug='/home/namhoon/develop/PoseRegression/data/rendout/tmp_y144_x256_aug'
path_img_pos=$path_aug'/pos'
path_img_neg=$path_aug'/neg'
path_jsdc_neg=$path_aug'/jsdc_neg'

path_list=$path_aug'/lists'
mkdir $path_list

# pos.txt
outFile_img_pos=$path_list'/img_pos.txt'
find $path_img_pos -name "*.jpg" ! -type d > $outFile_img_pos

# jsdc_pos.txt
outFile_jsdc_pos=$path_list'/jsdc_pos.txt'
cp $outFile_img_pos $outFile_jsdc_pos
sed -i -e 's/pos/jsdc/g' $outFile_jsdc_pos
sed -i -e 's/im/jsdc/g' $outFile_jsdc_pos
sed -i -e 's/jpg/mat/g' $outFile_jsdc_pos

# neg.txt
outFile_img_neg=$path_list'/img_neg.txt'
find $path_img_neg -name "*.jpg" ! -type d > $outFile_img_neg

# jsdc_neg.txt 
# This is okay for now, because all jsdc_neg are empty zeros,
# which means, you don't need to worry about data-label pairs
outFile_jsdc_neg=$path_list'/jsdc_neg.txt'
find $path_jsdc_neg -name "*.mat" ! -type d > $outFile_jsdc_neg

# create as one! {img, jsdc}
outFile_img=$path_list'/img.txt'
outFile_jsdc=$path_list'/jsdc.txt'
cat $outFile_img_neg $outFile_img_pos > $outFile_img
cat $outFile_jsdc_neg $outFile_jsdc_pos > $outFile_jsdc

# delete intermediate files
rm $outFile_img_pos
rm $outFile_img_neg
rm $outFile_jsdc_pos
rm $outFile_jsdc_neg

