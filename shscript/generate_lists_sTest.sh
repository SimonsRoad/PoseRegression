#!/bin/sh


# paths
path='/home/namhoon/develop/PoseRegression/data/rendout/anc_y420_x130'
path_list=$path'/lists'

# img_pos.txt and jsc_pos.txt
img_pos=$path_list'/img_pos.txt'
jsc_pos=$path_list'/jsc_pos.txt'

# create img_sTest.txt and jsc_sTest.txt
num_sTest=100
outfile_img_sTest=$path_list'/img_sTest.txt'
outfile_jsc_sTest=$path_list'/jsc_sTest.txt'
tail -$num_sTest $img_pos > $outfile_img_sTest
tail -$num_sTest $jsc_pos > $outfile_jsc_sTest



#sed -i -e 's/\/pos\//\/jsc_halfsig\//g' $outFile_jsc_pos
#sed -i -e 's/pos0000/jsc/g' $outFile_jsc_pos
#sed -i -e 's/jpg/mat/g' $outFile_jsc_pos
