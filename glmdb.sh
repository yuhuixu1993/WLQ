#!/usr/bin/en sh  
DATA=/home/qualcomm/xuyuhui/beifen/xu 
rm -rf $DATA/testnoise1_lmdb  
build/tools/convert_imageset -shuffle -resize_height 32 -resize_width 32 $DATA/ $DATA/a.txt $DATA/3bitquicknoise1_lmdb 


