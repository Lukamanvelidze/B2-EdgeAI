# B2-EdgeAI
programming project of group B2 in EdgeAI

step 1. 
use frame extractor in validate/S02 and validate/S05 file to transform video to png files.

step 2.
use convert_to_yolo.py in ../validate  on your edge devices to transform our data into the Yolo format, this will create the label folder, with corrseponding framenumber.txt in yolo format with bounding boxes and class_id.

step 3.
use split_train_validate.py to divide the data into training and validating parts, this will divide the data in each corresponding c00x to train and validate, with 80/20 percent ratio.        
