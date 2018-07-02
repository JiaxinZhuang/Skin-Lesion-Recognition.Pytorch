import color_constancy as cc
import cv2 as cv
import os
import process_bar
import sys

data_dir = '../data/ISIC2018/ISIC2018_Task3_Training_Input/'
#for cnt in [2,5,8,9]:
cnt = int(sys.argv[1])
output_dir = '../data/sog6_{}'.format(cnt)
os.makedirs(output_dir, exist_ok=True)
print('-----{}-----'.format(cnt))
process_bar_ = process_bar.process_bar(10015)
for img in os.listdir(data_dir):
    if img.split('.')[-1] != 'jpg':
        continue
    image_path = os.path.join(data_dir, img)
    image_np = cv.imread(image_path)
    new_image_np = cc.shade_of_gray(image_np, power=cnt)
    output_path = os.path.join(output_dir, img)
    #print(output_path)
    cv.imwrite(output_path,new_image_np)
    process_bar_.show_process()
