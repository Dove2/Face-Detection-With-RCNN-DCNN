import os
import traceback
import pickle
import numpy as np
from keras.applications import InceptionResNetV2, VGG16
from keras.preprocessing.image import load_img, img_to_array
from utils import load_wider_face_gt_boxes
WiderFace_dataset_path='E:/Document/Datasets/Wider Face'
img_path = os.path.join(WiderFace_dataset_path, 'WIDER_train/images')
gt_boxes_path =  os.path.join(WiderFace_dataset_path, 'wider_face_split/wider_face_train_bbx_gt.txt')

# pretrained_model = InceptionResNetV2(include_top=False)

gt_data = load_wider_face_gt_boxes(gt_boxes_path)

# for path, gt_boxes in gt_data.items():
#     fname = os.path.join(img_path, path)
#     if not os.path.exists("feature_maps"):
#         os.mkdir("feature_maps")

#     img=load_img(fname)
#     # img_width=np.shape(img)[1] * scale[1]
#     # img_height=np.shape(img)[0] * scale[0]
#     # img=img.resize((int(img_width),int(img_height)))
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     feature_map = pretrained_model.predict(img)
#     output_filename = os.path.join('feature_maps', path[:-4])
#     os.makedirs(os.path.dirname(output_filename), exist_ok=True)
#     fileObject = open(output_filename,'wb')
#     np.savez_compressed(fileObject,fc=feature_map)


###############################获取所有图片的大小信息#######################################

import csv

output_path = "w_h.csv"

with open(output_path, 'w', newline='') as f:
    csv_write = csv.writer(f)
    csv_head = ["img_path", "w","h"]
    csv_write.writerow(csv_head)  
    for path in gt_data.keys():
        img = load_img(os.path.join(img_path, path))        
        row = [path, np.shape(img)[1], np.shape(img)[0]]
        csv_write.writerow(row)