# import numpy as np
# f = open("/Users/zhangdefu/Desktop/wider_face_split/wider_face_train_bbx_gt.txt")
# data = f.read()
# f.close()

# # header = ["x1", "y1", "w", "h", "blur", "expression", "illumination", "invalid", "occlusion", "pose"]
# lines = data.split('\n')
# gt_data = {}
# i = 0
# while True:
#     gt_box_num = 1 if int(lines[i+1]) == 0 else int(lines[i+1])
#     gt_pos = np.zeros((gt_box_num, 4))
#     for j, bbox_list in enumerate([x.split(' ')[:4] for x in lines[i+2:i+gt_box_num+2]]):
#         gt_pos[j] = [float(x) for x in bbox_list]
#         # gt_pos[j, 2] = gt_pos[j, 0] + gt_pos[j, 2] - 1
#         # gt_pos[j, 3] = gt_pos[j, 1] + gt_pos[j, 3] - 1
#     gt_data[lines[i]] = gt_pos
#     i += gt_box_num + 2
#     if i >= len(lines) - 1: #最后一行有一个换行
#         break
# count = 0
# shreshold = 156
# for item in gt_data.values():
#     for t in item:
#         if t[2] >= shreshold and t[3] >= shreshold:
#             count+=1

# print(count)

from keras.applications import VGG16, InceptionResNetV2

# model = VGG16(include_top=False, input_shape=(320, 320, 3))
model = InceptionResNetV2(include_top=False, input_shape=(320, 320, 3))

print(model.summary())