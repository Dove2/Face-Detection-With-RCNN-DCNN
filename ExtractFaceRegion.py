import traceback
import numpy as np
import numpy.random as npr
from keras.layers import Conv2D
from keras.models import Input, Model
from keras.applications import InceptionResNetV2, VGG16
from keras.preprocessing.image import load_img, img_to_array
from utils import generate_anchors, draw_anchors, bbox_overlaps, bbox_transform, loss_cls, smoothL1, parse_label, unmap 

from keras import models, layers, Input
import numpy 

def base_nn(img_input):
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    
    return x

def rpn(base_layers):
    x = layers.Conv2D(512, (3, 3), name='rpn_conv1')(base_layers)
    x_class = layers.Conv2D(18, (1, 1), name='rpn_out_class')(x)
    x_regress = layers.Conv2D(36, (1, 1), name='rpn_out_regress')(x)

    return [x_class, x_regress, base_layers]

# input_tensor = Input(shape=(600, 800, 3))
# model = models.Model(input_tensor, base_nn(input_tensor))

# print(model.summary())

# 加载位置数据
# def load_gt_boxes(fpath): 
#     """
#     get the information about all groud true of images.
#     Args:
#         fpath: the path of the csv file.
#     Return:
#         num: the gt boxes num.
#         gt_boxes: [centre_x, centre_y, width, height] of all images.
#     """
#     f = open(fpath)
#     data = f.read()
#     f.close()

#     lines = data.split('\n')
#     header = lines[1].split(' ')
#     lines = lines[2:-1]

#     num = len(lines)
#     id_stride = 12
#     gt_data = np.zeros((len(lines), len(header) - 1))
#     for i, line in enumerate(lines):
#         x_min, y_min, width, height = [float(x) for x in list(filter(None, line[id_stride:].split(' ')))]
#         gt_data[i, 0] = x_min 
#         gt_data[i, 2] = x_min + width - 1
#         gt_data[i, 1] = y_min
#         gt_data[i, 3] = y_min + height - 1
    
#     return (num, gt_data)

def load_wider_face_gt_boxes(fpath): 
    """
    get the information about all groud true of images in wider face datasets.
    Args:
        fpath: the path of the txt file.
    Return:
        num: the gt boxes num.
        gt_boxes: [x_min, y_min, x_max, y_max]s of all images.
    """
    f = open(fpath)
    data = f.read()
    f.close()

    # header = ["x1", "y1", "w", "h", "blur", "expression", "illumination", "invalid", "occlusion", "pose"]
    lines = data.split('\n')
    gt_data = {}
    i = 0
    while True:
        gt_box_num = 1 if int(lines[i+1]) == 0 else int(lines[i+1])
        gt_pos = np.zeros((gt_box_num, 4))
        for j, bbox_list in enumerate([x.split(' ')[:4] for x in lines[i+2:i+gt_box_num+2]]):
            gt_pos[j] = [float(x) for x in bbox_list]
            gt_pos[j, 2] = gt_pos[j, 0] + gt_pos[j, 2] - 1
            gt_pos[j, 3] = gt_pos[j, 1] + gt_pos[j, 3] - 1
        gt_data[lines[i]] = gt_pos
        i += gt_box_num + 2
        if i >= len(lines) - 1: #最后一行有一个换行
            break
    
    return gt_data

gt_data = load_wider_face_gt_boxes("E:/Document/Datasets/Wider Face/wider_face_split/wider_face_train_bbx_gt.txt")

k=9 #anchor number for each point
##################  RPN Model  #######################
feature_map_tile = Input(shape=(None,None,1536))
convolution_3x3 = Conv2D(
    filters=512,
    kernel_size=(3, 3),
    padding='same',
    name="3x3"
)(feature_map_tile)

output_deltas = Conv2D(
    filters= 4 * k,
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="deltas1"
)(convolution_3x3)

output_scores = Conv2D(
    filters=1 * k,
    kernel_size=(1, 1),
    activation="sigmoid",
    kernel_initializer="uniform",
    name="scores1"
)(convolution_3x3)

model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
model.compile(optimizer='adam', loss={'scores1':loss_cls, 'deltas1':smoothL1})

##################  prepare batch  #######################
BG_FG_FRAC=2

#load an example to void graph problem
#TODO fix this.
# 由于InceptionResNetV2下采样太多倍，而人脸通常比较小，因此这里采用VGG16只下采样16倍
pretrained_model = InceptionResNetV2(include_top=False) # VGG16(include_top=False) #
img=load_img("E:/Share/ILSVRC2014_train_00010391.JPEG")
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
not_used=pretrained_model.predict(x)

def produce_batch(filepath, gt_boxes):
    # 首先加载图片
    img=load_img(filepath)
    # reshape
    img_width=np.shape(img)[1]
    img_height=np.shape(img)[0]
    img=img.resize((int(img_width),int(img_height)))
    #feed image to 预训练模型并得到feature map
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    feature_map=pretrained_model.predict(img)
    # 获得feature map的长乘宽，即所有像素点数量
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    num_feature_map=width*height
    # 用图片的长宽除以feature map的长宽，获得步长
    w_stride = img_width / width
    h_stride = img_height / height
    # print("w_stride, h_stride", w_stride, h_stride)
    # 根据步长计算anchors
    #base anchors are 9 anchors wrt a tile (0,0,w_stride-1,h_stride-1)
    base_anchors = generate_anchors(w_stride, h_stride, scales=np.asarray([1, 2, 4]))
    #slice tiles according to image size and stride.
    #each 1x1x1532 feature map is mapping to a tile.
    shift_x = np.arange(0, width) * w_stride
    shift_y = np.arange(0, height) * h_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) #这一步获得了分割点的所有横坐标及纵坐标
    # 计算出了所有偏移的(x, y, x, y)值，为什么会重复两下，因为base_anchors输出的就是(0,0,w_stride-1,h_stride-1)的模式，需要同步偏移
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    # 事实证明，对shape为(1, 9, 4)的矩阵与shape为(num_feature_map, 1, 4)的矩阵相加结果是得到shape为(num_feature_map, 9, 4)
    all_anchors = (base_anchors.reshape((1, 9, 4)) + shifts.reshape((1, num_feature_map, 4)).transpose((1, 0, 2)))
    total_anchors = num_feature_map*9
    all_anchors = all_anchors.reshape((total_anchors, 4))
    #only keep anchors inside image+borader.
    border=0
    inds_inside = np.where(
            (all_anchors[:, 0] >= -border) &
            (all_anchors[:, 1] >= -border) &
            (all_anchors[:, 2] < img_width+border ) &  # width
            (all_anchors[:, 3] < img_height+border)    # height
    )[0]
    anchors=all_anchors[inds_inside]
    if len(anchors) == 0:
        return None, None, None
    # calculate overlaps each anchors to each gt boxes,
    # a matrix with shape [len(anchors) x len(gt_boxes)]
    overlaps = bbox_overlaps(anchors, gt_boxes)
    # find the gt box with biggest overlap to each anchors,
    # and the overlap ratio. result (len(anchors),)
    argmax_overlaps = overlaps.argmax(axis=1) # overlaps中每一行的最大值的索引值，即每一个anchor与哪一个gt_box得分最高，返回的是一维张量
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # 获得overlaps中每一列的最大值，即得分
    # find the anchor with biggest overlap to each gt boxes,
    # and the overlap ratio. result (len(gt_boxes),)
    gt_argmax_overlaps = overlaps.argmax(axis=0) # overlaps中每一列的最大值的索引，即gt与哪个anchor最接近
    gt_max_overlaps = overlaps[gt_argmax_overlaps, 
                                np.arange(overlaps.shape[1])] # 获得overlaps中每一列的最大值
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0] # 获得与最大值相同的列值（纵坐标）
    #labels, 1=fg/0=bg/-1=ignore 指在图片范围内的anchors的标签
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)
    # 根据论文，设置positive标签:
    # 只对两种anchor设置positive标签
    # （1）与对每一个gt，IoU值最高的anchor
    # （2）对每一个anchor，其与所有gt的IoU最高分大于0.7的anchor
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= .7] = 1
    # 设置负面标签
    labels[max_overlaps <= .3] = 0
    # subsample positive labels if we have too many
    # num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    # if len(fg_inds) > num_fg:
    #     disable_inds = npr.choice(
    #         fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    #     labels[disable_inds] = -1
    # subsample negative labels if we have too many
    num_bg = int(len(fg_inds) * BG_FG_FRAC)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        # 因为背景太多了，随机选出多余个的设置成忽略
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False) # 从np.arange(0, bg_inds)中随机选len(bg_inds) - num_bg个
        labels[disable_inds] = -1
    # 从这里开始，计算batch，batch_inds是所有不被忽略的points
    batch_inds=inds_inside[labels!=-1]
    # 是这样的，首先batch_inds获得了在特征图内部的的anchor的索引值，又因为anchor排列是按9个9个排下来的，因此除9就是为了得到这个anchor对应的坐标
    batch_inds=(batch_inds / k).astype(np.int)
    # 获得对应于所有anchos的label
    full_labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    # batch_label_targets为n个1*1*k的
    batch_label_targets=full_labels.reshape(-1,1,1,1*k)[batch_inds]

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :]
    # 获得标签为fg的anchors
    pos_anchors=all_anchors[inds_inside[labels==1]]
    # 归一化？
    bbox_targets = bbox_transform(pos_anchors, gt_boxes[argmax_overlaps, :][labels==1])
    bbox_targets = unmap(bbox_targets, total_anchors, inds_inside[labels==1], fill=0)
    batch_bbox_targets = bbox_targets.reshape(-1,1,1,4*k)[batch_inds]
    # 在feature_map的第二个和第三个轴前后各填充一个值
    padded_fcmap=np.pad(feature_map,((0,0),(1,1),(1,1),(0,0)),mode='constant')
    # 把padded_fcmap中维度为1的轴去掉，预期得到的是3维
    padded_fcmap=np.squeeze(padded_fcmap)
    batch_tiles=[]
    for ind in batch_inds:
        x = ind % width
        y = int(ind/width)
        fc_3x3=padded_fcmap[y:y+3,x:x+3,:]
        batch_tiles.append(fc_3x3)
    return np.asarray(batch_tiles), batch_label_targets.tolist(), batch_bbox_targets.tolist()

def filter_out_gt_boxes(gt_boxes, shreshold):
    for item in gt_boxes:
        if item[2] - item[0] >= shreshold and item[3] - item[1] >= shreshold:
            return True
    return False

# exit()
##################  generate data  #######################
print("generate data")
CelebA_dataset_path='E:/Document/Downloads/CelebA/Img/img_celeba'
wider_face_dataset_path = "E:/Document/Datasets/Wider Face/WIDER_train/images"
import os

def input_generator(gt_data, batch_size=32):
    batch_tiles=[]
    batch_labels=[]
    batch_bboxes=[]
    while 1:
        # for jpg_index in range(20000):
            # fname = os.path.join(CelebA_dataset_path, "{}.jpg".format(jpg_index+1).zfill(10))
            # gt_boxes = np.expand_dims(gt_data[jpg_index, :], axis=0)
        for path, gt_boxes in gt_data.items():
            fname = os.path.join(wider_face_dataset_path, path)
            tiles, labels, bboxes = produce_batch(fname, gt_boxes)
            # print("produce batch done.", path, len(bboxes))
            if tiles is None or labels is None or bboxes is None:
                print("continue")
                continue
            for i in range(len(tiles)):
                batch_tiles.append(tiles[i])
                batch_labels.append(labels[i])
                batch_bboxes.append(bboxes[i])
                if(len(batch_tiles)==batch_size):
                    a=np.asarray(batch_tiles)
                    b=np.asarray(batch_labels)
                    c=np.asarray(batch_bboxes)
                    if not a.any() or not b.any() or not c.any(): #if a或b或c中所有的的值都是0
                        print("empty array found.") # 当gt_box比较小的时候就会这样
                        if not a.any():
                            print("It is because a.any() is False")
                        if not b.any():
                            print("It is because b.any() is False")
                        if not c.any():
                            print("It is because c.any() is False")
                        

                    yield a, [b, c]
                    batch_tiles=[]
                    batch_labels=[]
                    batch_bboxes=[]


##################   start train   #######################
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='model/RPN.hdf5', verbose=1, save_best_only=True)
history = model.fit_generator(input_generator(gt_data), steps_per_epoch=1000, epochs=800, callbacks=[checkpointer])

# 观察训练结果
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
