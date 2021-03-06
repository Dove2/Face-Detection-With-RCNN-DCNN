from keras import models, layers, Input

pic_num = 5000
train_pic_num = 3000
val_pic_num = 500
test_pic_num = 5000

img_original_width = 178
img_original_height = 218

overall_width = 39
overall_height = 39
topbot_width = 39
topbot_height = 31
channels_num = 1

def build_F1_model():
    overall_input_tensor = Input(shape=(overall_height, overall_width, channels_num))
    x = layers.Conv2D(20, (4, 4), activation='relu', name='block1_conv1')(overall_input_tensor)
    x = layers.MaxPooling2D((2, 2), name='block1_pooling1')(x)
    x = layers.Conv2D(40, (3, 3), activation='relu', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='block1_pooling2')(x)
    x = layers.Conv2D(60, (3, 3), activation='relu', name='block1_conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='block1_pooling3')(x)
    x = layers.Conv2D(80, (2, 2), activation='relu', name='block1_conv4')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu', name='block1_dense1')(x)
    x = layers.Dense(10, name='block1_dense2')(x)
    return models.Model(overall_input_tensor, x)

def build_EN1_model():
    top_input_tensor = Input(shape=(topbot_height, topbot_width, channels_num))
    y = layers.Conv2D(20, (4, 4), activation='relu', name='block2_conv1')(top_input_tensor)
    y = layers.MaxPooling2D((2,2), name='block2_pooling1')(y)
    y = layers.Conv2D(40, (3, 3), activation='relu', name='block2_conv2')(y)
    y = layers.MaxPooling2D((2, 2), name='block2_pooling2')(y)
    y = layers.Conv2D(60, (3, 3), name='block2_conv3')(y)
    y = layers.MaxPooling2D((2, 2), name='block2_pooling3')(y)
    y = layers.Conv2D(80, (2, 2), name='block2_conv4')(y)
    y = layers.Flatten()(y)
    y = layers.Dense(120, activation='relu', name='block2_dense1')(y)
    y = layers.Dense(6, name='block2_dense2')(y)
    return models.Model(top_input_tensor, y)

def build_NM1_model():
    bot_input_tensor = Input(shape=(topbot_height, topbot_width, channels_num))
    z = layers.Conv2D(20, (4, 4), activation='relu', name='block3_conv1')(bot_input_tensor)
    z = layers.MaxPooling2D((2,2), name='block3_pooling1')(z)
    z = layers.Conv2D(40, (3, 3), activation='relu', name='block3_conv2')(z)
    z = layers.MaxPooling2D((2, 2), name='block3_pooling2')(z)
    z = layers.Conv2D(60, (3, 3), name='block3_conv3')(z)
    z = layers.MaxPooling2D((2, 2), name='block3_pooling3')(z)
    z = layers.Conv2D(80, (2, 2), name='block3_conv4')(z)
    z = layers.Flatten()(z)
    z = layers.Dense(120, activation='relu', name='block3_dense1')(z)
    z = layers.Dense(6, name='block3_dense2')(z)
    return models.Model(bot_input_tensor, z)

# 测试图片读取是否成功
import matplotlib.pyplot as plt
def show_raw_image_with_points(i, imgs_data, pos_data, normalize):
    print("显示图片为第{}张图片".format(i+1))
    plt.figure()
    plt.imshow(np.squeeze(imgs_data[i]), cmap=plt.cm.gray)
    x_list = pos_data[i][np.arange(0,9,2)]
    if normalize:
        for j, x in enumerate(x_list):
            x_list[j] = transform_pos((img_original_width, img_original_height), (overall_width, overall_height), x, 'x')
    y_list = pos_data[i][np.arange(1,10,2)]
    if normalize:
        for j, y in enumerate(y_list):
            y_list[j] = transform_pos((img_original_width, img_original_height), (overall_width, overall_height), y, 'y')
    print(x_list[0], y_list[0], x_list[1], y_list[1], x_list[2], y_list[2], x_list[3], y_list[3], x_list[4], y_list[4])
    plt.scatter(x_list[0],y_list[0],c='b')
    plt.scatter(x_list[1],y_list[1],c='r')
    plt.scatter(x_list[2],y_list[2],c='g')
    plt.scatter(x_list[3],y_list[3],c='w')
    plt.scatter(x_list[4],y_list[4],c='y')
    plt.show()


# 坐标转换（因为图片大小变化带来的必要修改）
def transform_pos(original_size, target_size, input, type):
    """
    坐标转换（因为图片大小变化带来的必要修改），输入input转换成由图片变换随之变换的坐标
    Args:
        original_size: 图片变换前的大小，是一个tuple，（widht，height）
        target_size: 图片变换后的大小，是一个tuple，（widht，height）
        input: 变换前图片中某些坐标的值, 横坐标x或纵坐标y或坐标(x, y)
    Return:
        x或y或(x, y)表示图片变换后图片中某些坐标随之变换到的坐标
    """
    ox, oy = original_size
    ox = float(ox)
    oy = float(oy)
    tx, ty = target_size
    tx = float(tx)
    ty = float(ty)
    if type == 'x':
        # print(float(input), ox, tx, float(input) / ox * tx)
        return float(input) / ox * tx
    elif type == 'y':
        return float(input) / oy * ty
    elif type == 'tuple':
        x, y = input
        x = float(x)
        y = float(y)
        return (x / ox * tx, y / oy * ty)
    else:
        print('type error.')
        return None


def normalization(ref, input):
    """
    归一化处理
    Args:
        ref: 归一化参考
    Return:
        归一化之后的结果
    """
    ref = float(ref)
    input = float(input)
    return (input - 0.5 * ref) / (0.5 * ref)

def revert_normalization(ref, input):
    """
    反归一化处理
    Args:
        ref: 归一化参考
    Return:
        反归一化之后的结果
    """
    ref = float(ref)
    input = float(input)
    return (input * 0.5 * ref) + (0.5 * ref)

# 加载位置数据
def loadPosData(fpath): 
    """
    get the information about all image produced by processSSSegmentation(imgPath, outPutPath, width, height):.
    Args:
        fpath: the path of the csv file.
    Return:
        num: the image num.
        pos_data: ['ID', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 
                   'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y'] of all images.
    """
    f = open(fpath)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[1].split(' ')
    lines = lines[2:-1]

    num = len(lines)
    
    id_stride = 11
    pos_data = np.zeros((len(lines), len(header)))
    for i, line in enumerate(lines):
        pos_data[i, :] = [float(x) for x in list(filter(None, line[id_stride:].split(' ')))]
    
    return (num, pos_data)

def face_pos_generator(imgs_data, pos_data, style, min_index, max_index, shuffle=False, batch_size=20):
    if max_index is None:
        max_index = len(imgs_data) - 1
    i = min_index
    while 1:
        if shuffle:
            rows = np.random.randint(min_index, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        if style == 'overall':
            samples = np.zeros((len(rows), overall_height, overall_width, channels_num)) 
            targets = np.zeros((len(rows), 10))
            for j, row in enumerate(rows):
                samples[j] = imgs_data[row] / 255.
                for k, value in enumerate(pos_data[row]):
                    # if k % 2 == 0: # 对x坐标进行变换及归一化
                    targets[j, k] = normalization(overall_width,
                                                    transform_pos((img_original_width, img_original_height),
                                                                (overall_width, overall_height), 
                                                                int(value), 
                                                                'x' if k%2==0 else 'y'))
        elif style == 'top':
            samples = np.zeros((len(rows), topbot_height, topbot_width, channels_num)) 
            targets = np.zeros((len(rows), 6))
            for j, row in enumerate(rows):
                samples[j] = imgs_data[rows[j]] / 255.
                for k, value in enumerate(pos_data[rows[j], :6]):
                    # 坐标并不会因为裁剪而改变
                    targets[j, k] = normalization(overall_width,
                                                    transform_pos((img_original_width, img_original_height),
                                                                (overall_width, overall_height), 
                                                                int(value), 
                                                                'x' if k%2==0 else 'y'))
        
        elif style == 'bot':
            samples = np.zeros((len(rows), topbot_height, topbot_width, channels_num)) 
            targets = np.zeros((len(rows), 6))
            for j, row in enumerate(rows):
                samples[j] = imgs_data[rows[j]] / 255.
                for k, value in enumerate(pos_data[rows[j], 4:]):
                    targets[j, k] = normalization(overall_width,
                                                    transform_pos((img_original_width, img_original_height),
                                                                (overall_width, overall_height), 
                                                                int(value), 
                                                                'x' if k%2==0 else 'y'))
        else:
            exit()
        
        yield samples, targets

def show_train_result(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    print('training results:')
    plt.figure() 


def show_predict_result_with_real_points(i, img_data, model, pre_len):
    t = img_data[i] / 255.0
    t = np.expand_dims(t, axis=0)
    prediction=model.predict(t)[0]
    print(prediction)
    for j, x in enumerate(prediction):
        prediction[j] = revert_normalization(overall_height, x) 
    x_list=prediction[np.arange(0,pre_len-1,2)]
    y_list=prediction[np.arange(1,pre_len,2)]
    x_list_real = pos_data[i][np.arange(0,9,2)]
    for j, x in enumerate(x_list_real):
        x_list_real[j] = transform_pos((img_original_width, img_original_height), (overall_width, overall_height), x, 'x')
    y_list_real = pos_data[i][np.arange(1,10,2)]
    for j, y in enumerate(y_list_real):
        y_list_real[j] = transform_pos((img_original_width, img_original_height), (overall_width, overall_height), y, 'y')

    plt.imshow(np.squeeze(t[0]), cmap='gray')
    plt.scatter(x_list,y_list,c='r')
    plt.scatter(x_list_real,y_list_real,c='b')
    plt.show()



################################# 数据预处理 #######################################
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img

base_dir = 'E:/Document/Datasets/CelebA/Img'

train_dir = os.path.join(base_dir, 'img_align_celeba')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
predict_dir = os.path.join(base_dir, 'prediction')

# 把训练图片分别加载成预定大小的numpy数组

# 读顺序为：矩阵有多pic_num个，即pic_num张图片，每张图片height行，每行width个像素，每个像素channels_num个值
imgs_data_overall = np.zeros((pic_num, overall_height, overall_width, channels_num), dtype=int)
imgs_data_top = np.zeros((pic_num, topbot_height, topbot_width, channels_num), dtype=int)
imgs_data_bot = np.zeros((pic_num, topbot_height, topbot_width, channels_num), dtype=int)

for i, filename in enumerate(os.listdir(train_dir)):
    #由于os.listdir并不是按着名字顺序排序的，所以需要将名称映射出索引志
    index = int(filename[:-4]) - 1
    # print(i, filename, index)
    if index >= pic_num:
        continue
    img1 = load_img(os.path.join(train_dir, filename), color_mode="grayscale", target_size=(overall_height, overall_width))
    # img2 = load_img(os.path.join(train_dir, filename), color_mode="grayscale", target_size=(topbot_height, topbot_width))
    imgs_data_overall[index] = img_to_array(img1)
    # imgs_data_topbot[index] = img_to_array(img2)
    img1.close()
    # img2.close()

imgs_data_top[:, :, :, :] = imgs_data_overall[:, :topbot_height, :, :]
imgs_data_bot[:, :, :, :] = imgs_data_overall[:, overall_height - topbot_height:, :, :]

pos_num, pos_data = loadPosData('E:/Document/Datasets/CelebA/Anno/list_landmarks_align_celeba.txt')

# for i in np.random.randint(0, 5000, 5):
#     show_raw_image_with_points(i, imgs_data_top, pos_data, True)

batch_size = 20

val_steps = (val_pic_num) // batch_size
test_steps = (test_pic_num) // batch_size

from keras.optimizers import RMSprop, SGD
from keras import callbacks


###################################### 训练F1模型##########################################
# f1_model = build_F1_model()
# print(f1_model.summary())

# train_gen = face_pos_generator(imgs_data_overall,
#                                pos_data,
#                                'overall',
#                                min_index=0,
#                                max_index=train_pic_num,
#                             #    shuffle=True,
#                                batch_size=batch_size)
# val_gen = face_pos_generator(imgs_data_overall,
#                              pos_data,
#                              'overall',
#                              min_index=train_pic_num+1,
#                              max_index=train_pic_num+val_pic_num,
#                              batch_size=batch_size)
# test_gen = face_pos_generator(imgs_data_overall,
#                               pos_data,
#                               'overall',
#                               min_index=train_pic_num+val_pic_num+1,
#                               max_index=None,
#                               batch_size=batch_size)


# callbacks_list = [
#     callbacks.EarlyStopping(patience=5, monitor='val_loss'),
#     callbacks.ModelCheckpoint('model/F1_Model.h5', 
#                               save_best_only=True, save_weights_only=False, mode='auto', period=1),
#     callbacks.LearningRateScheduler(lambda epoch: float(np.linspace(0.03, 0.01, 500)[epoch]))
# ]

# f1_model.compile(optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), loss='mse')

# history = f1_model.fit_generator(train_gen,
#                               steps_per_epoch=train_pic_num // batch_size,
#                               epochs=500, 
#                               callbacks=callbacks_list,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# # print("evaluate train: ", model.evaluate_generator(train_gen, 3000))
# # print("evaluate test: ", model.evaluate_generator(test_gen, 500))

# show_train_result(history)

# show_predict_result_with_real_points(1, imgs_data_overall, f1_model, 10)

# for i in np.random.randint(4500, 5000, 4):
#     show_predict_result_with_real_points(i, imgs_data_overall, f1_model, 10)


########################## 训练EN1模型 ##########################
# en1_model = build_EN1_model()

# train_gen = face_pos_generator(imgs_data_top,
#                                pos_data,
#                                'top',
#                                min_index=0,
#                                max_index=train_pic_num,
#                                batch_size=batch_size)
# val_gen = face_pos_generator(imgs_data_top,
#                              pos_data,
#                              'top',
#                              min_index=train_pic_num+1,
#                              max_index=train_pic_num+val_pic_num,
#                              batch_size=batch_size)
# test_gen = face_pos_generator(imgs_data_top,
#                               pos_data,
#                               'top',
#                               min_index=train_pic_num+val_pic_num+1,
#                               max_index=None,
#                               batch_size=batch_size)



# callbacks_list = [
#     callbacks.EarlyStopping(patience=5, monitor='val_loss'),
#     callbacks.ModelCheckpoint('model/EN1_Model.h5', 
#                               save_best_only=True, save_weights_only=False, mode='auto', period=1),
#     callbacks.LearningRateScheduler(lambda epoch: float(np.linspace(0.03, 0.01, 500)[epoch]))
# ]

# en1_model.compile(optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), loss='mse')

# history = en1_model.fit_generator(train_gen,
#                               steps_per_epoch=train_pic_num // batch_size,
#                               epochs=500, 
#                               callbacks=callbacks_list,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# show_train_result(history)

# show_predict_result_with_real_points(1, imgs_data_top, en1_model, 6)

# for i in np.random.randint(4500, 5000, 4):
#     show_predict_result_with_real_points(i, imgs_data_top, en1_model, 6)



########################## 训练NM1模型 ##########################
# nm1_model = build_NM1_model()

# train_gen = face_pos_generator(imgs_data_bot,
#                                pos_data,
#                                'bot',
#                                min_index=0,
#                                max_index=train_pic_num,
#                                batch_size=batch_size)
# val_gen = face_pos_generator(imgs_data_bot,
#                              pos_data,
#                              'bot',
#                              min_index=train_pic_num+1,
#                              max_index=train_pic_num+val_pic_num,
#                              batch_size=batch_size)
# test_gen = face_pos_generator(imgs_data_bot,
#                               pos_data,
#                               'bot',
#                               min_index=train_pic_num+val_pic_num+1,
#                               max_index=None,
#                               batch_size=batch_size)



# callbacks_list = [
#     callbacks.EarlyStopping(patience=5, monitor='val_loss'),
#     callbacks.ModelCheckpoint('model/NM1_Model.h5', 
#                               save_best_only=True, save_weights_only=False, mode='auto', period=1),
#     callbacks.LearningRateScheduler(lambda epoch: float(np.linspace(0.03, 0.01, 500)[epoch]))
# ]

# nm1_model.compile(optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), loss='mse')

# history = nm1_model.fit_generator(train_gen,
#                               steps_per_epoch=train_pic_num // batch_size,
#                               epochs=500, 
#                               callbacks=callbacks_list,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# show_train_result(history)

# show_predict_result_with_real_points(1, imgs_data_bot, nm1_model, 6)

# for i in np.random.randint(4500, 5000, 4):
#     show_predict_result_with_real_points(i, imgs_data_bot, nm1_model, 6)


######################################## 加载模型并预测最终位置 ########################################################

f1_model = models.load_model("model/F1_Model.h5")
en1_model = models.load_model("model/EN1_Model.h5")
nm1_model = models.load_model("model/NM1_Model.h5")

def predict_lv1(f1_model, en1_model, nm1_model, img_index):
    img_overall = imgs_data_overall[img_index] / 255.
    img_top = imgs_data_top[img_index] / 255.
    img_bot = imgs_data_bot[img_index] / 255.

    img_overall = np.expand_dims(img_overall, axis=0)
    img_top = np.expand_dims(img_top, axis=0)
    img_bot = np.expand_dims(img_bot, axis=0)

    f1_result = f1_model.predict(img_overall)[0]
    en1_result = en1_model.predict(img_top)[0]
    nm1_result = nm1_model.predict(img_bot)[0]
    ############# test #######################
    show_predict_result_with_real_points(img_index, imgs_data_overall, f1_model, 10)
    show_predict_result_with_real_points(img_index, imgs_data_top, en1_model, 6)
    show_predict_result_with_real_points(img_index, imgs_data_bot, nm1_model, 6)
    ############# end test ###################
    for j, x in enumerate(f1_result):
        f1_result[j] = revert_normalization(overall_height, x) 
    for j, x in enumerate(en1_result):
        en1_result[j] = revert_normalization(overall_height, x) 
    for j, x in enumerate(nm1_result):
        nm1_result[j] = revert_normalization(overall_height, x) 

    result = np.zeros((10))
    result[:4] = (f1_result[:4] + en1_result[:4]) / 2
    result[4:6] = (f1_result[4:6] + en1_result[4:6] + nm1_result[:2]) / 3
    result[6:10] = (f1_result[6:10] + nm1_result[2:6]) / 2
    print("result: ", result)
    return result

i = 4392
prediction = predict_lv1(f1_model, en1_model, nm1_model, i)
t = np.expand_dims(imgs_data_overall[i], axis=0)
x_list=prediction[np.arange(0,9,2)]
y_list=prediction[np.arange(1,10,2)]
x_list_real = pos_data[i][np.arange(0,9,2)]
for j, x in enumerate(x_list_real):
    x_list_real[j] = transform_pos((img_original_width, img_original_height), (overall_width, overall_height), x, 'x')
y_list_real = pos_data[i][np.arange(1,10,2)]
for j, y in enumerate(y_list_real):
    y_list_real[j] = transform_pos((img_original_width, img_original_height), (overall_width, overall_height), y, 'y')

plt.imshow(np.squeeze(t[0]), cmap='gray')
plt.scatter(x_list,y_list,c='r')
plt.scatter(x_list_real,y_list_real,c='b')
plt.show()
