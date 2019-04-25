import os, shutil

# original_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'
baseDir = 'E:\Developer\keras\Face Detection\CelebA'
# os.mkdir(baseDir)

imgDir = os.path.join(os.path.join(baseDir, 'Img'), 'img_align_celeba')
landmarksDir = os.path.join(baseDir, 'Anno')

# # Directories for our training, validation and test splits
# train_dir = os.path.join(baseDir, 'train')
# os.mkdir(train_dir)
# validation_dir = os.path.join(baseDir, 'validation')
# os.mkdir(validation_dir)
# test_dir = os.path.join(baseDir, 'test')
# os.mkdir(test_dir)


# fnames = ['00{}.jpg'.format(i) for i in range(2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dir, fname)
#     shutil.copyfile(src, dst)

# fnames = ['00{}.jpg'.format(i) for i in range(2000, 4000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dir, fname)
#     shutil.copyfile(src, dst)

# fnames = ['00{}.jpg'.format(i) for i in range(4000, 5000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dir, fname)
#     shutil.copyfile(src, dst)

# 数据预处理
import numpy

imgWidth = 178
imgHeight = 218

# path指的是landmars文件所在的目录
def preprocessPosData(path): 
    fname = os.path.join(path, 'list_landmarks_align_celeba.txt')
    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    inputNum = lines[0]
    header = lines[1].split(' ')
    lines = lines[2:]

    # 检查数据正确性
    if int(inputNum) != len(lines): 
        print("not equal")

    # pos_data = numpy.zeros((len(lines), len(header)))
    pos_data = numpy.zeros((len(lines), 2))
    for i, line in enumerate(lines):
        values = [int(x) for x in line[10:].split("  ")]
        pos_data[i] = values[:2]
    
    return (int(inputNum), pos_data)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def generator(imgDataPath, labelDataPath, min_index, max_index, batch_size=20):
    # 位置信息
    (imgNum, posData) = preprocessPosData(labelDataPath)

    # 图片 （这儿可能有问题）
    imgData = numpy.zeros((imgNum, imgHeight, imgWidth, 3))
    i = 0
    for filename in os.listdir(imgDataPath):
        img = load_img(os.path.join(imgDataPath, filename))
        
        imgData[i] = numpy.asarray(img)
        img.close()
        i+=1


    if max_index is None:
        max_index = len(imgData) - 1
    i = min_index

    while 1:
        # 如果超出了上限，则重新开始轮回，符合generator的定义
        if i + batch_size >= max_index: 
            i = min_index
        rows = numpy.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = numpy.zeros((len(rows), imgHeight, imgWidth, 3))
        targets = numpy.zeros((len(rows), posData.shape[1]))
        
        for j, row in enumerate(rows):
            samples[j] = imgData[row]
            targets[j] = posData[row]
        yield samples, targets

# 构造网络

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (4, 4), activation='relu',
                        input_shape=(imgHeight, imgWidth, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_generator = generator(imgDir, landmarksDir, 0, 2000)
val_generator = generator(imgDir, landmarksDir, 2000, 4000)
test_generator = generator(imgDir, landmarksDir, 4000, 5000)


history = model.fit_generator(train_generator, 
                              samples_per_epoch=250,
                              epochs=30,
                              validation_data=val_generator, 
                              validation_steps=250)

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