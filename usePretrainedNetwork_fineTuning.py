#------------------------------------------------------------------------------------------
#------fine-tuning即模型微调，主要是调整已有网络的最后几层，具体步骤如下》------------------
#------1. 在训练好的卷积基网络上添加自定义网络(Flatten和Dense)
#------2. 冻结卷积基网络
#------3. 训练所添加的自定义网络
#------4. 解冻基网络中的一些层
#------5. 联合训练解冻的基网络和添加的自定义网络
#------------------------------------------------------------------------------------------
from keras.applications import VGG16      #从已有的VGG网络上截取不带Dense层的部分，即卷积基

conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150,150,3))
#----------------------------------------------------------------------------
conv_base.trainable = False

from keras import layers
from keras import models

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#model.summary()

import os
base_dir = 'D:/data/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                               target_size=(150, 150),
                                                               batch_size=20,
                                                               class_mode='binary')
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=2e-5),
             metrics=['acc'])

history = model.fit_generator(train_generator,
                             steps_per_epoch=100,
                             epochs=30,
                             validation_data=validation_generator,
                             validation_steps=50)
#-------------------------------------------------------------------------------------------
#-----以下部分将block5_conv1层及以后的所有层layer.trainable设为True-------------------------
#-----这意味着联合训练卷积基网络中上最后几层和新增加的分类器层-------------------------------
#-------------------------------------------------------------------------------------------
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

#-----下面epochs为100，注意训练时间比较长----------------------------------------------------
history = model.fit_generator(train_generator,
                             steps_per_epoch=100,
                             epochs=100,
                             validation_data=validation_generator,
                             validation_steps=50)
