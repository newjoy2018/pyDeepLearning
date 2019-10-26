#----------------------------------------------------------------------------
#-----这里用带数据增强的特征提取(features extraction)-------------------------
#----------------------------------------------------------------------------

from keras.applications import VGG16      #从已有的VGG网络上截取不带Dense层的部分，即卷积基

conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150,150,3))
#----------------------------------------------------------------------------
from keras import layers
from keras import models

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
#----------------------------------------------------------------------------
