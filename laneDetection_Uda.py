import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test2.jpg')

print('This image is: ',type(image), 'with dimensions:', image.shape)
plt.imshow(image)
plt.show()

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)
line_image = np.copy(image)

# 定义色彩阈值
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])
color_select[color_thresholds] = [0,0,0]
# # color_select[:320,:,:] = [0,0,0]
plt.imshow(color_select)
plt.show()

#-----------------------------------------------------------------#
