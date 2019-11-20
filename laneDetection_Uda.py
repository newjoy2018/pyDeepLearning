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
left_bottom = [100, 539]
right_bottom = [900, 539]
apex = [475, 250]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Color pixels red which are inside the region of interest
color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]

# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [0,255,0]

# Display our two output images
plt.imshow(color_select)
plt.show()
plt.imshow(line_image)
