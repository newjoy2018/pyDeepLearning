import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 定义色彩阈值
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

def process_image(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    color_select = np.copy(image)
    line_image = np.copy(image)
    
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                        (image[:,:,1] < rgb_threshold[1]) | \
                        (image[:,:,2] < rgb_threshold[2])
    color_select[color_thresholds] = [0,0,0]
    
    left_bottom = [int(xsize*0.1), ysize]
    right_bottom = [int(xsize*0.94), ysize]
    apex = [int(xsize*0.5), int(ysize*0.46)]
    
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))
    # Find where image is both colored right and in the region
    line_image[~color_thresholds & region_thresholds] = [0,255,0]
    return line_image

#--------------------------------------------------------------------#

from moviepy.editor import VideoFileClip

v1_out = 'masked_colored_solidWhiteRight.mp4'
clip1 = VideoFileClip('solidWhiteRight.mp4')
v1_clip = clip1.fl_image(process_image)
%time v1_clip.write_videofile(v1_out, audio=False)

#--------------------------------------------------------------------#
