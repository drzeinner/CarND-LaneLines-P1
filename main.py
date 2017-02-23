#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

import math
from scipy import stats


# Renders a line
#
# param     img         The image to render to
# param     line        The line to render
# param     color       The color of the line
# param     thickness   The thickness of the line
#
def renderLine(img, line, color=[255, 0, 0], thickness=1):
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Renders a collection of lines
#
# param     img         The image to render to
# param     lines       The collection of lines to render
# param     color       The color of the lines
# param     thickness   The thickness of the lines
#
def renderLines(img, lines, color=[255, 0, 0], thickness=1):
    for line in lines:
        renderLine(img, line, color, thickness)


# Calculates a new line given 2 points and upper/bottom bounds
#
# param     line        line defined by slope and y-intercept
# param     top_y       y coordinate of the top of the new line
# param     bottom_y    y coordinate of of the bottom of the new line
# returns               the 2 points of the discovered line
#
def extrapolateLine(line, top_y, bottom_y):
    # first get the slope
    bottom_x = 0
    top_x = 0
    for slope, intercept in line:
        bottom_x = (bottom_y - intercept) / slope
        top_x = (top_y - intercept) / slope

    return np.array([[bottom_x, bottom_y, top_x, top_y]], dtype=line.dtype)


# Wrapper for the grayscale function
#
# param    img      The img to convert
# param    bPlot    True, to plot the image
# returns           The grayscaled image
#
def grayscaleW(img, bPlot=False):
    gray = grayscale(img)
    if (bPlot):
        plt.imshow(gray, cmap='gray')
    return gray

# Mask out all colors but yellow and white
#
# param    img             The image to mask
# param    white_range     The range of white color to mask
# param    yellow_range    The range of yellow color to mask
# param    bPlot           True, to plot the image
# returns                  The color masked image
#
def colorMask(img, white_range, yellow_range, bPlot=False):
    gray = grayscaleW(img)
    subdued_gray = (gray / 2).astype('uint8')
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    white_mask = cv2.inRange(hsv_image, white_range[0], white_range[1])
    yellow_mask = cv2.inRange(hsv_image, yellow_range[0], yellow_range[1])
    yellow_white_mask = cv2.bitwise_and(white_mask, yellow_mask)
    masked_image = cv2.bitwise_or(gray, yellow_white_mask)
    if (bPlot):
        plt.imshow(masked_image)
    return masked_image


# Wrapper for the gaussian_blur function
#
# param    img            The img to convert
# param    kernel_size    The size of kernel to apply
# param    bPlot          True, to plot the image
# returns                 The blurred image
#
def gaussianblurW(img, kernel_size, bPlot=False):
    blurred = gaussian_blur(img, kernel_size)
    if (bPlot):
        plt.imshow(blurred, cmap='gray')
    return blurred


# Wrapper for the canny function
#
# param    img               The img to convert
# param    low_threshold     Values below this will be discarded
# param    high_threshold    Values above this will be included
# param    bPlot             True, to plot the image
# returns                    The gradient image
#
def cannyW(img, low_threshold, high_threshold, bPlot=False):
    gradient = canny(img, low_threshold, high_threshold)
    if (bPlot):
        plt.imshow(gradient, cmap='gray')
    return gradient


# Wrapper for the region_of_interest function
#
# param    img         The img to convert
# param    verts    The region to mask
# param    bPlot       True, to plot the image
# returns              The masked image
#
def maskRegionW(img, verts, bPlot=False):
    masked_region = region_of_interest(img, verts)
    if (bPlot):
        plt.imshow(masked_region, cmap='gray')
    return masked_region


# Wrapper for the hough_lines function
#
# param    img               The img to convert
# param    rho               The granularity of the rho parameter
# param    theta             The granularity of the theta parameter
# param    threshold         The minimum required votes
# param    minimum_length    The minimum require line length
# param    maximum_gap       The maximum allowed gap
# param    bPlot             True, to plot the image
# returns                    The image with detected lines
#
def houghTransformW(img, rho, theta, threshold, minimum_length, maximum_gap, bPlot=False):
    lines = hough_lines(img, rho, theta, threshold, minimum_length, maximum_gap)
    if (bPlot):
        plt.imshow(lines)
    return lines


# Wrapper for the hough_lines function
#
# param    imgA              The first image to combine
# param    imgB              The second image to combine
# param    α                 Weight of the first image
# param    β                 Weight of the second image
# param    λ                 Scalar added to each image
# param    bPlot             True, to plot the image
# returns                    The combined image
#
def combineImagesW(imgA, imgB, α=0.8, β=1., λ=0., bPlot=False):
    combined_img = weighted_img(imgA, imgB, α, β, λ)
    if bPlot:
        plt.imshow(combined_img)
    return combined_img


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    global frame
    global previous_left_lines
    global previous_right_lines
    max_history = 5  # number of previous frames to average
    frame += 1       # increment the frame counter

    # find line params (slope & y-intercept
    # Split into left and right groups based on slope
    slope_threshold = 0.4    # do not include lines with slopes between [-slope_threshold, slope_threshold[
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    test_left_lines = []
    test_right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate slope, y-intercept and the length of the vector
            diff = (x2 - x1, y2 - y1)
            slope = diff[1] / diff[0]
            intercept = y1 - slope * x1
            weight = np.linalg.norm(diff)
            # if the slop is less than -slope_threshold
            # add it to the left lines
            if slope < -slope_threshold:
                left_lines.append([[slope, intercept]])
                left_weights.append(weight)
                test_left_lines.append(line)
            # if the slop is greater than slope_threshold
            # add it to the right lines
            elif slope > slope_threshold:
                right_lines.append([[slope, intercept]])
                right_weights.append(weight)
                test_right_lines.append(line)

    mean = np.mean(left_lines, axis=0)
    standard_deviation = np.std(left_lines, axis=0)
    deviations = left_lines - mean

    # Remove lines far from the standard deviation
    # print("standard_deviation", standard_deviation)
    # print("left_lines", left_lines)
    #
    # only calculate the new average if there are any left lines
    if len(left_lines) > 0:
        # average the new line params
        new_avg_left = np.average(left_lines, axis=0)
        # average the last max_history frames if available
        previous_left_lines.insert(0, new_avg_left)

    # only calculate the new average if there are any right lines
    if len(right_lines) > 0:
        # average the new line params
        new_avg_right = np.average(right_lines, axis=0)
        # average the last max_history frames if available
        previous_right_lines.insert(0, new_avg_right)

    # Now average the new line with the previous lines (if available)
    # Decrement the weight of the line based on how far in the past it is
    previous_left_lines = previous_left_lines[:min(frame, max_history)]
    avg_left = np.average(previous_left_lines, axis=0, weights = np.linspace(1., 0.5, min(frame, max_history)))

    # Now average the new line with the previous lines (if available)
    # Decrement the weight of the line based on how far in the past it is (present = 1, oldest = 0.5)
    previous_right_lines = previous_right_lines[:min(frame, max_history)]
    avg_right = np.average(previous_right_lines, axis=0, weights = np.linspace(1., 0.5, min(frame, max_history)))

    # Extrapolate the lines the render
    left_line = extrapolateLine(avg_left, mask_verts[0,1,1], mask_verts[0,0,1])
    right_line = extrapolateLine(avg_right, mask_verts[0,2,1], mask_verts[0,3,1])

    # Convert back to integer array
    left_line = left_line.astype(lines.dtype)
    right_line = right_line.astype(lines.dtype)

    # Render the extrapolated lines
    renderLine(img, left_line, [0, 255, 255], 10)
    renderLine(img, right_line, [255, 255, 0], 10)
    # renderLines(img, test_left_lines, [255, 0, 0], 2)
    # renderLines(img, test_right_lines, [255, 0, 255], 2)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # === Define the parameters for the pipeline =================

    global img_width
    global img_height
    img_width = image.shape[1]
    img_height = image.shape[0]

    # === Color Masking ==========================================

    white_range = np.array([[0, 0, 230], [255, 255, 255]])
    yellow_range = np.array([[20, 100, 100], [35, 255, 255]])

    # === Gaussian Blur ==========================================

    kernel_size = 5

    # === Canny Edge =============================================

    low_threshold = 50
    high_threshold = 150

    # === Region Masking =========================================

    global vanishing_point
    vanishing_point = [img_width * 0.5, img_height * 0.6]
    vanishing_point_width = .05
    vert_bottom_left = [img_width * 0.01, img_height * 0.98]
    vert_bottom_right = [img_width * 0.99, img_height * 0.98]
    vert_top_left = [vanishing_point[0] - img_width * vanishing_point_width, vanishing_point[1]]
    vert_top_right = [vanishing_point[0] + img_width * vanishing_point_width, vanishing_point[1]]
    global mask_verts
    mask_verts = np.array([[vert_bottom_left, vert_top_left, vert_top_right, vert_bottom_right]], dtype=np.int32)

    # === Hough Transform ========================================

    rho = 1
    theta = np.pi / 180
    hough_threshold = 10
    minimum_line_length = 20
    maximum_line_gap = 10

    # === Lane Detection Pipeline ================================

    # Create yellow & white color masks
    color_masked = colorMask(image, white_range, yellow_range, bPlot=False)
    # img_out = cv2.cvtColor(color_masked, cv2.COLOR_HSV2RGB)
    # cv2.imwrite('screenShots/color_masked.jpg', color_masked)

    # Apply a Gaussian Blur to reduce noise in Canny Edge Detection
    blurred = gaussianblurW(color_masked, kernel_size)
    # cv2.imwrite('screenShots/blurred.jpg', blurred)

    # Apply Canny Edge Edge on the blurred image
    gradient = cannyW(blurred, low_threshold, high_threshold)
    # cv2.imwrite('screenShots/canny.jpg', gradient)

    # Mask out the region of interest
    masked_region = maskRegionW(gradient, mask_verts)
    # cv2.imwrite('screenShots/masked_region.jpg', masked_region)

    # Run the Hough Transform on the masked image to detect lines in the image
    lines = houghTransformW(masked_region, rho, theta, hough_threshold, minimum_line_length, maximum_line_gap)
    # cv2.imwrite('screenShots/houghTransformW.jpg', lines)


    # Render the lines on top of the original image
    result = combineImagesW(lines, image, β=0.8)
    # cv2.imwrite('screenShots/result.jpg', result)


    # Render the masked region boundary
    # cv2.polylines(result, mask_verts, 1, (0,255,0), thickness=3)

    # save the resulting image in the test_images directory
    img_out = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('screenShots/result.jpg', img_out)
    cv2.imwrite('testYellowVideo/result_' + str(frame) + '.jpg', img_out)

    return result

frame = 0
previous_left_lines = []
previous_right_lines = []
img_width = 100
img_height = 100
mask_verts = []

white_output = 'yellow.mp4'
clip1 = VideoFileClip("solidYellowLeft.mp4")
# clip1 = VideoFileClip("solidYellowLeft.mp4")
# clip1 = VideoFileClip("challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)