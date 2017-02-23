[image1]: ./screenShots/color_masked.jpg "Color Masked"
[image2]: ./screenShots/blurred.jpg "Blurred"
[image3]: ./screenShots/canny.jpg "Canny Edge"
[image4]: ./screenShots/masked_region.jpg "Region of Interest"
[image5]: ./screenShots/houghTransformW.jpg "Hough Transform"
[image6]: ./screenShots/result.jpg "Result"

**Finding Lane Lines on the Road**

---

## Reflection

### Pipeline Description

1. The first step of the pipeline is to apply a color mask to the image to separate the colors we want to focus on. In this case we only care about the yellow and white lane lines. I ended up creating the yellow range pretty wide so that it can account for the faded lines on the challenge video.

![alt text][image1]

2. Next I blur the image using a Gaussian Blur. This is used to reduce potential noise when applying Canny Edge Detection.

![alt text][image2]

3. Now apply I Canny Edge Detection to find the gradients in the image.

![alt text][image3]

4. Before running the Hough Transform I mask out a region-of-interest to eliminate areas of the image I do not care about.

![alt text][image4]

5. Now I apply the Hough Transform to determine where the strongest lines are in the image. I had to experiment quite a bit with the Hough Transform parameters before I got an acceptable result.
I ended up requiring a 20 pixel minimum line length in order to avoid small errant lines. I also needed to keep the maximum line gap small so that the segmented lane lines could be detected.

![alt text][image5]

6. Once I got a set of the detected lines in the image I needed to split the lines into left and right groups. I did this by calculating the slope of each line. By the nature of perspective the left lane lines will tend to slope upwards towards the vanishing point, and the right lines down.
However, since pixel positions in images increase as you go down the image I reverse that logic. At this step I also calculate the y-intercept and length for later use.
I discard lines with slopes between a specified threshold. This is to remove horizontal lines. Now with the remaining lines I calculate average slope and y-intercept. I then average these lines with the averages from the previous frames. This is done in order to remove sudden changes in the lines (seen as line jitter).
I apply a weight to historic line averages where the newer lines have higher weight. This has effect of putting more importance on the recent history. Now that I have average left and right lines calculated, I extrapolate the lines to the top and bottom of the region of interest.
Finally those lines are rendered on top of the original image.

![alt text][image6]

### Shortcomings

1. One shortcoming of the pipeline is that it doesn't handle curves very well. Since the pipeline removes horizontal lines a sharp turn would not be able to detect the lane lines.

2. Another issue if a car was very close to the viewport. In all of the test cases no cars were very close. That allowed for the pipeline to get a very clear picture of the road.
However, if a car was close it would obstruct the region-of-interest and many errant lines would be detected.


### Future Improvements

1. Dynamically adjust the vanishing point and masked regions based on the trends of the previous frames. For example if the road is beginning to curve to the left the vanishing point should shift to the left.
The shape of the masked region could also adjust based on whether or not the car is going down/up hill. This effectively changes  the horizon line and the vanishing point.

2. Dynamically adjust color masking based on the historic trends of the road. If the road has transitioned into faded lines, the color range should increase so the lines can be detected. However, this will in result allow more errant lines to be detected so the parameters of the other stages of the pipeline would also need to adjust.
On the other hand if algorithm detects that we are in a fresh painted region we could decrease the color range to get more focused and accurate lines.

3. Use a higher degree polynomial to render detected lane lines. A 1 degree polynomial performs poorly on a curves. It would be ideal to at least calculate a 2nd degree polynomial.
