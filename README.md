# singular-points-ORB_SURF_SIFT
Checking the operation of singular point methods of type ORB_SIFT_SURF
1. Initializing a video camera or video file.
2. Next, using the difference of 3 frames, movements are detected and the coordinates with the image are sent to the singular point method using a queue.
3. Next, we select the method of determining singular points of interest (SURF, SIFT, ORB) and the template (image).
4. Based on the target indications received, we cut out a fragment from the large image and then compare the points.
