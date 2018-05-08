import cv2
import numpy as np

# Load Input Images
filename1 = '1.jpg'
filename2 = '2.jpg'
img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)

# SIFT does blurring anyway while detecting keypoints, so it's not necessary to blur the input images
	
# Display Input Images
#cv2.imshow('1st Image',img1)
#cv2.imshow('2nd Image',img2)
#if cv2.waitKey(0) & 0xff == 27:
#   cv2.destroyAllWindows()

sift = cv2.xfeatures2d.SIFT_create()

# Detect keypoints & descriptors for input images
k1, d1 = sift.detectAndCompute(img1, None)
k2, d2 = sift.detectAndCompute(img2, None)

# Match the keypoints in the 2 points by Brute Force
bf = cv2.BFMatcher()
matches = bf.knnMatch(d1,d2, k=2)  #L2 distance

# Ratio between first and second shortest distance
ratio = 0.8 
approved_matches = []
for m1,m2 in matches:
	if m1.distance < ratio * m2.distance:
		approved_matches.append(m1)

if len(approved_matches) > 4:
	img1_pts = []
	img2_pts = []

	for match in approved_matches:
		img1_pts.append(k1[match.queryIdx].pt)
		img2_pts.append(k2[match.trainIdx].pt)

    # findHomography takes in Floating Point matrices(4x2) only. So, reshape !
	img1_pts = np.float32(img1_pts).reshape(-1,1,2)
	img2_pts = np.float32(img2_pts).reshape(-1,1,2)
		
	# Get homography matrix
	H, s = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 4.0)
	
else:
	print ('Could not match enough keypoints to Sitch the images')
	exit()

# Get Dimensions
w1,h1 = img2.shape[:2]
w2,h2 = img1.shape[:2]

img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

# Do perspective tranform
img2_dims = cv2.perspectiveTransform(img2_dims_temp, H)

# Create a big matrix to store the result
result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

# Make it black
[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
	

transform_dist = [-x_min,-y_min]
transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0,0,1]]) 

# Warp images to get the resulting image
result_img = cv2.warpPerspective(img1, transform_array.dot(H), (x_max-x_min, y_max-y_min))
# Stich the images
result_img[transform_dist[1]:w1+transform_dist[1], transform_dist[0]:h1+transform_dist[0]] = img2

# Display the output
#cv2.imshow ('Stitched Image !', result_image)
#cv2.waitKey()

# Save the output
result_filename = 'Panorama_'+filename1[:-4]+filename2
cv2.imwrite(result_filename, result_img)
