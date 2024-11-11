import numpy as np
import cv2

def computeH(x1, x2):
	"""
	OUTPUT:
	H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
	equation
	"""
	N = x1.shape[0]
	A = []
    
	for i in range(N):
		x1_i, y1_i = x1[i]
		x2_i, y2_i = x2[i]
    
		A.append([-x2_i, -y2_i, -1, 0, 0, 0, x2_i * x1_i, y2_i * x1_i, x1_i])
		A.append([0, 0, 0, -x2_i, -y2_i, -1, x2_i * y1_i, y2_i * y1_i, y1_i])
    
	A = np.array(A)
    
	U, S, Vt = np.linalg.svd(A)
    
	H = Vt[-1].reshape(3, 3)
    
	return H

def computeH_norm(x1, x2):
	#Q3.7
	#Compute the centroid of the points
	centroid1 = np.mean(x1, axis=0)
	centroid2 = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	x1_shifted = x1 - centroid1
	x2_shifted = x2 - centroid2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	norm1 = np.max(np.linalg.norm(x1_shifted, axis=1))
	norm2 = np.max(np.linalg.norm(x2_shifted, axis=1))

	# Scale factor to make average distance sqrt(2)
	scale1 = np.sqrt(2) / norm1
	scale2 = np.sqrt(2) / norm2

	# Normalize the points
	x1_normalized = x1_shifted * scale1
	x2_normalized = x2_shifted * scale2

	# Similarity transform 1
	T1 = np.array([[scale1, 0, -scale1 * centroid1[0]],
				   [0, scale1, -scale1 * centroid1[1]],
				   [0, 0, 1]])

	# Similarity transform 2
	T2 = np.array([[scale2, 0, -scale2 * centroid2[0]],
				   [0, scale2, -scale2 * centroid2[1]],
				   [0, 0, 1]])

	#Compute homography
	H_normalized = computeH(x1_normalized, x2_normalized)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H_normalized @ T2
	
	return H2to1

def computeH_ransac(x1, x2):
	"""
	OUTPUTS
	bestH2to1 - homography matrix with the most inliers found during RANSAC
	inliers - a vector of length N (len(matches)) with 1 at the those matches
		that are part of the consensus set, and 0 elsewhere.
	"""
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	max_iters = 1000
	inlier_tol = 1
	N = x1.shape[0]
	bestH2to1 = None
	max_inliers = 0
	inliers = np.zeros(N)

	for _ in range(max_iters):
		#print("new iteration")
		# Randomly sample 4 point pairs
		idx = np.random.choice(N, 4, replace=False)
		x1_sample = x1[idx]
		x2_sample = x2[idx]

		# Compute the homography using the sampled points
		H2to1 = computeH_norm(x1_sample, x2_sample)

		# Apply the homography to transform points in x2
		x2_homog = np.hstack([x2, np.ones((N, 1))])
		#print("shape",np.shape(x2_homog))
		x2_transformed_homog = (H2to1 @ x2_homog.T).T
		x2_transformed = x2_transformed_homog[:, :2] / x2_transformed_homog[:, 2][:, np.newaxis]

		# Compute the error between transformed x2 and actual x1
		distances = np.linalg.norm(x1 - x2_transformed, axis=1)

		# Count inliers based on the distance threshold
		current_inliers = distances < inlier_tol
		num_inliers = np.sum(current_inliers)

		# Update the best homography if the current one has more inliers
		if num_inliers > max_inliers:
			max_inliers = num_inliers
			bestH2to1 = H2to1
			inliers = current_inliers
			#print("hello", np.shape(bestH2to1))
	return bestH2to1, inliers

def compositeH(H2to1, template, img):
	#print("hihi", np.shape(H2to1))
	assert(np.shape(H2to1) == (3,3))
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	
	#Create mask of same size as template
	template_h, template_w = template.shape[:2]
	mask = np.ones((template_h, template_w), dtype=np.uint8) * 255

	#Warp mask by appropriate homography
	mask_warped = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))

	cv2.imwrite("mask_warped.jpg", mask_warped)
	mask_warped[mask_warped > 0] = 255  # Create a binary mask

	#Warp template by appropriate homography
	warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

	#Use mask to combine the warped template and the image
	composite_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_warped))
	composite_img += warped_template

	return composite_img