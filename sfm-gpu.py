import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('/Users/jaiwardhansinghrathore/Desktop/tensfm/img/viff.001.ppm')
img2 = cv2.imread('/Users/jaiwardhansinghrathore/Desktop/tensfm/img/viff.003.ppm')

# Set camera intrinsics
height, width, channels = img1.shape

K = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])

# Extract features
detector = cv2.AKAZE_create()
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# Match features
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.match(des1, des2)

# Compute camera poses
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# Triangulate points
proj1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
proj2 = K @ np.hstack((R, t))
print(proj1.shape) # should output (3, 4)
print(proj2.shape) # should output (3, 4)
print(pts1.shape)  # should output (N, 2)
print(pts2.shape)  # should output (N, 2)
pts1r = pts1.reshape((pts1.shape[0], 2))
pts2r = pts2.reshape((pts2.shape[0], 2))
pts4D = cv2.triangulatePoints(proj1, proj2, pts1r.T, pts2r.T)
pts3D = pts4D[:3,:] / pts4D[3,:]

# #perform bundle adjustment 
# pts3D = pts3D.reshape(-1, 1, 3)
# pts2 = pts2.reshape(-1, 1, 2)

# error, R_final, t_final, inliers = cv2.solvePnPRansac(pts3D, pts2, K, None)

# # Re-triangulate points with refined camera poses
# # proj1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
# proj2 = K @ np.hstack((R_final, t_final))
# proj2 = np.hstack((K @ proj2, np.zeros((3, 1))))
# proj2 = np.hstack((K @ proj2, np.zeros((3, 1))))
# # import pdb;pdb.set_trace()
# pts4D = cv2.triangulatePoints(proj1, proj2, pts1r.T, pts2r.T)
# pts3D = pts4D[:3,:] / pts4D[3,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3D[0,:], pts3D[1,:], pts3D[2,:], c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')    
ax.set_zlabel('Z')
plt.show()