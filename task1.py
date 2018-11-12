import cv2
import numpy as np

# Reading the images from the location
image1 = cv2.imread("C:/Users/roopa/Pictures/proj2_cse573/data/mountain1.jpg", 1)
image2 = cv2.imread("C:/Users/roopa/Pictures/proj2_cse573/data/mountain2.jpg", 1)

task1_sift1_img_path = "C:/Users/roopa/Pictures/proj2_cse573/output/task1_sift1"+".jpg"
task1_sift2_img_path = "C:/Users/roopa/Pictures/proj2_cse573/output/task1_sift2"+".jpg"
matches_knn_img_path = "C:/Users/roopa/Pictures/proj2_cse573/output/task1_matches_knn"+".jpg"
task1_matches_img_path = "C:/Users/roopa/Pictures/proj2_cse573/output/task1_matches"+".jpg"
task1_pano_img_path = "C:/Users/roopa/Pictures/proj2_cse573/output/task1_pano"+".jpg"

sift = cv2.xfeatures2d.SIFT_create()
key_point1, descriptor1 = sift.detectAndCompute(image1, None)
key_point2, descriptor2 = sift.detectAndCompute(image2, None)

task1_sift1 = cv2.drawKeypoints(image1, key_point1, image1, flags=2)
task1_sift2 = cv2.drawKeypoints(image2, key_point2, image2, flags=2)

np.set_printoptions(suppress=True)

# Using DescriptorMatcher knnMatch() to obtain all matches between the two images
bf = cv2.BFMatcher();
matches = bf.knnMatch(descriptor1, descriptor2, k=2)


def get_good_matches():
    good_matches = []
    good_key_point1 = []
    good_key_point2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            good_key_point1.append(key_point1[m.queryIdx].pt)
            good_key_point2.append(key_point2[m.trainIdx].pt)
    return good_matches, good_key_point1, good_key_point2


matches, image1_key_pts, image2_key_pts = get_good_matches()
img_knn = cv2.drawMatchesKnn(image1, key_point1, image2, key_point2, [matches], None, flags=0)

H, mask = cv2.findHomography(np.float32(image1_key_pts), np.float32(image2_key_pts), cv2.RANSAC, 5.0)

print("Homography Matrix H:")
print(H)

# Only Inliners of the matched points
def getinliers(img1, img2, kp1, kp2, matches_mask, H):
    matches_mask = np.asarray(matches)[mask.ravel() == 1]
    only_inliners = cv2.drawMatches(img1, kp1, img2, kp2, matches_mask[:10], None, flags=0)
    return only_inliners


matches_mask = np.asarray(matches)[mask.ravel() == 1]
task1_matches = getinliers(image1, image2, key_point1, key_point2, matches_mask, H)

h_img1, w_img1 = image1.shape[:2]
h_img2, w_img2 = image2.shape[:2]

# Panorama Image
corners = np.float32([[0, 0], [0, h_img1 - 1], [w_img1 - 1, h_img1 - 1],
                     [w_img1 - 1, 0]]).reshape(-1, 1, 2)
corners = cv2.perspectiveTransform(corners, H)[0]
# Finding the bounding rectangle
x, y, width, height = cv2.boundingRect(corners)
# Translation Homography to move (x,y) to (0,0)
t_h = np.array([
        [1, 0, -x],
        [0, 1, -y],
        [0, 0, 1]
    ])
combined_matrix = t_h.dot(H)
task1_pano = cv2.warpPerspective(image1, combined_matrix, (np.abs(x)+w_img2, np.abs(y)+h_img2))
task1_pano[np.abs(y):h_img2 + np.abs(y), np.abs(x):w_img2+np.abs(x)] = image2

cv2.imwrite(task1_sift1_img_path, task1_sift1)
cv2.imwrite(task1_sift2_img_path, task1_sift2)
cv2.imwrite(matches_knn_img_path, img_knn)
cv2.imwrite(task1_matches_img_path, task1_matches)
cv2.imwrite(task1_pano_img_path, task1_pano)

