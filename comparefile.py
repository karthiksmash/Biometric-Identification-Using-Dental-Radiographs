import cv2
import numpy as np
import glob

original = cv2.imread("examples/5.jpg")

# Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("dataset/Normal\*"):
    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)

for image_to_compare, title in zip(all_images_to_compare, titles):
    # 1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("Similarity: 100% (equal size and channels)")
            break
    # 2) Check for similarities between the 2 images
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance > 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    print("Title: " + title)
    percentage_similarity = len(good_points) / number_keypoints * 100
    print("Similarity: " + str(int(percentage_similarity)) + "\n")
