import cv2

sift = cv2.xfeatures2d.SIFT_create()


def get_descriptors(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, descs = sift.detectAndCompute(gray, None)
    return descs
