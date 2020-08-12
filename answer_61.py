import cv2
import numpy as np
import matplotlib.pyplot as plt


# Connect 4
def connect_4(img):
    # get shape
    H, W, C = img.shape

    # prepare temporary image
    tmp = np.zeros((H, W), dtype=np.int)

    # binarize
    tmp[img[..., 0] > 0] = 1

    # prepare out image
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # each pixel
    for y in range(H):
        for x in range(W):
            if tmp[y, x] < 1:
                continue



    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("renketsu.png").astype(np.float32)

# connect 4
out = connect_4(img)

# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()