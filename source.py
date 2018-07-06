import numpy as np
import imutils
import cv2


file_name="12.jpg"

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

"""
The algorithm works best when a person is on a background that does not merge with his hair. Also, the hair should not merge with the complexion.
"""

def get_head_mask(img):
    """
    Get the mask of the head
    Cuting  BG
    :param img: source image
    :return:   Returns the mask with the cut out BG
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))    # Find faces
    if len(faces) != 0:
        x, y, w, h = faces[0]
        (x, y, w, h) = (x - 40, y - 100, w + 80, h + 200)
        rect1 = (x, y, w, h)
        cv2.grabCut(img, mask, rect1, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)     #Crop BG around the head
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # Take the mask from BG

    return mask2

def is_bold(pnt, hair_mask):
    """
    Check band or not
    :param pnt: The upper point of the head
    :param hair_mask: Mask with hair
    :return: True if Bald, else False
    """
    roi = hair_mask[pnt[1]:pnt[1] + 40, pnt[0] - 40:pnt[0] + 40]    # Select the rectangle under the top dot
    cnt = cv2.countNonZero(roi) # Count the number of non-zero points in this rectangle
    # If the number of points is less than 25%, then we think that the head is bald
    if cnt < 800:
        print("Bald human on phoro")
        return True
    else:
        print("Not Bold")
        return False


img1 = cv2.imread(file_name)     # Load image
img1 = imutils.resize(img1, height=500)     # We result in 500px in height
mask = get_head_mask(img1)      # We get the mask of the head (without BG)

# Find the contours, take the largest one and memorize its upper point as the top of the head
cnts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cnt=cnts[0]
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])


# We remove the face by the color of the skin
lower = np.array([0, 0, 100], dtype="uint8")  # Lower limit of skin color
upper = np.array([255, 255, 255], dtype="uint8")  # Upper skin color limit
converted = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)   # We translate into HSV color format
skinMask = cv2.inRange(converted, lower, upper)     # Write a mask from places where the color is between the outside
mask[skinMask == 255] = 0   # We remove the face mask from the mask of the head

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.dilate(mask, kernel1, iterations=1)
i1 = cv2.bitwise_and(img1, img1, mask=mask)


# If the head is bald, then we deduce that the bald head shows the coordinates of the top point of the head
if is_bold(topmost,mask):
    cv2.rectangle(img1,topmost,topmost,(0,0,255),5)
    print(topmost)

# Otherwise we write that we are not bald and display the coordinates of the largest contour
else:
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img1,[cnts[0]],-1,(0,0,255),2)
    for c in cnts[0]:
        print(c)

# Display the image in a loop
while True:
    cv2.imshow("image1", img1)
    # Exit to Esc
    if cv2.waitKey(5) == 27:
        break

