import cv2 as cv

thresh_blue_min = 90
thresh_blue_max = 255
thresh_orange_min = 50
thresh_orange_max = 255

img = cv.imread('test.png')
print('img size = ',img.shape)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_blue = 0*img[:,:,0] + 0*img[:,:,1] + 1*img[:,:,2]
gray_orange = 1*img[:,:,0] + 0*img[:,:,1] + 0*img[:,:,2]
cv.imwrite('gray-orange.png', gray_orange)
cv.imwrite('gray-blue.png', gray_blue)

# ---------- orange --------------
_, threshed_orange = cv.threshold(gray_orange, thresh_orange_min, thresh_orange_max, cv.THRESH_BINARY_INV)

cv.imwrite('threashed-orange.png',threshed_orange)

contours_orange, _ = cv.findContours(threshed_orange, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnt_orange = sorted(contours_orange, key=cv.contourArea)[-1]
# img_orange = cv.drawContours(img, [cnt_orange], 0, (0,255,0), 3)
# cv.imwrite('result-orange.png', img_orange)
M = cv.moments(cnt_orange)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
position_orange = [cx, cy]
print(position_orange)
img_pos_orange = cv.circle(img, (cx,cy), 6, (255,0,0), -1)
cv.imwrite('pos_orange.png', img_pos_orange)

# ---------- blue --------------
_, threshed_blue = cv.threshold(gray_blue, thresh_blue_min, thresh_blue_max, cv.THRESH_BINARY_INV)

cv.imwrite('threashed-blue.png',threshed_blue)

contours_blue, _ = cv.findContours(threshed_blue, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(len(contours_blue))
cnt_blue = sorted(contours_blue, key=cv.contourArea)[-1]
# img_blue = cv.drawContours(img, [cnt_blue], 0, (0,255,0), 3)
# cv.imwrite('result-orange.png', img_blue)
M = cv.moments(cnt_blue)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
position_blue = [cx, cy]
print(position_blue)
img_pos_blue = cv.circle(img, (cx,cy), 6, (0,0,255), -1)
cv.imwrite('pos_blue.png', img_pos_blue)
