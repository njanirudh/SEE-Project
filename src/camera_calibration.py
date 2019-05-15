import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Iterate through all Calibration images
images = glob.glob('DataSets/Calibration/*.jpg')
for i,fname in enumerate(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners according to the size of the pattern
    # The pattern size according to this report is (8 x 6)
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)

    # If pattern is  found, add object points,
    # image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refining the corner found on the pattern in the calibration image
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                  cv.CALIB_ZERO_TANGENT_DIST, None)


print("Root mean square re-projection error : ",ret)
print("3x3 Camera intrensic matrix : ",mtx)
print("Distortion coefficients : ",dist)

