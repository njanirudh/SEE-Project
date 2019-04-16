from main import MarkerFinder
from glob import *
import cv2 as cv2
import os
import imutils
import numpy as np

if __name__ == "__main__" :

    finder = MarkerFinder()
    finder.set_sheet_corner_id(0, 1, 2, 3)
    finder.set_vehicle_marker_id(46, 47)

    text_file = "Front.txt"

    for files in glob("Dataset/Main/Right/*.jpg"):

        #print(os.path.basename(files))

        test_img = cv2.imread(files)
        rotate = imutils.rotate_bound(test_img, 180)

        output = finder.process_image(rotate)
        if output is not None:
            cv2.imwrite("Results/Right/" + os.path.basename(files), output)
            print(finder.get_output())