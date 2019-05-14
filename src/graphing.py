from utilities import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import json
import ast
import math
from matplotlib.ticker import MultipleLocator


def text_reader(path):
    """
    :param path: Path to the saved file.
    :return: returns an array of x and y coordinates.
    """
    x_front, y_front = [],[]
    x_back , y_back = [],[]
    back_marker = []
    front_marker = []
    with open(path) as fp:
        for line in fp.read().splitlines():
            val = ast.literal_eval(line)
            #print(val["back"])
            x_back.append(val["back"][0])
            y_back.append(val["back"][1])

            x_front.append(val["front"][0])
            y_front.append(val["front"][1])

        back_marker.append(x_back)
        back_marker.append(y_back)

        front_marker.append(x_front)
        front_marker.append(y_front)

    #return x,y
    return np.array(back_marker),np.array(front_marker)

def create_csv(path):
    """
    Creating a results csv file
    :param path: path to file
    :return:
    """
    with open(path) as fp:
        for line in fp.read().splitlines():
            val = ast.literal_eval(line)
            print(val["back"][0],val["back"][1],val["front"][0],val["front"][1],
                  math.degrees(angle(get_vector(val["back"],val["front"]),(952,0))))

def drawArrow(A, B):
    #plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
    #          length_includes_head=False, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], head_width=3, length_includes_head=True, head_length=4.0,
                  fc='red', ec='black')

if __name__ == "__main__":

    SCALE_FACTOR = 1

    start_back_marker, start_front_marker     = text_reader("Text/start.txt")
    forward_back_marker, forward_front_marker = text_reader("Text/forward.txt")
    left_back_marker, left_front_marker       = text_reader("Text/left.txt")
    right_back_marker, right_front_marker     = text_reader("Text/right.txt")

    start_back_marker *= SCALE_FACTOR
    start_front_marker *= SCALE_FACTOR
    forward_back_marker *= SCALE_FACTOR
    forward_front_marker *= SCALE_FACTOR
    left_back_marker *= SCALE_FACTOR
    left_front_marker *= SCALE_FACTOR
    right_back_marker *= SCALE_FACTOR
    right_front_marker *= SCALE_FACTOR


    #create_csv("Text/start.txt")
    #create_csv("Text/forward.txt")
    #create_csv("Text/right.txt")
    #create_csv("Text/left.txt")

    #img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results_1/Right/IMG_20190413_233532.jpg")
    img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results_1/Start/IMG_20190413_223849.jpg")
    #img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results_1/Left/IMG_20190413_232209.jpg")
    #img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results_1/Forward/IMG_20190413_225323.jpg")

    # Setting the axis size according to the sheet size of the map
    # in which the robot is run
    axes = plt.gca()
    axes.set_xlim([0, 952.5 *SCALE_FACTOR])
    axes.set_ylim([0, 685.8 *SCALE_FACTOR])

    # I want minor ticks to be every 5
    major_ticks_x = np.arange(0, 952.5, 100)
    major_ticks_y = np.arange(0, 685.8, 100)
    minor_ticks_x = np.arange(0, 952.5, 20)
    minor_ticks_y = np.arange(0, 685.8, 20)

    axes.set_xticks(major_ticks_x)
    axes.set_yticks(major_ticks_y)
    axes.set_xticks(minor_ticks_x, minor=True)
    axes.set_yticks(minor_ticks_y, minor=True)

    # Specify tick label size
    axes.tick_params(axis='both', which='major', labelsize=5)
    axes.tick_params(axis='both', which='minor', labelsize=0)

    axes.grid(which='minor', alpha=0.2)
    axes.grid(which='major', alpha=0.4)


    # Showing the BG image in the plot
    #plt.imshow(img, zorder=0, extent=[0, 952.5*SCALE_FACTOR, 685.8*SCALE_FACTOR, 0])
    #plt.imshow(img, zorder=0, extent=[0,  685.8 ,952.5, 0])

    # Draw scatter plot of the (x,y) coordinates
    # for start , forward , left and right pose
    plt.scatter(start_back_marker[0], start_back_marker[1], alpha=1 ,s=1)
    plt.scatter(forward_back_marker[0], forward_back_marker[1], alpha=1 ,s=1)
    plt.scatter(left_back_marker[0], left_back_marker[1], alpha=1,s=1)
    plt.scatter(right_back_marker[0], right_back_marker[1], alpha=1,s=1)

    # plt.scatter(start_front_marker[0], start_front_marker[1], alpha=1 ,s=1)
    # plt.scatter(forward_front_marker[0], forward_front_marker[1], alpha=1 ,s=1)
    # plt.scatter(left_front_marker[0], left_front_marker[1], alpha=1,s=1)
    # plt.scatter(right_front_marker[0], right_front_marker[1], alpha=1,s=1)

    # Drawing the pose arrow

    for i in range(len(forward_front_marker[0])-1):
        drawArrow([forward_back_marker[0][i], forward_back_marker[1][i]],
              [forward_front_marker[0][i],forward_front_marker[1][i]])

    for i in range(len(right_front_marker[0])-1):
        drawArrow([right_back_marker[0][i], right_back_marker[1][i]],
              [right_front_marker[0][i],right_front_marker[1][i]])

    for i in range(len(left_front_marker[0])-1):
        drawArrow([left_back_marker[0][i], left_back_marker[1][i]],
              [left_front_marker[0][i],left_front_marker[1][i]])

    for i in range(len(start_front_marker[0])-1):
        drawArrow([start_back_marker[0][i], start_back_marker[1][i]],
              [start_front_marker[0][i],start_front_marker[1][i]])


    plt.title('All poses')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')


    plt.savefig('all_wo_bg.png', dpi=1000)

    plt.show()

