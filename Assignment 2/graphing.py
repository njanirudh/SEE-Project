from utilities import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import json
import ast
import math

def text_reader(path):
    """
    :param path: Path to the saved file.
    :return: returns an array of x and y coordinates.
    """
    x = []
    y = []
    with open(path) as fp:
        for line in fp.read().splitlines():
            val = ast.literal_eval(line)
            #print(val["back"])
            x.append(val["back"][0])
            y.append(val["back"][1])

    return x,y



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

if __name__ == "__main__":

    x_start , y_start     = text_reader("Text/start.txt")
    x_forward , y_forward = text_reader("Text/forward.txt")
    x_left , y_left       = text_reader("Text/left.txt")
    x_right , y_right     = text_reader("Text/right.txt")

    #create_csv("Text/start.txt")
    #create_csv("Text/forward.txt")
    #create_csv("Text/right.txt")
    #create_csv("Text/left.txt")

    #img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results/Right/IMG_20190413_233532.jpg")
    #img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results/Start/IMG_20190413_223849.jpg")
    img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results/Left/IMG_20190413_232209.jpg")
    #img = cv2.imread("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 2/Results/Forward/IMG_20190413_225323.jpg")

    # Setting the axis size according to the sheet size of the map
    # in which the robot is run
    axes = plt.gca()
    axes.set_xlim([0, 952.5])
    axes.set_ylim([0, 685.8])

    # Showing the BG image in the plot
    plt.imshow(img, zorder=0, extent=[0, 952.5, 685.8, 0])
    #plt.imshow(img, zorder=0, extent=[0,  685.8 ,952.5, 0])

    # Draw scatter plot of the (x,y) coordinates
    # for start , forward , left and right pose
    plt.scatter(x_start, y_start, alpha=1 ,s=1)
    plt.scatter(x_forward, y_forward, alpha=1 ,s=1)
    plt.scatter(x_left, y_left, alpha=1,s=1)
    plt.scatter(x_right, y_right, alpha=1,s=1)

    plt.title('End pose - Right')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    plt.savefig('right.png', dpi=300)

    plt.show()

