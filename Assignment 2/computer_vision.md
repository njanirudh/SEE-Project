The steps in computer vision are as following :

### Experimental Setup

1. Four aruco markers of different id are used to find the corners of the grid sheet. The aruco markers are aligned such that the first corner of the marker overlaps with the corner of the sheet.

1. The markers are also aligned to be perfectly parallel to the sheet edges. These markers should be aligned properly since the next preprocessing steps are dependedn on this.

1. Using the template given in the below picture the shapes with the relevant distances should be marked on the sheet.

1. The camera is placed rigidly such that the whole sheet is visible and the centre of the camera points approximately to the centre of the sheet.

### Algorithm :
1. Preprocessing the image involves the following steps :-
    1. Camera calibration is done on the camera to find the intrensic and extrensic parameters of the camera.
    These parameters are then used to remove the radial distorsion from the image. 

    1. The aruco markers on the four corners are then used to perform perspective transformation. This helps in removing minor skews in the image.

    1. Using the four corners of the sheet the area of the sheet is cropped.

1. Using the preprocessed image the computer vision algorithm tries to find the two aruco markers that are placed on the robot.

1. Using geometric calculation the centre of the aruco markers on the robots are found. 

1. The marker position are in pixels since it is directly calculated from the image, a transformation is performed on the pixel distances to convert it into 'mm' distance.

1. The scripts that are written for the experiment along with the documentation of each of the function is given in the following repository. (https://github.com/njanirudh/SEE-Project)
