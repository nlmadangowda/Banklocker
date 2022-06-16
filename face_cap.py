#! /usr/bin/python

import cv2
import sys
import os
from imutils import paths
import face_recognition
import pickle

def cap_face():
    name = sys.argv[1]
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)
    data_set_dir = "dataset"
    #to check the cam module functionality
    ret, frame = cam.read()
    cv2.imshow("press space to take a photo", frame)
    isdir=os.path.isdir(data_set_dir)
    print(isdir)
    if(isdir==False):
        print("\nCreating data set folder")
        os.mkdir(data_set_dir)
    else:
        print("\Dataset folder already exist")
    img_path = os.path.join(data_set_dir, name)
    isdir=os.path.isdir(img_path)
    print(isdir)
    if(isdir==False):
        print("\nCreating folder on user name")
        os.mkdir(img_path)
    else:
        print("\nThis user name is occupied, Pleas use a different name")
        quit()
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing…")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()


def tarin_face():
    print("[INFO] start processing faces…")
    imagePaths = list(paths.list_images("dataset"))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []
    knowncmd = []
    voice_cmd = sys.argv[2]

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        print(name)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,model="hog")
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)
            knowncmd.append(voice_cmd)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings…")
    data = {'encodings': knownEncodings, 'names': knownNames, 'voice_cmd' : knowncmd}
    f = open('encodings.p', 'wb')
    f.write(pickle.dumps(data))
    f.close()
    print("[INFO] serializing encodings:Done")

n= len(sys.argv)
print("Pass the user name in the following foramt \"user name\" (or) \"username\"")

if((n-1)!=2):
    print("Pass the user name in the following foramt \"user name\" (or) \"username\"")
    quit()

cap_face()
tarin_face()