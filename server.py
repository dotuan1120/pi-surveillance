import time
import pyrebase
import os

from imutils import build_montages
import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import shutil

# convert the video from avi format to mp4 format
def convert_avi_to_mp4(avi_file_path, output_name):
    pro = os.popen(
        "ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict "
        "experimental -f mp4 '{output}'".format(
            input=avi_file_path, output=output_name))
    pro.read()
    return True

# Convert the video from avi format to mp4 format
# Upload videos to the Firebase Cloud Storage
# Move the videos to the local directory
def handle_outputs(video_name_avi, image_name, date, time):
    # Initialize video name and paths
    video_name_mp4 = "({ts}).mp4".format(ts=time)
    video_cloud_path = 'videos/{date}/{vid}'.format(date=date, vid=video_name_mp4)
    video_name_avi_path = "/home/tuan/Downloads/pi-surveillance/{vid}".format(vid=video_name_avi)
    local_storage_path = '/home/tuan/Downloads/pi-surveillance/{loc}'.format(loc=date)
    # Convert from avi to mp4 format
    convert_avi_to_mp4(video_name_avi_path, video_name_mp4)
    # Upload to the cloud storage and realtime database
    storage.child(video_cloud_path).put(video_name_mp4)
    vidRef = storage.child('videos/{date}/{vid}'.format(date=date, vid=video_name_mp4)).get_url(None)
    db.child("surveillance").child(date).child(time).update({"video": vidRef})
    # Move video and images to the local storage and delete the avi file
    move_file(video_name_avi, video_name_mp4, image_name, local_storage_path)

# Create a local directory to store videos and images
def create_dir(date):
    parent_dir = "/home/tuan/Downloads/pi-surveillance"
    path = os.path.join(parent_dir, date)
    try:
        os.mkdir(path)
        print("Directory '%s' created" % date)
    except OSError as error:
        print(error)

# Move the file to the destination directory
def move_file(video_name_avi, video_name_mp4, image_name, local_storage_path):
    try:
        os.remove(video_name_avi)
        shutil.move(video_name_mp4, local_storage_path)
        shutil.move(image_name, local_storage_path)
        print("success")
    except OSError as error:
        print(error)

# Firebase configuration
config = {
    "apiKey": "AIzaSyDzALNGaFzBfKTwQiEvht1brD5KxVqGyEE",
    "authDomain": "pi-surveillance-9dc05.firebaseapp.com",
    "databaseURL": "https://pi-surveillance-9dc05.firebaseio.com",
    "projectId": "pi-surveillance-9dc05",
    "storageBucket": "pi-surveillance-9dc05.appspot.com",
    "messagingSenderId": "408628332478",
    "appId": "1:408628332478:web:4b1ac996f34981136d9bef"
};

# Initialize the Firebase services
firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-mW", "--montageW", required=True, type=int,
                    help="montage frame width")
    ap.add_argument("-mH", "--montageH", required=True, type=int,
                    help="montage frame height")
    args = vars(ap.parse_args())

    # initialize the ImageHub object
    imageHub = imagezmq.ImageHub()

    # initialize the list of class labels MobileNet SSD was trained to detect,
    # then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the frame dictionary
    frameDict = {}

    # assign montage width and height of the video interface
    mW = args["montageW"]
    mH = args["montageH"]


    frame_counter = 0
    # start looping over all the frames
    while True:
        # receive Pi name and frame from the RPi and send REPLY to the Pi
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')

        # resize the frame to have a maximum width of 600 pixels
        # construct a blob using the frame dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        size = (w, h)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # detected variable is True when there is human in the frame, otherwise is False
        detected = False

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # check if the confidence is greater than the minimum confidence
            # and the detection contains human presence
            if confidence > args["confidence"] and int(detections[0, 0, i, 1]) == 15:

                detected = True

                # extract the index of the class label from the detections
                idx = int(detections[0, 0, i, 1])

                # if there are 10 consecutive frames containing human(s)
                if frame_counter == 10:

                    # Creat the date and time
                    date = datetime.datetime.now().strftime("%d-%b-%Y")
                    time = datetime.datetime.now().strftime("%H:%M:%S")

                    # Create a storage directory which its name is the date
                    create_dir(date)

                    # Upload the image first
                    image_name = '({ts}).jpg'.format(ts=time)
                    cv2.imwrite(image_name, frame)
                    image_cloud_path = 'images/{date}/{img}'.format(date=date, img=image_name)
                    storage.child(image_cloud_path).put(image_name)
                    imgRef = storage.child('images/{date}/{img}'.format(date=date, img=image_name)).get_url(None)

                    # Update name and image in Realtime Database
                    db.child("surveillance").child(date).child(time).update({"image": imgRef})
                    db.child("surveillance").child(date).child(time).update({"name": time})

                    # Initialize a video object
                    video_name_avi = "({ts}).avi".format(ts=time)
                    result = cv2.VideoWriter(video_name_avi, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                    print("begin to record")
                    frame_counter += 1
                # If there are more than 10 consecutive frames containing human(s), begin to record the video
                elif frame_counter > 10:

                    result.write(frame)

                    # if there are more than 100 consecutive frames containing human(s),
                    # reset frame counter to stop recording, and handle the video
                    if frame_counter > 110:
                        frame_counter = 0
                        print("end recording")

                        # upload video to cloud storage
                        handle_outputs(video_name_avi, image_name, date, time)
                    else:
                        frame_counter += 1
                else:
                    frame_counter += 1

        # If there is no human in the current frame
        if detected is False:

            # reset frame counter to stop recording, and handle the video if the program are recording
            if frame_counter > 10:

                # upload video to cloud storage
                handle_outputs(video_name_avi, image_name, date, time)
                frame_counter = 0

            # reset frame counter if there are less than 10 consecutive frames containing human
            else:
                frame_counter = 0
        print("frame_counter = ", frame_counter)

        # update the new frame in the frame dictionary
        frameDict[rpiName] = frame

        # build a montage using images in the frame dictionary
        montages = build_montages(frameDict.values(), (w, h), (mW, mH))

        # display the montage(s) on the screen
        for (i, montage) in enumerate(montages):
            cv2.imshow("Monitor ({})".format(i),
                       montage)

        # detect any key pressed
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # Cleanup
    cv2.destroyAllWindows()
