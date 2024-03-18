#Different way to go about this
import mediapipe as mp
import numpy as np
import time
import os
import cv2
import argparse
from mediapipe.python._framework_bindings.timestamp import Timestamp
from util import FaceDetector,imageOptions, videoOptions, liveOptions



def processImage(image, result):
    imageCopy = np.copy(image.numpy_view())

    for detection in result.detections:
        bbox = detection.bounding_box
        x1,y1,h,w = bbox.origin_x, bbox.origin_y, bbox.width,bbox.height

        #only blurring the face we detected, strength of blur is the last parameter
        imageCopy[y1: y1 + h,x1: x1 + w, :] = cv2.blur(imageCopy[y1: y1 + h, x1:x1 + w, :] ,(30,30))    
    return imageCopy

args = argparse.ArgumentParser()
args.add_argument("--mode", default="live")
args.add_argument("--filepath",default="./data/sample.mp4")
args = args.parse_args()

options = None

if(args.mode in ['image']):
    options = imageOptions
elif(args.mode in ['video']):
    options = videoOptions
elif(args.mode in ['live']):
    options = liveOptions

with FaceDetector.create_from_options(options) as detector:
    if(args.mode in ['image']):
        mpImage = mp.Image.create_from_file(args.filepath)
        
        result = detector.detect(mpImage)
        processedImage = processImage(mpImage,result)
    
        cv2.imshow("Frame",processedImage)
        cv2.waitKey(0)
    elif(args.mode in ['video']):
        #Capturing the video as usual
        capture = cv2.VideoCapture(args.filepath)
        ret, frame = capture.read()
        #converting the frame to an np array
        frame = np.asarray(frame)
        #creating the video file we will write the output to
        outputVideo = cv2.VideoWriter(os.path.join('.','data','output.mp4'), 
                            cv2.VideoWriter_fourcc(*'avc1'), 25,
                            (frame.shape[1],frame.shape[0]))
        while ret:
            #converting the frame to an mpImage in RGB format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            #Detecting the face in the current frame, passing the timestamp and the frame(mpImage)
            face_detector_result = detector.detect_for_video(mp_image, Timestamp.from_seconds(time.time()).value)
            #processing the frame, applying the blur to it
            processedFrame = processImage(mp_image,face_detector_result)
            #writing the frame to the output video file we created
            outputVideo.write(processedFrame)
            #reading the next frame to do the same process on that frame
            ret, frame = capture.read()
                
        capture.release()
        outputVideo.release()

    elif(args.mode in ['live']):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        #This part does not work right now, the detectAsync method does not return anything
        def getLiveResult(result, output_image: mp.Image, timestamp_ms: int):
            print('detection result: {}',result)
        
        frame = np.asarray(frame)
        while(ret):
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            faceDetectorResult = detector.detect_async(frame, Timestamp.from_seconds(time.time()).value)

            processedFrame = processImage(frame,faceDetectorResult)

            cv2.imshow("Frame",frame)
            cv2.waitKey(25)

            ret, frame = cap.read()
        cap.release()
