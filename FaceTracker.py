# Detect, track and Denoise face using opencv and python
# Author@raghu
import cv2 as opencv;
from matplotlib import pyplot as plot

ON = True
OFF = False

class FaceTracker:    
    """Utility class to implement face tracking function"""
    def __init__(self, face_cascade):
        self.face_cascade = face_cascade


    def track_face(self, frame):
        """Tracks face with a white rectangle, feel free to play around with the numbers"""
        while 1:
            _, image= frame.read()
            
            if self.denoising == ON:
                image = opencv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

            gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY);
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, width, height) in faces:
                # play with the numbers here to change the color, rectangle co-ords
                opencv.rectangle(
                                image, 
                                (x, y), 
                                (x + width, y + height),
                                (255, 255, 255),
                                2
                                )
                roi_gray = gray[y : y + height, x : x + width]
                roi_color = image[y : y + height, x : x + width]

            opencv.imshow("Face Tracker", image);

            k = opencv.waitKey(5) & 0xFF
            if k == 27:
                # break once the user presses ESC
                break

        opencv.destroyAllWindows()

    def toggle_denoising(self, denoising):
        """Toggle denoising as per your need. Denoising can introduce lag in rendering"""
        self.denoising = denoising


# Load the pre-existing data set in haarcascade_frontalface_default.xml
face_cascade = opencv.CascadeClassifier(
                                        opencv.data.haarcascades + 
                                        "haarcascade_frontalface_default.xml"
                                       )
# play with the path to see how opencv rendering changes with each video
# 0 defaults to web cam here
path_to_video = 0  
frame = opencv.VideoCapture(path_to_video);

# Start Tracking and Enjoy ! 
ft = FaceTracker(face_cascade)
# Denoising introduces lag while rendering hence toggling off
# Feel free to play, toggle and see the result
ft.toggle_denoising(OFF);
ft.track_face(frame)