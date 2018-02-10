# Detect and track face using opencv and python
import cv2 as opencv;

class TrackFace:
    """Utility class to implement face tracking function"""
    def __init__(self, face_cascade, frame):
        self.face_cascade = face_cascade
        self.frame = frame

    def track_face(self):
        """Tracks face with a white rectangle, feel free to play around with the numbers"""
        while 1:
            _, image = self.frame.read()
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

        self.frame.release()
        opencv.destroyAllWindows()



# Load the pre-existing data set in haarcascade_frontalface_default.xml
face_cascade = opencv.CascadeClassifier(
                                        opencv.data.haarcascades + 
                                        "haarcascade_frontalface_default.xml"
                                        )
# Capture frames from camera
frame = opencv.VideoCapture(0)

# Start Tracking and Enjoy 
face_tracker = TrackFace(face_cascade, frame)
face_tracker.track_face()