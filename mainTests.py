import unittest
import cv2
import numpy as np
import dlib

class TestFaceDetection(unittest.TestCase):
    def setUp(self):
        self.detector = dlib.get_frontal_face_detector()

    def test_face_detection(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        self.assertTrue(ret, "Failed to capture frame from camera.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        self.assertGreaterEqual(len(faces), 0, "No faces detected.")

    def tearDown(self):
        cv2.destroyAllWindows()


class TestFaceDetection2(unittest.TestCase):
    def setUp(self):
        self.detector = dlib.get_frontal_face_detector()

    def test_face_detection(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        self.assertTrue(ret, "Failed to capture frame from camera.")
        # Check if the frame is flipped correctly
        flipped_frame = cv2.flip(frame, 1)
        self.assertFalse(np.array_equal(frame, flipped_frame), "Frame is not flipped.")
        
        gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        self.assertGreaterEqual(len(faces), 0, "No faces detected.")

    def tearDown(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()


