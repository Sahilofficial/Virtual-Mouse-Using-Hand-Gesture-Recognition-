'''/*
 * @Author: Sahil Kumar 
 * @Date: 2021-06-21 19:03:05 
 * @Last Modified by: Sahil Kumar 
 * @Last Modified time: 2021-06-15 19:03:05 
 */
'''


import cv2 as cv
import mediapipe as mp
import math
import statistics
import time

class Hand_Movement_Tracking:
    
    def __init__(self, STATIC_IMAGE_MODE = False, MAX_NUM_HANDS = 1, MIN_DETECTION_CONFIDENCE = 0.6, MIN_TRACKING_CONFIDENCE = 0.6):
    # STATIC_IMAGE_MODE: the solution treats the input images as a video stream. It will try to detect hands in the first input images,
    #  and upon a successful detection further localizes the hand landmarks.
        self.STATIC_IMAGE_MODE = STATIC_IMAGE_MODE

    # MAX_NUM_HANDS: Maximum number of hands to detect.
    # Default set 1.
        self.MAX_NUM_HANDS = MAX_NUM_HANDS

    # MIN_DETECTION_CONFIDENCE :Minimum confidence value `([0.0, 1.0])` from the hand detection model for the 
    # detection to be considered successful. 
        self.MIN_DETECTION_CONFIDENCE = MIN_DETECTION_CONFIDENCE

    #MIN_TRACKING_CONFIDENCE: Minimum confidence value `([0.0, 1.0])` from the landmark-tracking model for the hand 
    # landmarks to be considered tracked successfully, or otherwise hand detection will be invoked automatically on 
    # the next input image
        self.MIN_TRACKING_CONFIDENCE = MIN_TRACKING_CONFIDENCE

        self.mp_hands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands(self.STATIC_IMAGE_MODE,
                                              self.MAX_NUM_HANDS,
                                              self.MIN_DETECTION_CONFIDENCE,
                                              self.MIN_TRACKING_CONFIDENCE)
                                              
    # this is for drawing a landmarks in hands 
        self.mpDraw = mp.solutions.drawing_utils
        self.tipId = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.output = self.hands.process(image)
        

        if self.output.multi_hand_landmarks:
            for hndlmksks in self.output.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hndlmksks,
                    self.mp_hands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        pts_x = []
        pts_y = []
        bbox = []
        self.lmks_lst = []
        if self.output.multi_hand_landmarks:
            myHand = self.output.multi_hand_landmarks[handNo]
            for id, lmks in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx = int(lmks.x * w)
                cy = int(lmks.y * h)
                pts_x.append(cx)
                pts_y.append(cy)
                self.lmks_lst.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (201, 235, 52), cv.FILLED)

            xmin, xmax = min(pts_x), max(pts_x)
            ymin, ymax = min(pts_y), max(pts_y)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                (0, 255, 0), 2)

        return self.lmks_lst, bbox


    def fingersUp(self):
        fingers = []
        if self.lmks_lst[self.tipId[0]][1] < self.lmks_lst[self.tipId[0] - 1][1]:
            fingers.append(0)
        else:
            fingers.append(1)

        # Fingers
        for id in range(1, 5):
            if self.lmks_lst[self.tipId[id]][2] < self.lmks_lst[self.tipId[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmks_lst[p1][1:]
        x2, y2 = self.lmks_lst[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (180, 235, 2), t)
            cv.circle(img, (x1, y1), r, (180, 235, 2), cv.FILLED)
            cv.circle(img, (x2, y2), r, (180, 235, 2), cv.FILLED)
            cv.circle(img, (cx, cy), r, (158, 250, 5), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    fps_lst = []
    detector = Hand_Movement_Tracking()
    while True:
        ret, frame = cap.read()
        img = detector.findHands(img = frame)
        lmks_lst, bbox = detector.findPosition(img = frame) 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        fps_lst.append(fps)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70),
                            cv.FONT_HERSHEY_PLAIN, 3,
                            (201, 235, 52), 3)

        cv.imshow("Image", frame)

        if cv.waitKey(1) == 27:
            break
    print("\n******\n [+] AVERAGE FPS :-  ",int(statistics.mean(fps_lst)), "\n******\n")
if __name__ == "__main__" :
    main()