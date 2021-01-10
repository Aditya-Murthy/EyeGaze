import cv2 as cv
import numpy as np
import math


def init_windows():
    cv.namedWindow("Main_frame", 1)
    cv.namedWindow("Eye", 2)



def load_cascade_classifier_from(filepath, status=1):
    if type(filepath) == str:
        eye_cascade = cv.CascadeClassifier(filepath)
        load_success = eye_cascade.load(filepath)
        return eye_cascade, load_success
    else:
        print("Please enter file path in string format")
        return None


def detect_eyes(frame, eye_cascade):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    eyes_vec = eye_cascade.detectMultiScale(gray_frame, 1.1, 50)
    return eyes_vec


def mark_eye(eyes_vec, target_frame):
    for (x, y, w, h) in eyes_vec:
        cv.rectangle(target_frame, (x, y), (x + w, y + h), (255, 0, 0))


def get_eye_frame(eyes_vec, target_frame):
    for (x, y, w, h) in eyes_vec:
        eye = target_frame[y: y + h, x:x + w]
        eye = cv.cvtColor(eye, cv.COLOR_BGR2GRAY)
        return eye


def get_eye_centroid(eyes_vec, target_frame, process_thresh=127):
    for (x, y, w, h) in eyes_vec:
        eye = target_frame[y: y + h, x:x + w]
        gray_eye = cv.cvtColor(eye, cv.COLOR_BGR2GRAY)
        ret, thresh_eye = cv.threshold(gray_eye, process_thresh, 255, 0)
        M = cv.moments(thresh_eye)
        if M["m00"] == 0:
            cX = 0
            cY = 0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
        return centroid


def get_eye_contour(eyes_vec, frame):
    for (x, y, w, h) in eyes_vec:
        eye = frame[y:y + h, x:x + w]
        dst = cv.fastNlMeansDenoisingColored(eye, None, 10, 10, 7, 21)
        blur = cv.GaussianBlur(dst, (5, 5), 0)
        inv = cv.bitwise_not(blur)
        thresh = cv.cvtColor(inv, cv.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv.erode(thresh, kernel, iterations=1)
        ret, thresh1 = cv.threshold(erosion, 210, 255, cv.THRESH_BINARY)
        cnts, hierarchy = cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return cnts


def get_final_contour(cnts, centroid, flag=10000):
    if cnts is None:
        return None
    else:
        final_cnt = None
        for cnt in cnts:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            distance = abs(centroid[0] - x) + abs(centroid[1] - y)
            if distance < flag:
                flag = distance
                final_cnt = cnt
            else:
                continue
        return final_cnt


def get_eye_location(final_cnt):
    if final_cnt is None:
        return None
    else:
        (x, y), radius = cv.minEnclosingCircle(final_cnt)
        center = (int(x), int(y))
        return center


def draw_eye(eye_loc, eye_frame):
    if eye_loc is None:
        return
    else:
        cv.circle(eye_frame, eye_loc, 3, (255, 0, 0), 2)


def display_gaze_direction(center, ext_counter, frame):
    if ext_counter == 0:
        init_center = center[0]
    if init_center - center[0] > 2:
        cv.putText(frame, "Right", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    elif init_center - center[0] < -3:
        cv.putText(frame, "Left", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    else:
        cv.putText(frame, "Center", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

#global vars
eye_cascade, load_status = load_cascade_classifier_from("venv\Lib\site-packages\cv2\data\haarcascade_righteye_2splits.xml")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
init_windows()

if cap.isOpened() and load_status:
    while True:
        ret, full_frame = cap.read()
        eye_vec = detect_eyes(full_frame, eye_cascade)
        mark_eye(eye_vec, full_frame)
        eye_snip = get_eye_frame(eye_vec, full_frame)
        centroid = get_eye_centroid(eye_vec, full_frame)
        contours = get_eye_contour(eye_vec, full_frame)
        final_contour = get_final_contour(contours, centroid)
        eye_loc = get_eye_location(final_contour)
        draw_eye(eye_loc, eye_snip)

        cv.imshow("Main_frame", full_frame)
        if eye_snip is not None:
            cv.imshow("Eye", eye_snip)
        if cv.waitKey(30) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()



