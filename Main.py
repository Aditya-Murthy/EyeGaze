import cv2 as cv
import numpy as np
import math
import tkinter
from tkinter import simpledialog
from tkinter import messagebox


def init_windows():
    cv.namedWindow("Main_frame", 1)
    cv.namedWindow("Eye", 2)
    cv.namedWindow("Dialogue",1)


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


def get_eye_contour(eyes_vec, frame, thresh_ip):
    for (x, y, w, h) in eyes_vec:
        eye = frame[y:y + h, x:x + w]
        dst = cv.fastNlMeansDenoisingColored(eye, None, 10, 10, 7, 21)
        blur = cv.GaussianBlur(dst, (5, 5), 0)
        inv = cv.bitwise_not(blur)
        thresh = cv.cvtColor(inv, cv.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv.erode(thresh, kernel, iterations=1)
        ret, thresh1 = cv.threshold(erosion, thresh_ip, 255, cv.THRESH_BINARY)
        cnts, hierarchy = cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return cnts

def get_thresh_img(eyes_vec, frame, thresh_ip):
    for (x, y, w, h) in eyes_vec:
        eye = frame[y:y + h, x:x + w]
        dst = cv.fastNlMeansDenoisingColored(eye, None, 10, 10, 7, 21)
        blur = cv.GaussianBlur(dst, (3, 3), 0)
        inv = cv.bitwise_not(blur)
        thresh = cv.cvtColor(inv, cv.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv.erode(thresh, kernel, iterations=1)
        ret, thresh1 = cv.threshold(erosion, thresh_ip, 255, cv.THRESH_BINARY)
        return thresh1


def get_final_contour(cnts, centroid, flag=15000):
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


def get_right_calib_key():
    r_calib_key = False
    parent = tkinter.Tk()
    parent.overrideredirect(1)
    parent.attributes("-topmost", True)
    parent.withdraw()

    response = messagebox.showinfo("Right gaze calibration", "Shift your gaze to the desired right boundary and hold constant gaze for 3 seconds. Click OK when ready")
    if response == 'ok':
        r_calib_key = True
    return r_calib_key

def get_left_calib_key():
    l_calib_key = False
    parent = tkinter.Tk()
    parent.overrideredirect(1)
    parent.attributes("-topmost", True)
    parent.withdraw()

    response = messagebox.showinfo("Left gaze calibration", "Shift your gaze to the desired left boundary and hold constant gaze for 3 seconds. Click OK when ready")
    if response == 'ok':
        l_calib_key = True
    return l_calib_key


    
def display_gaze_direction(eye_loc, x_right, x_left, frame):

    if eye_loc[0] > x_left:
        cv.putText(frame, "Left", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    elif eye_loc[0] < x_right:
        cv.putText(frame, "Right", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    else:
        cv.putText(frame, "Center", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


def run_threshold_calib(cap, load_status, eye_cascade, threshold_ip):
    parent = tkinter.Tk()
    parent.overrideredirect(1)
    parent.attributes("-topmost", True)
    parent.withdraw()
    messagebox.showinfo("Threshold calibration", "Repeatedly click '+' and '-' to adjust image thresholding until only pupil is visible in the Binary image")
    cv.namedWindow("Stream", 1)
    cv.namedWindow("Eye", 2)
    cv.namedWindow("Binary Image", 2)
    while cap.isOpened() and load_status:
        ret, frame = cap.read()
        eye_vec = detect_eyes(frame, eye_cascade)
        if len(eye_vec) == 0 or eye_vec is None:
            pass
        else:
            mark_eye(eye_vec, frame)
            eye = get_eye_frame(eye_vec, frame)
            centroid = get_eye_centroid(eye_vec, frame)
            cnts = get_eye_contour(eye_vec, frame, threshold_ip)
            thresh = get_thresh_img(eye_vec, frame, threshold_ip)
            fnlcntr = get_final_contour(cnts, centroid)
            center = get_eye_location(fnlcntr)
            if center is not None:
                diffx = eye_vec[0][0] + center[0]
                diffy = eye_vec[0][1] + center[1]
                centerf = (diffx, diffy)
                cv.circle(frame, centerf, 3, (0, 255, 0), 1)
                draw_eye(center, eye)

            cv.imshow("Eye", eye)
            cv.imshow("Binary Image", thresh)
        cv.imshow("Stream", frame)

        if cv.waitKey(10) == ord("q"):
            break
        elif cv.waitKey(10) == ord("+") and threshold_ip < 255:
            threshold_ip += 2
        elif cv.waitKey(10) == ord("-") and threshold_ip > 0:
            threshold_ip -= 2
        else:
            pass
        # print(thres_ip)
    cv.destroyAllWindows()
    return threshold_ip


# Point the cascade classifier loader to local copy of haarcascade_righteye_2splits.xml
eye_cascade, load_status = load_cascade_classifier_from("venv\Lib\site-packages\cv2\data\haarcascade_righteye_2splits.xml")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

l_calib_array=[]
l_calib_counter= 0
l_key = False
l_calib_flag = True
l_command_counter = 0
r_calib_array=[]
r_calib_counter= 0
r_key = False
r_calib_flag = True
r_command_counter = 0
default_threshold = 210

calib_threshold = run_threshold_calib(cap, load_status, eye_cascade, default_threshold)
init_windows()


if cap.isOpened() and load_status:
    while True:
        ret, full_frame = cap.read()
        eye_vec = detect_eyes(full_frame, eye_cascade)
        mark_eye(eye_vec, full_frame)
        eye_snip = get_eye_frame(eye_vec, full_frame)
        centroid = get_eye_centroid(eye_vec, full_frame)
        contours = get_eye_contour(eye_vec, full_frame, calib_threshold)
        final_contour = get_final_contour(contours, centroid)
        eye_loc = get_eye_location(final_contour)
        draw_eye(eye_loc, eye_snip)

# -- start of calibration--

        if eye_snip is not None:
            cv.imshow("Eye", eye_snip)
            if l_command_counter == 0:
                l_key = get_left_calib_key()
                l_command_counter += 1
            if l_calib_flag:
                if eye_loc is not None and l_key:
                    if l_calib_counter < 20:
                        l_calib_array.append(eye_loc[0])
                        l_calib_counter += 1
                    else:
                        x_left = np.average(l_calib_array)
                        print("x_left=  ", x_left)
                        l_calib_flag = False

            if r_command_counter == 0 and l_calib_flag == False:
                r_key = get_right_calib_key()
                r_command_counter += 1
            if r_calib_flag:
                if eye_loc is not None and r_key:
                    if r_calib_counter < 20:
                        r_calib_array.append(eye_loc[0])
                        r_calib_counter += 1
                    else:
                        x_right = np.average(r_calib_array)
                        print("x_right=  ", x_right)
                        r_calib_flag = False
# -- end of calibration --
            if r_calib_flag == False and eye_loc is not None:
                display_gaze_direction(eye_loc, x_right, x_left, full_frame)
        cv.imshow("Main_frame", full_frame)
        if cv.waitKey(30) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()