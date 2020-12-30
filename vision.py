import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
eye_cascade = cv.CascadeClassifier(
    "venv\Lib\site-packages\cv2\data\haarcascade_righteye_2splits.xml")

loadSuccess = eye_cascade.load("venv\Lib\site-packages\cv2\data\haarcascade_righteye_2splits.xml")
print(loadSuccess)
if not cap.isOpened():
    print("Cannot access camera")

while True:
    ret, frame = cap.read()
    cv.namedWindow("Stream", 1)
    cv.namedWindow("Eye", 2)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in eyes:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
        eye = gray[y: y + h, x:x + h]
        blurredEye = cv.GaussianBlur(eye, (5, 5), 1)
        val1, val2, minLoc, val3 = cv.minMaxLoc(blurredEye)
        cv.circle(blurredEye, minLoc, radius=3, color=(255, 0, 0))
        cv.imshow("Stream", frame)
        cv.imshow("Eye", blurredEye)
    if cv.waitKey(30) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
