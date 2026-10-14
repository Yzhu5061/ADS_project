import cv2
import time

cap = cv2.VideoCapture(0,cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

exposure = -4
cap.set(cv2.CAP_PROP_EXPOSURE,exposure)
print(cap.get(cv2.CAP_PROP_EXPOSURE))

start = time.time()
count = 0

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        left = gray[:,:1920]
        right = gray[:,1920:]

        count = count + 1

        cv2.imshow('left',left)
        cv2.imshow('right',right)
        Key = cv2.waitKey(1) & 0xFF
        if Key == ord('q'):
            break
        # if time.time() - start > 30:
        #     break
finally:
    print(f'fps:{count/(time.time() - start)}')
    cap.release()
    cv2.destroyAllWindows()
