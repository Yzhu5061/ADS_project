import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
#DSHOW 在 3840×1080 只有 1.2 fps，MSMF 有 28.5 fps,MSMF 下索引编号与 DSHOW 不同：双目=0，笔记本=1
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#先定格式再定格式下的分辨率

if not cap.isOpened():
    print('相机没打开')
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

exposure = -4
cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
print('曝光设定值:', exposure)

start = time.time()
count = 0

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if count == 0:
            print('帧尺寸:', frame.shape)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w = frame.shape[1] // 2
        left = gray[:, :w]
        right = gray[:, w:]

        count += 1

        cv2.imshow('left', left)
        cv2.imshow('right', right)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # if time.time() - start > 30:
        #     break
finally:
    print(f'fps:{count / (time.time() - start)}')
    cap.release()
    cv2.destroyAllWindows()
