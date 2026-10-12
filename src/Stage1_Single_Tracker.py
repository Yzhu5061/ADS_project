import cv2
import numpy as np

cap = cv2.VideoCapture(1)     #捕捉摄像头

out_write = False
out = None


try:
    while True:
        ret, frame = cap.read()    #读取帧
        if not ret:      # 读取失败保护
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)     #转换成HSV图像

        lower_green = np.array([50, 40, 40])                 #颜色阈值范围，最小和最大
        upper_green = np.array([100, 250, 250])
        mask = cv2.inRange(hsv, lower_green, upper_green)    #提取掩膜
        res = cv2.bitwise_and(frame, frame, mask=mask)       #复合图像

        contours , hierarchy = cv2.findContours(mask , cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #↑这里的contours是一个列表，这个列表里装着很多的数组numpy，数组里装着很多的点
        largest_cnt = None
        max_area = 0
        for cnt in contours:                #遍历所有的轮廓
            area = cv2.contourArea(cnt)
            if not area > 500:              #把噪点筛出去
                continue
            if area > max_area:             #确定最大的面积
                max_area = area
                largest_cnt = cnt
        if largest_cnt is not None:         #保护措施：如果捕捉到才画轮廓
            cv2.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 3)
            M = cv2.moments(largest_cnt)    #计算几何距，加权平均值。
            if M["m00"] != 0:               #同样，防止除以零而崩溃
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print(f"\r质心坐标：({cx}, {cy}),面积：{max_area}",end = '',flush = True)   #打印中心和面积
                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)  #画出质心
        else:
            print("\r","no contours",end='',flush=True)               #没捕捉到就提醒


        if out_write:
            out.write(frame)

        Key = cv2.waitKey(1)
        if Key == ord('q'):
            if out is not None:
                out.release()
            break
        if Key == ord('s'):
            if not out_write:
                out_write = True
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter('output.avi',fourcc,20.0, (640,480),True)
            else:
                out_write = False
                out.release()
                out = None



        cv2.imshow("res", res)         #复合后的图像
        cv2.imshow("mask", mask)       #掩膜
        cv2.imshow("frame", frame)     #原图像



finally:
    cap.release()
    cv2.destroyAllWindows()