import cv2
import glob
import numpy as np

criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.001)

objp = np.zeros((8*11,3),np.float32)             #往空的数组里乘啥啊？？？？
objp[:, :2] = np.mgrid[0:11,0:8].T.reshape(-1,2) * 20   #棋盘格*大小（踩坑②）

objpoints_left = []
objpoints_right = []
imgpoints_left = []
imgpoints_right = []

#单眼尺寸只量一次，循环内外都用它
images = sorted(glob.glob("calib_raw/*.bmp"))
img0 = cv2.imread(images[0])
W = img0.shape[1] // 2      #单眼宽
H = img0.shape[0]           #高

for fname in images:
    img = cv2.imread(fname)

    left = img[:, :W]          #需要整除！（踩坑①）
    right = img[:, W:]
    gray_left = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)

    ret_left,corners_left = cv2.findChessboardCorners(gray_left,(11,8),None)
    ret_right,corners_right = cv2.findChessboardCorners(gray_right,(11,8),None)

    if ret_left and ret_right:
        objpoints_left.append(objp.copy())


        corners2_left = cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
        imgpoints_left.append(corners2_left)

        cv2.drawChessboardCorners(left,(11,8),corners2_left,ret_left)
        #--------------------------------------------------------------------------
        objpoints_right.append(objp.copy())

        corners2_right = cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1, -1),criteria)
        imgpoints_right.append(corners2_right)

        cv2.drawChessboardCorners(right,(11,8),corners2_right,ret_right)
    else:
        print(f"{fname} is not done")
    #缩小尺寸
    left_small = cv2.resize(left,None,fx = 0.4,fy = 0.4)            #这个函数有返回值
    right_small = cv2.resize(right,None,fx = 0.4,fy = 0.4)

    cri_image = np.hstack([left_small,right_small])
    cv2.imshow("image",cri_image)

    cv2.waitKey(1)

cv2.destroyAllWindows()

print(f"左眼检出率:{len(objpoints_left)}/{len(images)}")
print(f"右眼检出率:{len(objpoints_right)}/{len(images)}")

ret_left_cali,mtx_left,dist_left,rvecs_left,tvecs_left = cv2.calibrateCamera(
    objpoints_left,imgpoints_left,
    (W,H),None,None)
ret_right_cali,mtx_right,dist_right,rvecs_right,tvecs_right = cv2.calibrateCamera(
    objpoints_right,imgpoints_right,
    (W,H),None,None)

# print(f"RMS:\nleft:{ret_left_cali},\nright:{ret_right_cali}")
# print(f"mtx:\nleft:{mtx_left},\nright:{mtx_right}")

ret_F,Mtx_L,Dist_L,Mtx_R,Dist_R,R,T,E,F = cv2.stereoCalibrate(
    objpoints_left,
    imgpoints_left,imgpoints_right,
    mtx_left,dist_left,mtx_right,
    dist_right,
    (W,H),None,None,None,None,cv2.CALIB_FIX_INTRINSIC,criteria)

print(f"RMS_F:{ret_F},\nMTX:\nleft:{Mtx_L},\nright:{Mtx_R},\n"
      f"DIST:{Dist_L},\nright:{Dist_R},"
      f"\nR:{R},\nT:{T},\nE:{E},\nF:{F}")
print(f"||T||:{np.linalg.norm(T)}")

#存数值
# np.savez("calib_result.npz",note = "2026-10-16 / 18 对 / RMS 1.0766 / 物理基线 120mm",Mtx_L = Mtx_L,Mtx_R = Mtx_R,
# Dist_L = Dist_L,Dist_R = Dist_R,R = R,T = T,E = E,F = F)

R1,R2,P1,P2,Q,ROI1,ROI2 = cv2.stereoRectify(#这样写不容易出错（踩坑③）
    Mtx_L, Dist_L,          #左眼：内参、畸变
    Mtx_R, Dist_R,          #右眼：内参、畸变
    (W,H), R, T)
mapL1,mapL2 = cv2.initUndistortRectifyMap(Mtx_L,Dist_L,R1,P1,(W,H),cv2.CV_32FC1)
mapR1,mapR2 = cv2.initUndistortRectifyMap(Mtx_R,Dist_R,R2,P2,(W,H),cv2.CV_32FC1)

left0 = img0[:,:W]
right0 = img0[:, W:]

LEFT = cv2.remap(left0,mapL1,mapL2,cv2.INTER_LINEAR)
RIGHT = cv2.remap(right0,mapR1,mapR2,cv2.INTER_LINEAR)

#缩小尺寸
LEFT_small = cv2.resize(LEFT,None,fx=0.4,fy=0.4)
RIGHT_small = cv2.resize(RIGHT,None,fx=0.4,fy=0.4)

for i in range(0, LEFT_small.shape[0], 40):        #每 40 像素画一条红线
    cv2.line(LEFT_small,(0,i),(LEFT_small.shape[1],i),(0,0,255),1)
    cv2.line(RIGHT_small,(0,i),(RIGHT_small.shape[1],i),(0,0,255),1)

cv2.imshow("rectified",np.hstack([LEFT_small,RIGHT_small]))
cv2.waitKey(0)
cv2.destroyAllWindows()