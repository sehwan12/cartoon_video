import numpy as np
import cv2

#아쉬운 사례
def cartoonize_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=mask)
    return cartoon
#실패사례1
def cartoon_fail(frame):
    threshold1 = 500
    threshold2 = 1200
    aperture_size = 5

    car=cv2.Canny(frame, threshold1, threshold2, apertureSize=aperture_size)
    return car
#성공사례
def american_cartoonize_frame(frame):
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    gray=cv2.morphologyEx(gray, cv2.MORPH_CLOSE,kernel)
    gray=cv2.morphologyEx(gray, cv2.MORPH_DILATE,kernel)
    ret,cartoon=cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=cartoon)
    return cartoon

video_file='webcam.avi'
video=cv2.VideoCapture(video_file)

if video.isOpened():
    frame_prev=None
    
    while True:
        valid, frame=video.read()
        frame=cv2.resize(frame,(150,150))
        if not valid:
            break
        if frame_prev is None:
            frame_prev=frame.copy()
            continue
        
        cartoon_frame=cartoonize_frame(frame_prev)
        # cartoon_frame=cv2.resize(cartoon_frame,(50,50))
        frame_prev=frame.copy()
        edges=cartoon_fail(frame_prev)
        edges = cv2.resize(edges, (frame.shape[1], frame.shape[0]))
        frame_prev=frame.copy()
        usa=american_cartoonize_frame(frame_prev)
        # japan = cv2.resize(japan, (frame.shape[1], frame.shape[0]))
        frame_prev=frame.copy()
        
        merge = cv2.vconcat([frame, cartoon_frame,cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR),usa])


                       
        cv2.imshow("show cartoon video!",merge)
        # cv2.imshow("japan", japan)

        key= cv2.waitKey(1)
        if key==27:
            break
    cv2.destroyAllWindows()    
# 주어진 입력 비디오를 만화 스타일로 변환하여 출력





