# -*- coding: utf-8 -*-

import cv2
import sys


# 입력파일 지정하기
image_file = "face.jpeg"

# 캐스케이드 파일의 경로 지정하기
cascade_file = "haarcascade_eye.xml"

# 이미지 읽어 들이기
image = cv2.imread(image_file)
# 그레이스케일로 변환하기
image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 얼굴 인식 특징 파일 읽어 들이기
cascade = cv2.CascadeClassifier(cascade_file)
#얼굴인식 실행하기
face_list = cascade.detectMultiScale(image_gs)

if len(face_list) > 0:
    # 인식한 부분 표시하기
    print(face_list)
    color = (0, 0, 255)
    for face in face_list:
        x,y,w,h = face
        cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=2)
    # 파일로 출력하기
    cv2.imwrite("facedetect_output.PNG", image)
else:
    print("no face")
        