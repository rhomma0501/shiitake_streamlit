import streamlit as st


import os
import cv2
import numpy as np
 
IMG_SIZE_h   = 60 # 画像サイズ
IMG_SIZE_w   = 30 # 画像サイズ
BLOCK_SIZE_h = 6  # 黒ブロックサイズ
BLOCK_SIZE_w = 3  # 黒ブロックサイズ
 
fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
video  = cv2.VideoWriter('ImgVideo.avi', fourcc, 20.0, (IMG_SIZE_w, IMG_SIZE_h))
 
for h in range(0, IMG_SIZE_h, BLOCK_SIZE_h):
    for w in range(0, IMG_SIZE_w, BLOCK_SIZE_w):
         
        # IMG_SIZE_h, IMG_SIZE_wの白塗り画像作成
        img = np.empty((IMG_SIZE_h, IMG_SIZE_w))
        img.fill(255)
 
        # 黒ブロックを白塗り画像に書き込み
        img[h:h+BLOCK_SIZE_h,w:w+BLOCK_SIZE_w] = np.zeros((BLOCK_SIZE_h, BLOCK_SIZE_w))
 
        # 画像出力
        cv2.imwrite('work.png', img)
        img = cv2.imread('work.png')
        video.write(img)
 
video.release()




_outputpath = 'C:/workspace_python/YOLOv5-Streamlit-Deployment/ImgVideo.avi'
# st_video2 = open(_outputpath, 'rb')

# video_bytes2 = st_video2.read()
# st.video(video_bytes2)

video_file = open(_outputpath, 'rb')
 
video_bytes = video_file.read()
with open('bin_output1.txt', mode='w') as f:
    f.write(str(video_bytes))
st.video(video_bytes, format=)