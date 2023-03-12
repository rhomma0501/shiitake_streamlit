import streamlit as st
import torch
from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time


## CFG
cfg_model_path = "models/best.pt" 

cfg_enable_url_download = False
if cfg_enable_url_download:
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt" #Configure this if you set cfg_enable_url_download to True
    cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
## END OF CFG


def imageInput(device, src):
    
    # if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path= cfg_model_path, force_reload=True) 
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')



def videoInput(device, src):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'MOV'])
    if uploaded_video != None:

        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name)
        _outputpath = os.path.join('data/video_output', os.path.basename(imgpath))
        outputpath = os.path.join('data/video_output')
        print('outputpath:', outputpath)
        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(imgpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")

        detect(weights=cfg_model_path, source=imgpath, device=0) if device == 'cuda' else detect(weights=cfg_model_path, source=imgpath, device='cpu', outputpath=outputpath)
        
        filename = os.path.basename(imgpath)
        finalpath = os.path.join(outputpath, filename)

        st_video2 = open(finalpath, 'rb')
        # st_video2 = open(outputpath, 'rb')

        video_bytes2 = st_video2.read()
        st.video(video_bytes2)
        st.write("Model Prediction")


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
              
    datasrc = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar

    st.header('🍄Shiitake Detection')
    st.subheader('👈🏽 Select options left-haned menu bar.')
    if datasrc == "Image":    
        imageInput(deviceoption, datasrc)
    elif datasrc == "Video": 
        videoInput(deviceoption, datasrc)

    

if __name__ == '__main__':
  
    main()

# Downlaod Model from url.    
@st.cache
def loadModel():
    start_dl = time.time()
    model_file = wget.download(url, out="models/")
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
if cfg_enable_url_download:
    loadModel()