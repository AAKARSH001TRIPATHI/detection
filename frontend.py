import streamlit as st
import cv2
import numpy as np
st.set_page_config(page_title="FACE DETECTION SYSTEM",page_icon="icon.jpg")
choice=st.sidebar.selectbox("MY MENU",("HOME", "IMAGE"))
detectface=cv2.CascadeClassifier("face.xml")
st.title("Face Detection System")
if(choice=="HOME"):
    st.header("welcome")
    st.image("icon1.gif")
elif(choice=="IMAGE"):
    img=st.file_uploader("please upload your image")
    if img:
        bytes=img.getvalue()
        img=cv2.imdecode(np.frombuffer(bytes,np.uint8),cv2.IMREAD_COLOR)
        face=detectface.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        for (x,y,w,h) in face:
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),5)
        st.image(img,channels='BGR')
