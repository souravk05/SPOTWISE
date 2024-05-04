import cv2
import pickle
import cvzone
import numpy as np
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

width, height = 107, 48

# Video feed
#cap = cv2.VideoCapture('carPark.mp4')
cap = cv2.VideoCapture('http://192.168.167.212:4747/mjpegfeed?640x480')
# cap=cv2.VideoCapture(0)
# cap=cv2.VideoCapture("http://10.10.195.106:4747/mjpegfeed?640x480")
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

class SpaceCounter(BaseModel):
    count: int

def checkParkingSpace(imgPro):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 100:
            spaceCounter += 1
    return spaceCounter

def process_frame():
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    return imgDilate

def update_space_counter():
    while True:
        imgDilate = process_frame()
        space_counter = checkParkingSpace(imgDilate)
        print(space_counter)
        time.sleep(2)

@app.get("/space_counter/", response_model=SpaceCounter)
async def get_space_counter():
    imgDilate = process_frame()
    space_counter = checkParkingSpace(imgDilate)
    return JSONResponse(content={"count": space_counter})

if __name__ == "__main__":
    import threading

    # Start a thread to continuously update space counter
    update_thread = threading.Thread(target=update_space_counter)
    update_thread.start()

    # Run FastAPI app
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
