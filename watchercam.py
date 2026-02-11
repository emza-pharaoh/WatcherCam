import cv2 
import numpy as np
import time
from telegram import Bot
import os
from dotenv import load_dotenv
from ultralytics import YOLO
from collections import deque

# ==================== CONFIG




VIDEO_DURATION = 5  # seconds
VIDEO_FPS = 20
PRE_RECORD_SECONDS = 3
COOLDOWN = 15
MOTION_FRAMES_REQUIRED = 4
BUFFER_SIZE = PRE_RECORD_SECONDS * VIDEO_FPS

from collections import deque
frame_buffer = deque(maxlen=BUFFER_SIZE)
motion_history = []
load_dotenv()

PHONE_NUMBER = os.getenv("PHONE_NUMBER")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not loaded")

bot = Bot(token=BOT_TOKEN)




cap = cv2.VideoCapture(1) #Opening default camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 460)
cap.set(cv2.CAP_PROP_FPS, 30)


model = YOLO("yolov8n.pt")

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    
    history=500,
    varThreshold=50,
    detectShadows=False
)

motion_counter = 0
last_alert_time = 0
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
while cap.isOpened():
    ret, frame = cap.read()
    frame_buffer.append(frame.copy())
    if not ret:
        break
    frame = cv2.filter2D(frame, -1, kernel)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    fg_mask = bg_subtractor.apply(gray)

    #clean mask
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

   
    
    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    

    for contour in contours:

        #ignore small movements
        min_area = frame.shape[0] * frame.shape[1] * 0.02
        if cv2.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))


    #motion persistance

    motion_history.append(1 if boxes else 0)
    if len(motion_history) > MOTION_FRAMES_REQUIRED:
        motion_history.pop(0)

    motion_counter = sum(motion_history)

    # Draw boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        #send whatsapp alert
    current_time = time.time()
    if (
        motion_counter >= MOTION_FRAMES_REQUIRED
        and current_time - last_alert_time > COOLDOWN
    ):
        print("motion confirmed. RUnning person detection")

        small_frame = cv2.resize(frame, (640, 480))

        results = model(small_frame)

        person_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0: #class 0 = human
                    person_detected = True
        
        if person_detected:

            print("Person detected! Recording video...")

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"alert_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                filename,
                fourcc,
                VIDEO_FPS,
                (frame.shape[1], frame.shape[0])
            )

                        # Write buffered frames first (pre-motion)
            for buffered_frame in frame_buffer:
                out.write(buffered_frame)

            # Then record additional seconds (post-motion)
            start_record_time = time.time()

            while time.time() - start_record_time < VIDEO_DURATION:
                ret, record_frame = cap.read()
                if not ret:
                    break

                frame_buffer.append(record_frame.copy())  # Keep buffer updated
                out.write(record_frame)
                cv2.imshow("WatcherCam", record_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out.release()

            print("Sending video alert...")

            with open(filename, "rb") as vid:
                bot.send_video(
                    chat_id=CHAT_ID,
                    video=vid,
                    caption="🚨 Person detected!"
                )

            os.remove(filename)

            motion_counter = 0
            last_alert_time = current_time

        else:
            print("Motion Detected but No Person Found")
        


    #show live camera feed
    cv2.imshow("WatcherCam", frame)


    if cv2.waitKey(10) == 10:
        break

cap.release()
cv2.destroyAllWindows()