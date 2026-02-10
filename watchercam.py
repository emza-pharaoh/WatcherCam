import cv2 
import numpy as np
import time
from telegram import Bot
import os
from dotenv import load_dotenv

# ==================== CONFIG

COOLDOWN = 15
motion_history = []
MOTION_FRAMES_REQUIRED = 4

load_dotenv()

PHONE_NUMBER = os.getenv("PHONE_NUMBER")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not loaded")

bot = Bot(token=BOT_TOKEN)



cap = cv2.VideoCapture(0) #Opening default camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 460)
cap.set(cv2.CAP_PROP_FPS, 30)




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
        # ====== MOTION DETECTED ======= #
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"motion_{timestamp}.png"

        #save screenshot of motion
        cv2.imwrite(filename, frame)


        print(f"[ALERT] Motion detected -> saved {filename}")

        with open(filename, "rb") as img:
            bot.send_photo(
                chat_id = CHAT_ID ,
                photo = img,
                caption = "motion detected!"
                )
            
        os.remove(filename)
        motion_counter = 0
        last_alert_time = current_time


    #show live camera feed
    cv2.imshow("WatcherCam", frame)


    if cv2.waitKey(10) == 10:
        break

cap.release()
cv2.destroyAllWindows()