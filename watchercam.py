import cv2 
import numpy as np
import time
from telegram import Bot
import os
# ==================== CONFIG
PHONE_NUMBER = "+27691163533"
COOLDOWN = 15
MOTION_FRAMES_REQUIRED = 10

BOT_TOKEN = '8532265037:AAHfZf9IBLOGSqJNuMgG7UFMhJOlelWVjng'
CHAT_ID = '6076583381'
bot = Bot(token=BOT_TOKEN)



cap = cv2.VideoCapture(1) #Opening default camera

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

motion_counter = 0
last_alert_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bg_subtractor.apply(frame)

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
        if cv2.contourArea(contour) < 15000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))


        #motion persistance
    if boxes:
        motion_counter += 1
    else:
        motion_counter = 0

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