📹 WatcherCam

WatcherCam is a smart motion detection surveillance system built with OpenCV and YOLOv8.
It detects meaningful human movement in real time and sends Telegram alerts when a person is identified.

This project was built as a progressive upgrade from basic motion detection to AI-enhanced human-aware surveillance.

🚀 Features

✅ Real-time webcam monitoring
✅ Background subtraction (MOG2)
✅ Motion persistence filtering
✅ Cooldown anti-spam system
✅ AI-based person detection (YOLOv8)
✅ Telegram photo alerts
✅ Automatic image cleanup
✅ Secure token handling via .env

🧠 Detection Pipeline

WatcherCam does not alert on simple pixel changes.
It follows a layered detection system:

Background subtraction detects motion.
Small movements are filtered by minimum area threshold.
Motion must persist across multiple frames.
A cooldown timer prevents alert spam.
YOLOv8 verifies that the motion belongs to a person.
If confirmed → a Telegram alert is sent.
This prevents alerts from:
Lighting flicker
Small object movement
Curtains
Minor body shifts

📦 Tech Stack

Python 3.10+
OpenCV
NumPy
Ultralytics YOLOv8
python-telegram-bot
python-dotenv

🛠 Installation

Clone the repository:
git clone https://github.com/yourusername/WatcherCam.git
cd WatcherCam


Create a virtual environment:
python -m venv .venv
.venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt

🔐 Environment Setup

Create a .env file in the root directory:

BOT_TOKEN=your_telegram_bot_token
CHAT_ID=your_chat_id


Never commit your .env file.
Make sure .gitignore contains:
.env
__pycache__/
alerts/

▶️ Running WatcherCam
python watchercam.py


When motion + person detection is confirmed:

A snapshot is captured
It is sent via Telegram
The image is deleted locally

⚙️ Configuration

Inside watchercam.py:

COOLDOWN = 15
MOTION_FRAMES_REQUIRED = 4
YOLO_SKIP_FRAMES = 5


You can tune these values for sensitivity.

🧪 Current Limitations

Single camera only

No web streaming yet

No video recording (image snapshot only)

No facial recognition

🔮 Planned Improvements

Record short video clips instead of images

Live web dashboard (remote viewing)

Mobile app integration

Face recognition (known vs unknown)

Multi-camera support

Deployment-ready architecture

📸 Example Alert
🚨 Person detected!

🎯 Purpose of This Project

This project demonstrates:
Real-time computer vision processing
AI integration into classical CV pipelines
Secure environment configuration
Event-based notification systems
Practical surveillance system design

📜 License
MIT License
