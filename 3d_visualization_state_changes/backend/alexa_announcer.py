import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

def trigger_echo_announcement(monkey, message):
    url = "https://api.voicemonkey.io/v2/trigger"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"monkey": monkey, "announcement": message}

    try:
        r = requests.post(url, headers=headers, json=payload)
        if r.status_code == 200:
            print(f"✅ Alexa said: {message}")
        else:
            print(f"❌ Failed ({r.status_code}): {r.text}")
    except Exception as e:
        print("❌ Error:", e)

def speak_for_state(state):
    monkey_map = {
        "stress": {
            "monkey": "stress_alert",
            "message": "You are showing signs of high stress. Please breathe deeply and take a moment."
        },
        "fatigue": {
            "monkey": "fatigue_alert",
            "message": "You're running low. Consider taking a short break or stretching your body."
        },
        "emergency": {
            "monkey": "emergency_alert",
            "message": "Warning! Emergency state detected. Please take immediate action."
        }
    }

    state = state.lower()
    if state in monkey_map:
        info = monkey_map[state]
        trigger_echo_announcement(info["monkey"], info["message"])
    else:
        print(f"🟡 No Alexa message for state: {state}")
