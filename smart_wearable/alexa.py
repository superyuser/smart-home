import requests
import os
from dotenv import load_dotenv
from classify_state import classify_3_states
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN") or "fa6ca360f339ccef6bb3540a4e95987d_aa52b9593fbaae67a6c7fe17cef5e1ca"

def trigger_echo_announcement(monkey, message):
    url = "https://api.voicemonkey.io/v2/trigger"  # ⬅️ v2 endpoint
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "monkey": monkey,
        "announcement": message
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"✅ Alexa said: {message}")
        else:
            print(f"❌ Failed ({response.status_code}): {response.text}")
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
        }
    }

    if state in monkey_map:
        monkey = monkey_map[state]["monkey"]
        message = monkey_map[state]["message"]
        trigger_echo_announcement(monkey, message)
    else:
        print(f"🟡 No voice alert for state: {state}")

state = classify_3_states(input_data, baseline)
print("Current state:", state)
speak_for_state(state)
