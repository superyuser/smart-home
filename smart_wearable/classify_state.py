# classifier.py
import requests
from time import time

# Cache last trigger to avoid spamming Home Assistant
last_triggered = {"state": None, "timestamp": 0}

def should_trigger(new_state):
    now = time()
    if last_triggered["state"] != new_state or (now - last_triggered["timestamp"] > 180):
        last_triggered["state"] = new_state
        last_triggered["timestamp"] = now
        return True
    return False

def classify_state(data, baseline):
    hr = data['heart_rate']
    hrv = data['hrv']
    steps = data['steps']
    sleep = data['sleep_hours']

    focus_score = 0
    fatigue_score = 0
    stress_score = 0

    if hrv > baseline['hrv']:
        focus_score += 1
    if 60 < hr < 85:
        focus_score += 1
    if steps > 1500:
        focus_score += 1

    if hrv < baseline['hrv'] * 0.8:
        fatigue_score += 1
    if sleep < 6:
        fatigue_score += 1
    if steps < 500:
        fatigue_score += 1

    if hrv < baseline['hrv'] * 0.6:
        stress_score += 1
    if hr > baseline['hr'] + 20 and steps < 100:
        stress_score += 1

    if stress_score >= 2:
        return "stress"
    elif fatigue_score >= 2:
        return "fatigue"
    elif focus_score >= 2:
        return "focus"
    else:
        return "neutral"

def send_webhook(state):
    webhook_map = {
        "focus": "focus_triggered",
        "fatigue": "fatigue_triggered",
        "stress": "stress_triggered"
    }

    webhook_id = webhook_map.get(state)
    if not webhook_id:
        print("Unknown state or no action.")
        return

    url = f"http://localhost:8123/api/webhook/{webhook_id}"
    try:
        r = requests.post(url)
        if r.status_code == 200:
            print(f"✅ Triggered: {webhook_id}")
        else:
            print(f"⚠️ Webhook error: {r.status_code}")
    except Exception as e:
        print("❌ Webhook failed:", e)

if __name__ == "__main__":
    # Example input from simulated or real data
    input_data = {
        "heart_rate": 95,
        "hrv": 35,
        "steps": 200,
        "sleep_hours": 5.0
    }

    user_baseline = {
        "hr": 70,
        "hrv": 60
    }

    state = classify_state(input_data, user_baseline)
    print("Current state:", state)

    if should_trigger(state):
        send_webhook(state)
