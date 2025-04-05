import requests
from time import time
import pandas as pd

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

    # Compute continuous scores based on how far from baseline
    # All values are clamped between 0 and 1 using min/max scaling

    # --- Focus ---
    focus_hr = max(0, min(1, 1 - abs(hr - 75) / 15))  # Best focus zone ~75 bpm
    focus_hrv = max(0, min(1, (hrv - baseline['hrv']) / 20))  # Higher HRV = better focus
    focus_steps = max(0, min(1, (steps - 1500) / 1500))  # More steps = active, alert

    # Weighted average
    focus_score = 0.4 * focus_hr + 0.4 * focus_hrv + 0.2 * focus_steps

    # --- Fatigue ---
    fatigue_hrv = max(0, min(1, (baseline['hrv'] - hrv) / 30))  # Lower HRV = more fatigue
    fatigue_sleep = max(0, min(1, (6 - sleep) / 3))  # Less sleep = more fatigue
    fatigue_steps = max(0, min(1, (500 - steps) / 500))  # Inactivity = fatigue

    fatigue_score = 0.5 * fatigue_hrv + 0.3 * fatigue_sleep + 0.2 * fatigue_steps

    # --- Stress ---
    stress_hrv = max(0, min(1, (baseline['hrv'] * 0.6 - hrv) / 20))
    stress_hr = max(0, min(1, (hr - baseline['hr'] - 15) / 20))
    stress_steps = max(0, min(1, (100 - steps) / 100))

    stress_score = 0.5 * stress_hr + 0.3 * stress_hrv + 0.2 * stress_steps

    # Classify based on thresholds
    if stress_score > 0.6:
        state = "stress"
    elif fatigue_score > 0.6:
        state = "fatigue"
    elif focus_score > 0.6:
        state = "focus"
    else:
        state = "neutral"

    return state, round(focus_score, 3), round(fatigue_score, 3), round(stress_score, 3)


if __name__ == "__main__":
    df = pd.read_csv('Simulated_Daily_Biometric_Data.csv')

    user_baseline = {
        "hr": 70,
        "hrv": 60
    }

    output = []

    for _, row in df.iterrows():
        data = {
            "heart_rate": row["heart_rate"],
            "hrv": row["hrv"],
            "steps": row["steps"],
            "sleep_hours": row["sleep_hours"]
        }

        state, focus, fatigue, stress = classify_state(data, user_baseline)
        output.append({
            "timestamp": row["timestamp"],
            "heart_rate": data["heart_rate"],
            "hrv": data["hrv"],
            "steps": data["steps"],
            "sleep_hours": data["sleep_hours"],
            "state": state,
            "focus": round(focus, 2),
            "fatigue": round(fatigue, 2),
            "stress": round(stress, 2)
        })

    # Save output to CSV for 3D plotting later
    output_df = pd.DataFrame(output)
    output_df.to_csv("labeled_output.csv", index=False)
    print("✅ Exported to labeled_output.csv")
