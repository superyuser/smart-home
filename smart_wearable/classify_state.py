# classifier.py
import requests
from time import time
from chip.clusters import Objects as Clusters
from chip import ChipDeviceCtrl``
import asyncio

# Cache last trigger to avoid spamming Home Assistant
last_triggered = {"state": None, "timestamp": 0}

def should_trigger(new_state):
    now = time()
    if last_triggered["state"] != new_state or (now - last_triggered["timestamp"] > 180):
        last_triggered["state"] = new_state
        last_triggered["timestamp"] = now
        return True
    return False

class MatterLightController:
    def __init__(self, setup_code="10602235997"):
        self.setup_code = setup_code
        self.controller = None
        self.device = None
        
        # State to lighting mapping
        self.lighting_states = {
            "focus": {
                "brightness": 90,  # 90% brightness
                "color_temp": 4000,  # Cool white
            },
            "fatigue": {
                "brightness": 30,  # 30% brightness
                "color_temp": 2700,  # Warm white
            },
            "stress": {
                "brightness": 50,  # 50% brightness
                "color_temp": 2200,  # Very warm amber
            },
            "neutral": {
                "brightness": 70,
                "color_temp": 3000,
            }
        }

    async def connect(self):
        """Initialize connection to Matter light"""
        try:
            self.controller = ChipDeviceCtrl.ChipDeviceController()
            self.device = await self.controller.CommissionDevice(
                setupPayload=self.setup_code,
                nodeid=1,  # Node ID for the light
            )
            print("✅ Connected to Matter light")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Matter light: {e}")
            return False

    async def apply_lighting_state(self, state):
        """Apply lighting settings based on emotional state"""
        if not self.device:
            if not await self.connect():
                return
            
        settings = self.lighting_states.get(state, self.lighting_states["neutral"])
        
        try:
            # Convert brightness percentage to Matter's 0-254 range
            brightness = int((settings["brightness"] / 100) * 254)
            
            # Set brightness
            await self.device.WriteAttribute(
                Clusters.OnOff.Cluster,
                [(Clusters.OnOff.Attributes.OnOff, True)]  # Turn on
            )
            await self.device.WriteAttribute(
                Clusters.LevelControl.Cluster,
                [(Clusters.LevelControl.Attributes.CurrentLevel, brightness)]
            )
            
            # Set color temperature
            color_temp = settings["color_temp"]
            await self.device.WriteAttribute(
                Clusters.ColorControl.Cluster,
                [(Clusters.ColorControl.Attributes.ColorTemperatureMireds, color_temp)]
            )
            
            print(f"✅ Applied {state} lighting state")
            
        except Exception as e:
            print(f"❌ Failed to apply lighting state: {e}")

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

async def main():
    # Initialize Matter light controller
    light_controller = MatterLightController()
    
    # Example input data
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

    # Classify state and apply lighting
    state = classify_state(input_data, user_baseline)
    print("Current state:", state)
    
    # Apply the lighting state
    await light_controller.apply_lighting_state(state)

if __name__ == "__main__":
    asyncio.run(main())

# def send_webhook(state):
#     webhook_map = {
#         "focus": "focus_triggered",
#         "fatigue": "fatigue_triggered",
#         "stress": "stress_triggered"
#     }

#     webhook_id = webhook_map.get(state)
#     if not webhook_id:
#         print("Unknown state or no action.")
#         return

#     url = f"http://localhost:8123/api/webhook/{webhook_id}"
#     try:
#         r = requests.post(url)
#         if r.status_code == 200:
#             print(f"✅ Triggered: {webhook_id}")
#         else:
#             print(f"⚠️ Webhook error: {r.status_code}")
#     except Exception as e:
#         print("❌ Webhook failed:", e)

# if __name__ == "__main__":
#     # Example input from simulated or real data