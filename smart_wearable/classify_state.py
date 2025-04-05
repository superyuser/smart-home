# classifier.py
import requests
from time import time
from matter_server.client.client import MatterClient
import asyncio
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache last trigger to avoid spamming Home Assistant
last_triggered = {"state": None, "timestamp": 0}

def should_trigger(new_state):
    now = time()
    if last_triggered["state"] != new_state or (now - last_triggered["timestamp"] > 180):
        last_triggered["state"] = new_state
        last_triggered["timestamp"] = now
        return True
    return False

# --------------- THIS CONTROLS LIGHT ---------------------------------
class MatterLightController:
    def __init__(self, ha_url="http://localhost:8123", ha_token=None):
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json"
        }
        
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
        """Verify connection to Home Assistant"""
        try:
            response = requests.get(
                f"{self.ha_url}/api/",
                headers=self.headers
            )
            if response.status_code == 200:
                logger.info("✅ Connected to Home Assistant")
                return True
            else:
                logger.error(f"❌ Failed to connect to Home Assistant: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Home Assistant: {str(e)}")
            return False

    async def apply_lighting_state(self, state):
        """Apply lighting settings based on emotional state"""
        if not self.ha_token:
            logger.error("❌ Home Assistant token not set")
            return
            
        settings = self.lighting_states.get(state, self.lighting_states["neutral"])
        
        try:
            # Turn on the light and set brightness
            data = {
                "entity_id": "light.matter_light",  # Replace with your light's entity_id
                "brightness": int((settings["brightness"] / 100) * 255),
                "color_temp": settings["color_temp"]
            }
            
            response = requests.post(
                f"{self.ha_url}/api/services/light/turn_on",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Applied {state} lighting state")
            else:
                logger.error(f"❌ Failed to apply lighting state: {response.status_code}")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply lighting state: {str(e)}")

# ------------------- THIS CONTROLS FAN -----------------------------
class MatterFanController:
    def __init__(self, setup_code="123456789", nodeid=2):
        self.setup_code = setup_code
        self.nodeid = nodeid
        self.controller = None
        self.device = None

        self.state_map = {
            "focus": True,     # fan ON
            "stress": True,    # fan ON
            "fatigue": False,  # fan OFF
            "neutral": False   # fan OFF
        }

    async def connect(self):
        """Connect to the Matter fan device."""
        try:
            self.controller = ChipDeviceCtrl.ChipDeviceController()
            self.device = await self.controller.CommissionDevice(
                setupPayload=self.setup_code,
                nodeid=self.nodeid,
            )
            print("✅ Connected to Matter fan")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to fan: {e}")
            return False

    async def apply_state(self, state):
        """Turn the fan ON or OFF based on emotional state."""
        if not self.device:
            if not await self.connect():
                return

        power = self.state_map.get(state, False)

        try:
            await self.device.WriteAttribute(
                Clusters.OnOff.Cluster,
                [(Clusters.OnOff.Attributes.OnOff, power)]
            )
            print(f"✅ Fan turned {'ON' if power else 'OFF'} for state: {state}")
        except Exception as e:
            print(f"❌ Failed to control fan: {e}")


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

async def test_light_integration():
    """Test function to verify Matter light integration"""
    light_controller = MatterLightController()
    
    # Test connection
    logger.info("\nTesting Matter light connection...")
    if await light_controller.connect():
        logger.info("✅ Connection successful")
    else:
        logger.error("❌ Connection failed")
        return
    
    # Test each lighting state
    states = ["focus", "fatigue", "stress", "neutral"]
    for state in states:
        logger.info(f"\nTesting {state} state...")
        await light_controller.apply_lighting_state(state)
        await asyncio.sleep(2)  # Wait 2 seconds between states
    
    logger.info("\nLight integration test complete!")
    
    # Clean up
    if light_controller.client:
        await light_controller.client.disconnect()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_light_integration())

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