# classifier.py
import requests
from time import time
from chip.clusters import Objects as Clusters
from chip import ChipDeviceCtrl``
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
        
        self.lighting_states = {
            "focus": {"brightness": 90, "color_temp": 4000},
            "fatigue": {"brightness": 30, "color_temp": 2700},
            "stress": {"brightness": 50, "color_temp": 2200},
            "neutral": {"brightness": 70, "color_temp": 3000},
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
            logger.error(f"❌ Failed to connect to Home Assistant: {str(e)}")
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
            logger.error(f"❌ Failed to apply lighting state: {str(e)}")

# ------------------- THIS CONTROLS FAN -----------------------------
class MatterFanController:
    def __init__(self, setup_code="123456789", nodeid=2):
        self.setup_code = setup_code
        self.nodeid = nodeid
        self.controller = None
        self.device = None
        self.state_map = {
            "focus": True,
            "stress": True,
            "fatigue": False,
            "neutral": False
        }

    async def connect(self):
        try:
            self.controller = ChipDeviceCtrl.ChipDeviceController()
            self.device = await self.controller.CommissionDevice(
                setupPayload=self.setup_code, nodeid=self.nodeid
            )
            print("✅ Connected to Matter fan")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to fan: {e}")
            return False

    async def apply_state(self, state):
        if not self.device:
            if not await self.connect(): return

        power = self.state_map.get(state, False)
        try:
            await self.device.WriteAttribute(
                Clusters.OnOff.Cluster,
                [(Clusters.OnOff.Attributes.OnOff, power)]
            )
            print(f"✅ Fan turned {'ON' if power else 'OFF'} for state: {state}")
        except Exception as e:
            print(f"❌ Failed to control fan: {e}")

# ------------------- THIS CONTROLS TAPO SMART PLUG -----------------------------
class MatterSmartPlugController:
    def __init__(self, setup_code="555555555", nodeid=3, name="Smart Plug"):
        self.setup_code = setup_code
        self.nodeid = nodeid
        self.name = name
        self.controller = None
        self.device = None
        self.state_map = {
            "focus": True,
            "stress": True,
            "fatigue": False,
            "neutral": False
        }

    async def connect(self):
        try:
            self.controller = ChipDeviceCtrl.ChipDeviceController()
            self.device = await self.controller.CommissionDevice(
                setupPayload=self.setup_code, nodeid=self.nodeid
            )
            print(f"✅ Connected to Matter plug ({self.name})")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to plug ({self.name}): {e}")
            return False

    async def apply_state(self, state):
        if not self.device:
            if not await self.connect(): return

        power = self.state_map.get(state, False)
        try:
            await self.device.WriteAttribute(
                Clusters.OnOff.Cluster,
                [(Clusters.OnOff.Attributes.OnOff, power)]
            )
            print(f"✅ {self.name} turned {'ON' if power else 'OFF'} for state: {state}")
        except Exception as e:
            print(f"❌ Failed to control {self.name}: {e}")

def classify_3_states(data, baseline):
    hr = data['heart_rate']
    hrv = data['hrv']
    steps = data['steps']
    sleep = data['sleep_hours']

    focus_score = fatigue_score = stress_score = 0

    if hrv > baseline['hrv']: focus_score += 1
    if 60 < hr < 85: focus_score += 1
    if steps > 1500: focus_score += 1

    if hrv < baseline['hrv'] * 0.8: fatigue_score += 1
    if sleep < 6: fatigue_score += 1
    if steps < 500: fatigue_score += 1

    if hrv < baseline['hrv'] * 0.6: stress_score += 1
    if hr > baseline['hr'] + 20 and steps < 100: stress_score += 1

    if stress_score >= 2:
        return "stress"
    elif fatigue_score >= 2:
        return "fatigue"
    elif focus_score >= 2:
        return "focus"
    else:
        return "neutral"

async def main():
    light_controller = MatterLightController()
    fan_controller = MatterFanController()
    plug_controller = MatterSmartPlugController(name="Tapo P15 Plug")

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

    state = classify_3_states(input_data, user_baseline)
    print("Current state:", state)

    await light_controller.apply_lighting_state(state)
    await fan_controller.apply_state(state)
    await plug_controller.apply_state(state)

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
