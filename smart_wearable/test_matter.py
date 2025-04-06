import asyncio
import logging
from classify_state import MatterLightController

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_light():
    # Replace with your Home Assistant URL and token
    controller = MatterLightController(
        ha_url="http://localhost:8123",
        ha_token="YOUR_LONG_LIVED_ACCESS_TOKEN"  # Get this from Home Assistant
    )
    
    # Test connection
    logger.info("Testing connection to Home Assistant...")
    if await controller.connect():
        logger.info("✅ Connected successfully")
    else:
        logger.error("❌ Connection failed")
        return
    
    # Test each state
    states = ["focus", "fatigue", "stress", "neutral"]
    for state in states:
        logger.info(f"\nTesting {state} state...")
        await controller.apply_lighting_state(state)
        await asyncio.sleep(2)  # Wait 2 seconds between states
    
    logger.info("\nTest complete!")
    
    # Clean up
    if controller.client:
        await controller.client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_light()) 