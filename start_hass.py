import asyncio
import os
import logging
from homeassistant import bootstrap
from homeassistant.core import HomeAssistant
from homeassistant.config import async_hass_config_yaml

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    """Start Home Assistant."""
    try:
        # Get the config directory path
        config_dir = os.path.expanduser("~/.homeassistant")
        logger.info(f"Using config directory: {config_dir}")
        
        # Create the config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        logger.info("Config directory created/verified")
        
        # Load the configuration
        logger.info("Loading configuration...")
        config = await async_hass_config_yaml(config_dir)
        
        # Initialize Home Assistant
        logger.info("Initializing Home Assistant...")
        hass = HomeAssistant()
        hass.config.config_dir = config_dir
        
        logger.info("Setting up Home Assistant...")
        await bootstrap.async_setup_hass(hass, config)
        
        logger.info("Starting Home Assistant...")
        await hass.async_run()
        
    except Exception as e:
        logger.error(f"Failed to start Home Assistant: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 