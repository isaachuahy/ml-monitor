import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("alerting")

def send_discord_alert(message: str):
    """
    Sends an alert to a Discord channel.
    Requires the following environment variables:
    - DISCORD_WEBHOOK_URL: The URL of the Discord webhook
    """

    # Get the webhook URL from the environment variable
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    # Check if the environment variable is set
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL is not set")
        return
    
    payload = {
        "content": message
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info(f"Alert sent to Discord: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send alert to Discord: {e}")