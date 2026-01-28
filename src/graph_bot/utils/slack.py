from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Dict

logger = logging.getLogger(__name__)


def send_slack_notification(webhook_url: str | None, payload: Dict[str, Any]) -> None:
    """Send a notification to Slack.

    Args:
        webhook_url: The Slack webhook URL. If None, the function does nothing.
        payload: The JSON payload to send.
    """
    if not webhook_url:
        return

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                logger.error(
                    f"Failed to send Slack notification: {response.status} {response.reason}"
                )
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
