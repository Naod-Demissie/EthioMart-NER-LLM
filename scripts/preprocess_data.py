import json
import pandas as pd
from datetime import datetime
import random

import re
import string


def extract_messages(data_paths):
    """
    Extracts messages from JSON files and returns a Pandas DataFrame.

    Parameters:
        data_paths (list): List of file paths to JSON message data.

    Returns:
        pd.DataFrame: DataFrame containing extracted message data.
    """
    messages_data = []

    for data_path in data_paths:
        channel_address = data_path.split("/")[-2]

        with open(data_path, "r") as file:
            data = json.load(file)

        for message in data.get("messages", []):
            # Combine text from different parts
            combined_text = "".join(
                part["text"] if isinstance(part, dict) and "text" in part else part
                for part in message.get("text", [])
            )

            # Append processed message data
            messages_data.append(
                {
                    "channel_name": data.get("name", ""),
                    "channel_address": channel_address,
                    "channel_id": data.get("id", ""),
                    "message_id": message.get("id", ""),
                    "date": datetime.strptime(message["date"], "%Y-%m-%dT%H:%M:%S"),
                    "message": combined_text.strip(),
                }
            )

    return pd.DataFrame(messages_data)


def clean_text(text):
    """
    Removes emojis and keeps Ethiopic characters, English letters, numbers, punctuation, and spaces.
    """
    pattern = (
        rf"[\u1200-\u137F{string.ascii_letters}{string.digits}{string.punctuation}\s]+"
    )
    cleaned_text = "".join(re.findall(pattern, text))
    return cleaned_text
