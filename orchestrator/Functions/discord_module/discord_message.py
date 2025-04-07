import builtins

from Data.VectorDB import retrieve_contact
from Secrets.keys import bot_token
from dependencies import BaseTool
from models.llm import gemini_pro
import requests
import json

model = gemini_pro


def get_dm_channel(user_id):
    url = 'https://discord.com/api/v9/users/@me/channels'
    headers = {
        'Authorization': f'Bot {bot_token}',
        'Content-Type': 'application/json',
    }
    payload = {
        'recipient_id': user_id
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        # Successfully retrieved the DM channel
        return response.json()['id']
    else:
        # Handle errors
        print(f"Error getting DM channel: {response.status_code}")
        print(response.json())
        return None


def send_message(channel_id, message):
    url = f'https://discord.com/api/v9/channels/{channel_id}/messages'
    headers = {
        'Authorization': f'Bot {bot_token}',
        'Content-Type': 'application/json',
    }
    payload = {
        'content': message
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        # Message sent successfully
        print("Message sent successfully.")
    else:
        # Handle errors
        print(f"Error sending message: {response.status_code}")
        print(response.json())
    return response


# -------------------------------------------------------------------------------------------------------------------

class DiscordBot(BaseTool):
    name = "discord_message"
    description = "Useful to send me or someone else a message via Discord, cannot be used to open discord app"

    def __init__(self, bot_token: str):
        bot_token = bot_token
        object.__setattr__(self, "bot_token", bot_token)
        super().__init__()

    def _run(self, tool_input: str, **kwargs) -> str:
        """Send a message to a Discord channel."""
        message = tool_input
        print(message)
        try:
            recipient = model.invoke(
                f"From the following text return 'me' if it contains the word 'me' else return the user/person it is intended for:{builtins.global_prompt}")
            # output format is: content='recipient' class 'langchain_core.messages.ai.AIMessage'
            print(recipient)

            recipient = {"discord_id": '346148004002267156'}

            print(f"recipient_id: {recipient['discord_id']}\n{tool_input}")
            input(f"recipient: {recipient}\n send message?")

            channel_id = get_dm_channel(recipient["discord_id"])  # (input("\nenter user id: "))

            if tool_input[0] == '{' and tool_input[-1] == '}':
                message = json.loads(message)
                if 'message' in message:
                    response = send_message(channel_id, message["message"])
                else:
                    for i in message:
                        response = send_message(channel_id, message[i])
            else:
                response = send_message(channel_id, message)

            if response.status_code == 200 or response.status_code == 201:
                return "Message sent successfully"
            else:
                return f"Failed to send message, status code: {response.status_code}"
        except Exception as e:
            print(e)
            return "failed to send message"


discord_messaging = DiscordBot(bot_token)
