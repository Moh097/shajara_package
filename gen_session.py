# gen_session.py
# Small helper to generate a TELEGRAM_STRING_SESSION locally using Telethon.
# Usage: python gen_session.py
from telethon.sync import TelegramClient
from telethon.sessions import StringSession

api_id = int(input("API_ID: ").strip())
api_hash = input("API_HASH: ").strip()

with TelegramClient(StringSession(), api_id, api_hash) as client:
    print("String session (copy this exact line):")
    print(client.session.save())
