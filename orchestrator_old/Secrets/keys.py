import dotenv
import os

dotenv.load_dotenv()

bot_token = os.environ['DISCORD_KEY']
google_api = os.environ['GOOGLE_API_KEY']
wolfram_app_id = os.environ['WOLFRAM_ID']

vector_store_url = "http://localhost:8000/query/"
user = "BiscuitBobby"
