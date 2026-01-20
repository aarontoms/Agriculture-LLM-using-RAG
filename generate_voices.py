from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
text = "Hi! I'm AgrowBot. I'm happy to help you!"

print("Generating voice previews...")

if not os.path.exists("./static/voices"):
    os.makedirs("./static/voices")

for voice in voices:
    print(f"Generating {voice}...")
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    with open(f"./static/voices/{voice}.mp3", "wb") as f:
        f.write(response.content)

print("All voice previews generated in static/voices/")
