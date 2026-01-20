import pickle
import faiss
import numpy as np
import os
import time
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading RAG resources...")
try:
    index = faiss.read_index("agri.index")
    with open("chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    print("RAG resources loaded successfully.")
except Exception as e:
    print(f"Error loading RAG resources: {e}")
    all_chunks = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")

import base64

INSTRUCTION = """You are a helpful and jolly agricultural assistant named AgrowBot.

Core rules:
- Answer the question using the provided context whenever possible.
- Prefer Kerala-specific agricultural practices, crop varieties, pest control methods, climate, soil, and farming recommendations when they appear in the context.
- If Kerala-specific information is present, do not replace it with general or global practices.

Relevance handling:
- Treat any question that can reasonably relate to agriculture, farming, crops, soil, climate, irrigation, pests, fertilizers, livestock, tools, weather, or rural livelihood as agricultural.
- Only classify a question as non-agricultural if it is clearly and completely unrelated (for example: movies, programming, politics, celebrities).

Refusal rule (strict and last-resort):
- Respond with exactly: "I am sorry, I don't have the information to answer that question." only if clearly unrelated.

Tone and Style:
- Be cheerful, happy, and encouraging!
- Answer in very few direct words (max 2-3 short sentences).
- No standard intros/outros.
- Do not use markdown, bullets, numbering, or asterisks.

Greeting:
- If greeted, respond happily and ask how you can help with farming today!
"""

# ... (Previous assistant creation code remains similar, updated instructions used) ...
assistant = client.beta.assistants.create(
    name="AgrowBot",
    instructions=INSTRUCTION,
    model="gpt-4o",
)
print(f"Assistant created with ID: {assistant.id}")

def generate_audio(text, voice="alloy"):
    """
    Generates audio using OpenAI TTS-1 (Standard) and returns base64 string.
    """
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        # Convert binary audio to base64 for easy transport in JSON
        audio_b64 = base64.b64encode(response.content).decode("utf-8")
        return audio_b64
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def contextualize_question(question, session_id):
    """
    If a session exists, use history to rewrite the question for better retrieval.
    """
    if not session_id:
        return question
    
    try:
        msgs = client.beta.threads.messages.list(thread_id=session_id, limit=3, order="desc")
        history_text = ""
        for msg in reversed(list(msgs.data)):
            if msg.role == "user" or msg.role == "assistant":
                content_text = ""
                for part in msg.content:
                    if part.type == "text":
                        content_text += part.text.value
                # Avoid adding the bulky Context block to the history we send for rewriting
                if "Context:" in content_text:
                     content_text = content_text.split("Context:")[0].strip()
                     if "Question:" in content_text:
                         try:
                             content_text = content_text.split("Question:")[1].strip()
                         except:
                             pass
                
                history_text += f"{msg.role}: {content_text}\n"

        if not history_text:
            return question

        # Ask LLM to rewrite
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Rewrite the last user question to be a standalone search query based on the chat history. Retain specific entities like crop names (e.g., 'mushrooms', 'tomatoes') from previous turns if the new question refers to them implicitly (e.g., 'what about light?'). Return only the rewritten question."},
                {"role": "user", "content": f"History:\n{history_text}\n\nLast Question: {question}"}
            ],
            temperature=0.3
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"Rewrote query: '{question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"Error contextualizing: {e}")
        return question


def retrieve_context(question, k=3):
    try:
        q_emb = embedder.encode([question])
        distances, ids = index.search(np.array(q_emb), k)
        
        chunks = []
        for i, dist in zip(ids[0], distances[0]):
             if dist < 1.3: 
                chunks.append(all_chunks[i])
        
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

@app.route("/ask", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    session_id = data.get("session_id")
    # Default to 'alloy' if not provided
    voice = data.get("voice", "alloy")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    search_query = contextualize_question(question, session_id)
    context = retrieve_context(search_query)
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

    try:
        run = None
        if not session_id:
            print("Starting new session...")
            run = client.beta.threads.create_and_run(
                assistant_id=assistant.id,
                thread={"messages": [{"role": "user", "content": user_content}]}
            )
            session_id = run.thread_id
        else:
            print(f"Continuing session: {session_id}")
            client.beta.threads.messages.create(
                thread_id=session_id,
                role="user",
                content=user_content
            )
            run = client.beta.threads.runs.create(
                thread_id=session_id,
                assistant_id=assistant.id
            )

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=session_id, run_id=run.id)
            if run_status.status == 'completed':
                break
            if run_status.status in ['failed', 'cancelled', 'expired']:
                return jsonify({"error": f"Run failed with status: {run_status.status}"}), 500
            time.sleep(0.5)

        messages = client.beta.threads.messages.list(thread_id=session_id)
        answer = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == "text":
                        answer += content.text.value
                break
        
        answer = answer.replace("*", "").strip()

        # Generate Audio
        audio_b64 = generate_audio(answer, voice=voice)

        return jsonify({
            "answer": answer,
            "session_id": session_id,
            "audio": audio_b64
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)