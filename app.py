import os, warnings, logging

# 1. Suppress all terminal bloat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import pickle
import faiss
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torchaudio
import torch
from speechbrain.inference import SpeakerRecognition
import tempfile
import shutil
import base64
import threading

load_dotenv()

print("\n" + "="*40)
print("  🌱 AgrowBot Backend Initializing...  ")
print("="*40)

print("Loading Speaker Verification Model...")
try:
    verification_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )
    print("Speaker Model Loaded.")
except Exception as e:
    print(f"Error loading Speaker Model: {e}")
    verification_model = None

# Store speaker raw waveforms: { session_id: waveform_tensor }
session_speakers = {}

# Store conversation history locally: { session_id: [ {"role": "user", "content": "..."}, ... ] }
conversation_history = {}

MODEL_NAME = "gpt-4o-mini"

def process_verification(session_id, audio_b64):
    """
    Returns:
        is_same_user (bool): True if match or uncertain, False if definitely new user
        new_session_id (str|None): If mismatch, session_id is None (reset).
        current_wave (tensor|None): The processed waveform.
        duration (float): Duration of audio.
    """
    if not verification_model or not audio_b64:
        return True, session_id, None, 0

    try:
        # 1. Save Base64 to Temp WAV in current directory to avoid path issues
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=".") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        # 2. Check Duration & Load
        signal, fs = torchaudio.load(temp_audio_path)
        
        # Force Mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        # Ensure 16kHz
        if fs != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
            fs = 16000

        # --- WINDOWS DEEP CLEAN ---
        # Convert to numpy and back to a fresh tensor to strip weird strides/metadata
        sig_np = signal.detach().cpu().numpy()
        signal = torch.from_numpy(sig_np).to(torch.float32)
        
        # Enforce strict [1, Time] shape as required by SpeakerRecognition
        # signal is likely [1, T] after mono conversion, but we ensure it here.
        if signal.ndim == 2:
            pass  # already correct [1, T] or [C, T] where C=1
        elif signal.ndim == 1:
            signal = signal.unsqueeze(0)
        else:
            raise ValueError(f"Invalid signal shape: {signal.shape}")

        duration = signal.shape[1] / fs
        if duration < 0.5:
             print(f"Audio too short ({duration:.2f}s), skipping verification.")
             return True, session_id, None, duration

        print(f"Audio Duration: {duration:.2f}s | Shape: {list(signal.shape)}")

        # 3. Verification Logic
        is_same = True
        verification_model.eval()
        
        if session_id in session_speakers:
            # We have a reference waveform. Compare using raw waveforms.
            ref_wave = session_speakers[session_id]
            with torch.no_grad():
                # Correct way for SpeakerRecognition: pass waveforms, let it handle encoding
                score, prediction = verification_model.verify_batch(ref_wave, signal)
            
            print(f"Verification Score: {score[0].item():.4f}, Prediction: {prediction[0].item()}")
            
            if not prediction[0]:
                print(f"User Mismatch detected! Resetting session.")
                is_same = False
                session_id = None # Reset session
            else:
                print("User verified.")
        
        # Note: We return the waveform itself to be stored if it's a new session
        return is_same, session_id, signal, duration

    except Exception as e:
        print(f"Verification Check Error: {e}")
        return True, session_id, None, 0
    finally:
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

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

# Ensure static/audio exists for Async TTS
AUDIO_DIR = os.path.join("static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

def generate_audio_file(text, filename, voice="alloy"):
    """
    Generates audio in background and saves to static/audio/filename.
    """
    try:
        start = time.time()
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        filepath = os.path.join(AUDIO_DIR, filename)
        response.stream_to_file(filepath)
        print(f"TTS Generated: {filename} in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"TTS Error: {e}")

INSTRUCTION = """You are a helpful and jolly agricultural assistant named AgrowBot.

Core rules:
- Answer the question using the provided context whenever possible.
- Prefer Kerala-specific agricultural practices, crop varieties, pest control methods, climate, soil, and farming recommendations when they appear in the context.
- If Kerala-specific information is present, do not replace it with general or global practices.
- If context is not present, answer it normally but do not acknowledge the lack of context.

Relevance handling:
- Treat any question that can reasonably relate to agriculture, farming, crops, soil, climate, irrigation, pests, fertilizers, livestock, tools, weather, or rural livelihood as agricultural.
- Only classify a question as non-agricultural if it is clearly and completely unrelated (for example: movies, programming, politics, celebrities).

Refusal rule (strict and last-resort):
- Respond with exactly: "I am sorry, I don't have the information to answer that question." only if clearly unrelated.

Tone and Style:
- Be cheerful, happy, and encouraging!
- Answer in very few direct words (max 2-3 short sentences).
- Respond using direct, meaningful phrases and move straight to the answer with no introductions or filler.
- No standard intros/outros.
- Do not use markdown, bullets, numbering, or asterisks.

Greeting:
- If greeted, respond happily and ask how you can help with farming today!
"""

assistant = client.beta.assistants.create(
    name="AgrowBot",
    instructions=INSTRUCTION,
    model="gpt-4o",
)
print(f"Assistant created with ID: {assistant.id}")

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

# Store last mentioned entity for each session to improve RAG: { session_id: "mushroom" }
last_active_entities = {}

def extract_and_track_entity(text, session_id):
    """
    Simple keyword extraction to track the current subject (e.g., 'mushroom', 'tomato').
    This avoids using an LLM to rewrite queries.
    """
    # Common agricultural entities in Kerala/General context
    keywords = [
        "mushroom", "paddy", "rice", "banana", "coconut", "arecanut", 
        "pepper", "cardamom", "ginger", "turmeric", "rubber", "tea", "coffee",
        "vegetable", "tomato", "chilli", "brinjal", "cow", "goat", "chicken",
        "fertilizer", "manure", "soil", "irrigation", "corn", "maize", "wheat",
        "sugarcane", "cassava", "yam", "pulses", "pest", "disease",
    ]
    
    text_lower = text.lower()
    for word in keywords:
        if word in text_lower:
            last_active_entities[session_id] = word
            return
    # If no new entity found, we keep the previous one in the global dict

@app.route("/ask", methods=["POST"])
def chat():
    request_start = time.time()
    data = request.json
    question = data.get("question")
    session_id = data.get("session_id")
    audio_b64 = data.get("audio") 
    voice = data.get("voice", "alloy")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # 1. Initialize Thread (Assistants API History)
    if not session_id:
        print("Starting new session (Thread)...")
        # Create a new Thread with the initial system message logic handled by Assistant Instructions
        thread = client.beta.threads.create()
        session_id = thread.id
    else:
        print(f"Continuing session: {session_id}")

    # 2. Speaker Verification (Enabled)
    session_reset = False
    if audio_b64:
        is_same_user, ver_session_id, current_wave, current_duration = process_verification(session_id, audio_b64)
        if not is_same_user:
            print("User mismatch! Resetting history (New Thread).")
            # Create a brand new thread for the new user
            thread = client.beta.threads.create()
            session_id = thread.id
            session_reset = True
            # Reset active entity as well
            if session_id in last_active_entities:
                del last_active_entities[session_id]
        
        # Store Reference 
        if session_id and session_id not in session_speakers and current_wave is not None:
             if current_duration > 1.5:
                 print(f"Locking new speaker reference for session {session_id}")
                 session_speakers[session_id] = current_wave

    # 2. Fast Context Injection
    extract_and_track_entity(question, session_id)
    search_query = question
    if session_id in last_active_entities:
        entity = last_active_entities[session_id]
        if entity not in question.lower():
            search_query = f"{entity} {question}"
            print(f"Context injected: '{question}' -> '{search_query}'")

    # 3. Retrieve Context
    context_start = time.time()
    context = retrieve_context(search_query)
    print(f"Retrieval took: {time.time() - context_start:.2f}s")

    # 4. Add User Message to Thread
    # We inject context into this message
    user_message_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    
    try:
        client.beta.threads.messages.create(
            thread_id=session_id,
            role="user",
            content=user_message_content
        )

        # 5. Run Assistant
        llm_start = time.time()
        run = client.beta.threads.runs.create(
            thread_id=session_id,
            assistant_id=assistant.id
        )

        # Polling Loop
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=session_id, run_id=run.id)
            if run_status.status == 'completed':
                break
            if run_status.status in ['failed', 'cancelled', 'expired']:
                return jsonify({"error": f"Run failed with status: {run_status.status}"}), 500
            time.sleep(0.5)

        # 6. Retrieve Answer
        messages = client.beta.threads.messages.list(thread_id=session_id)
        answer = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == "text":
                        answer += content.text.value
                break
        
        answer = answer.replace("*", "").strip()
        print(f"LLM (Assistant) took: {time.time() - llm_start:.2f}s")
        
        # 7. Async TTS Generation
        # Generate a unique filename
        filename = f"{session_id}_{int(time.time())}.mp3"
        audio_url = f"/static/audio/{filename}"
        
        # Start thread
        thread = threading.Thread(target=generate_audio_file, args=(answer, filename, voice))
        thread.start()

        total_time = time.time() - request_start
        print(f"Total Request Latency: {total_time:.2f}s (Text Only)")

        return jsonify({
            "answer": answer,
            "session_id": session_id,
            "audio_url": audio_url,
            "session_reset": session_reset
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)