import pickle
import faiss
import numpy as np
import os
import time
import warnings
from flask import Flask, request, jsonify

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load RAG resources (FAISS index and chunks)
print("Loading RAG resources...")
try:
    index = faiss.read_index("agri.index")
    with open("chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    print("RAG resources loaded successfully.")
except Exception as e:
    print(f"Error loading RAG resources: {e}")
    all_chunks = [] # Fallback

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Define Assistant Instructions
INSTRUCTION = """You are a helpful assistant that answers questions primarily based on the provided context. Prefer Kerala-specific agricultural practices, pest control methods, crop varieties, and recommendations whenever they are available in the context. If Kerala-specific information is present, do not use general or global practices. If the user greets (e.g., hello, hi), respond with a short greeting and ask how you can help with agriculture or farming. Summarize in 2-3 short points only. Be concise. No intro or explanation needed and use phrases if needed. If the context does not contain sufficient specific information, respond with general agricultural guidance if the question is related to agriculture or farming. Assume the end user is a farmer and prioritize clear, practical, field-ready guidance over general or theoretical explanations. If the question is not related to agriculture or farming at all, respond with "I am sorry, I don't have the information to answer that question." Do not include asterisk marks or markdown formatting."""

# Create Assistant
# Note: Ideally, you should reuse the assistant ID.
assistant = client.beta.assistants.create(
    name="Agri Assistant",
    instructions=INSTRUCTION,
    model="gpt-4o",
)
print(f"Assistant created with ID: {assistant.id}")

def retrieve_context(question, k=4):
    try:
        q_emb = embedder.encode([question])
        _, ids = index.search(np.array(q_emb), k)
        return "\n\n".join([all_chunks[i] for i in ids[0]])
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

@app.route("/ask", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    session_id = data.get("session_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Retrieve context and prepare message
    context = retrieve_context(question)
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

    try:
        run = None
        if not session_id:
            print("Starting new session...")
            run = client.beta.threads.create_and_run(
                assistant_id=assistant.id,
                thread={"messages": [{"role": "user", "content": user_content}]}
            )
            session_id = run.thread_id # Capture the new session ID
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

        # Poll for completion
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=session_id, run_id=run.id)
            if run_status.status == 'completed':
                break
            if run_status.status in ['failed', 'cancelled', 'expired']:
                return jsonify({"error": f"Run failed with status: {run_status.status}"}), 500
            time.sleep(0.5)

        # Get the latest answer
        messages = client.beta.threads.messages.list(thread_id=session_id)
        answer = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == "text":
                        answer += content.text.value
                break
        
        # Clean up response
        answer = answer.replace("*", "").strip()

        return jsonify({
            "answer": answer,
            "session_id": session_id
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)