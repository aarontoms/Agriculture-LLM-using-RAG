import pickle
import faiss
import numpy as np
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and chunks
print("Loading RAG resources...")
index = faiss.read_index("agri.index")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

# Initialize Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Assistant Instructions
instruction = """You are a helpful assistant that answers questions primarily based on the provided context. Prefer Kerala-specific agricultural practices, pest control methods, crop varieties, and recommendations whenever they are available in the context. If Kerala-specific information is present, do not use general or global practices. If the user greets (e.g., hello, hi), respond with a short greeting and ask how you can help with agriculture or farming. Summarize in 2-3 short points only. Be concise. No intro or explanation needed and use phrases if needed. If the context does not contain sufficient specific information, respond with general agricultural guidance if the question is related to agriculture or farming. Assume the end user is a farmer and prioritize clear, practical, field-ready guidance over general or theoretical explanations. If the question is not related to agriculture or farming at all, respond with "I am sorry, I don't have the information to answer that question." Do not include asterisk marks or markdown formatting."""

# Create or Retrieve Assistant
# In a production environment, you would store this ID. For this setup, we verify if we can reuse one or create new.
# To avoid creating a new assistant on every restart in development, you can hardcode the ID after first creation.
# For now, we create a new one on startup to ensure instructions are up to date.
assistant = client.beta.assistants.create(
    name="Agri Assistant",
    instructions=instruction,
    model="gpt-4o",
)
print(f"Assistant created with ID: {assistant.id}")

def retrieve(question, k=3):
    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), k)
    return [all_chunks[i] for i in ids[0]]

@app.route("/chat", methods=["POST"])
def chat():
    print("Received request")
    data = request.json
    question = data.get("question")
    session_id = data.get("session_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Manage Thread (Session)
    if not session_id:
        thread = client.beta.threads.create()
        session_id = thread.id
        print(f"Created new thread: {session_id}")
    else:
        # We assume the thread exists if ID is provided.
        # In production, wrap in try-except to handle invalid thread IDs.
        pass

    # RAG Retrieval
    context_chunks = retrieve(question, k=4)
    context = "\n\n".join(context_chunks)

    # Prepare message with context
    user_message_content = f"""Context:
{context}

Question:
{question}
"""

    # Add message to thread
    try:
        client.beta.threads.messages.create(
            thread_id=session_id,
            role="user",
            content=user_message_content
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Run Assistant
    run = client.beta.threads.runs.create(
        thread_id=session_id,
        assistant_id=assistant.id
    )

    # Poll for completion
    # A simple polling loop
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=session_id,
            run_id=run.id
        )
        if run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            return jsonify({"error": f"Run failed with status: {run_status.status}"}), 500
        time.sleep(1)

    # Retrieve Messages
    messages = client.beta.threads.messages.list(
        thread_id=session_id
    )
    
    # Get the latest assistant response
    # The default sort order is 'desc' (newest first).
    answer = ""
    for msg in messages.data:
        if msg.role == "assistant":
            for content_block in msg.content:
                if content_block.type == "text":
                    answer += content_block.text.value
            break # Get only the latest message
    
    # Format Answer (remove markdown if requested by instruction, but model should handle it)
    # The user instruction says "Do not include asterisk marks or markdown formatting."
    # The model should follow this, but we can clean up if needed.
    
    response_data = {
        "answer": answer,
        "session_id": session_id
    }

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(port=5000, debug=True)