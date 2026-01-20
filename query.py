import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
import os
import requests
from datetime import datetime, timedelta, timezone
from statistics import mean

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chat_history = []

index = faiss.read_index("agri.index")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(question, k=3):
    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), k)
    return [all_chunks[i] for i in ids[0]]


question = "whats the ideal climate to grow mushrooms"

context = "\n\n".join(retrieve(f"{question}", k=4))

instruction = """You are a helpful assistant that answers questions primarily based on the provided context. Prefer Kerala-specific agricultural practices, pest control methods, crop varieties, and recommendations whenever they are available in the context. If Kerala-specific information is present, do not use general or global practices. If the user greets (e.g., hello, hi), respond with a short greeting and ask how you can help with agriculture or farming. Summarize in 2-3 short points only. Be concise. No intro or explanation needed and use phrases if needed. If the context does not contain sufficient specific information, respond with general agricultural guidance if the question is related to agriculture or farming. Assume the end user is a farmer and prioritize clear, practical, field-ready guidance over general or theoretical explanations. If the question is not related to agriculture or farming at all, respond with "I am sorry, I don't have the information to answer that question." Do not include asterisk marks or markdown formatting."""

prompt = f"""
{instruction}

Context:
{context}

Question:
{question}

Answer:
"""

contents = chat_history.copy()

contents.append({
    "role": "user",
    "parts": [{"text": prompt}]
})

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config=genai.types.GenerateContentConfig(
        # max_output_tokens=256,
        # temperature=0.3,
        # top_p=0.95,
        # top_k=40,
    )
)
answer = response.text.replace("*", "").strip()

# print(context + "\n\n")
print(answer)

chat_history.append({
    "role": "user",
    "parts": [{"text": question}]
})

chat_history.append({
    "role": "model",
    "parts": [{"text": answer}]
})
chat_history = chat_history[-8:]