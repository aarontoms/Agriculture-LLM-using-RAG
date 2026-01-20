import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

index = faiss.read_index("agri.index")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(question, k=3):
    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), k)
    return [all_chunks[i] for i in ids[0]]

print("API Key loaded:", os.getenv("GROQ_API_KEY") is not None)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

question = "How to control pests in rice?"

context = "\n\n".join(retrieve(question))  # Your RAG retrieval function stays the same

instruction = """You are a helpful assistant that answers questions primarily based on the provided context. Prefer Kerala-specific agricultural practices, pest control methods, crop varieties, and recommendations whenever they are available in the context. If Kerala-specific information is present, do not use general or global practices. If the user greets (e.g., hello, hi), respond with a short greeting and ask how you can help with agriculture or farming. Summarize in 2-3 short points only. Be concise. No intro or explanation needed and use phrases if needed. If the context does not contain sufficient specific information, respond with general agricultural guidance if the question is related to agriculture or farming. Assume the end user is a farmer and prioritize clear, practical, field-ready guidance over general or theoretical explanations. If the question is not related to agriculture or farming at all, respond with "I am sorry, I don't have the information to answer that question." Do not include asterisk marks or markdown formatting."""

messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"}
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Strong & capable (similar level to Gemini Pro); very fast on Groq
    # Alternatives: "llama-3.1-8b-instant" (faster/cheaper), "llama-3.1-70b-versatile" if available
    messages=messages,
    max_tokens=256,      # Enforces short responses (~150-200 words max)
    temperature=0.3,     # Low for consistent, focused output
    top_p=0.95,
)

response_text = response.choices[0].message.content.strip()
# No need for .replace("*", "") since we instructed no markdown/asterisks

print(response_text)