from pypdf import PdfReader
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

all_chunks = []

pdf_files = [
    "pop2016.pdf",
    "08-crop health mangement.pdf",
    "KSBB-Final-Technical-Report.pdf",
    "MushroomProcessing.pdf",
    "a0443e04.pdf",
    "a1193e00.pdf",
    "cb3963en.pdf",
    "cc1163en.pdf",
]

for pdf in pdf_files:
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    words = text.split()
    for i in range(0, len(words), 500):
        all_chunks.append(" ".join(words[i:i+500]))

print("Number of chunks:", len(all_chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(all_chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
# print("Vectors stored:", index.ntotal)

faiss.write_index(index, "agri.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)
