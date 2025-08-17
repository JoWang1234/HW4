import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw text from page
        # (Optional) clean page_text here (remove headers/footers)
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text

full_text = extract_text_from_pdf("./python-developer-resume-example.pdf")
from typing import List
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks
list_of_chunks = chunk_text(full_text)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(list_of_chunks) 

import faiss
import numpy as np

# Assume embeddings is a 2D numpy array of shape (num_chunks, dim)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # using a simple L2 index
index.add(np.array(embeddings))  # add all chunk vectors

# Example: search for a query embedding
query_embedding =embeddings[0].reshape(1,-1).astype("float32")  # get embedding for the query (shape: [1, dim])
k = 3
distances, indices = index.search(query_embedding, k)
# indices[0] holds the top-k chunk indices

from fastapi import FastAPI
import numpy as np
import uvicorn

app = FastAPI()

@app.get("/search")
async def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages, and return them.
    """
    # TODO: Embed the query 'q' using your embedding model
    query_vector = model.encode([q],convert_to_numpy=True).astype("float32")  # e.g., model.encode([q])[0]
    #query_vector = model.encode([q])[0]  # e.g., model.encode([q])[0]
    
    # Perform FAISS search
    k = 1
    '''distances, indices = index.search(np.array([query_vector]), k) '''
    distances, indices = index.search(query_vector, k)
    # Retrieve the corresponding chunks (assuming 'chunks' list and 'indices' shape [1, k])
    results = []
    for idx in indices[0]:
        results.append(list_of_chunks[idx])
    return {"query": q, "results": results}
    

if __name__ =='__main__':
    uvicorn.run('fasttest:app', reload = True)