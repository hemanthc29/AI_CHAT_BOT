from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

import chromadb
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS - allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables AFTER app creation
load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_CLOUD_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_CLOUD_TENANT")
CHROMA_DB = "Kinematics"
CHROMA_COLLECTION = "Kinematics"

if not PERPLEXITY_API_KEY or not CHROMA_API_KEY or not CHROMA_TENANT:
    raise RuntimeError("❌ One or more required API keys/tenant missing in environment variables.")

# Initialize ChromaDB client and collection
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DB,
)

# Attempt to get or create collection
try:
    knowledge_collection = client.get_collection(CHROMA_COLLECTION)
except Exception as e:
    raise RuntimeError(f"❌ Failed to access ChromaDB collection '{CHROMA_COLLECTION}': {e}")

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3

class QueryResponse(BaseModel):
    thinking: str
    answer: str

def query_chroma_collection(query: str, n_results: int = 3) -> str:
    """
    Query the ChromaDB collection and return combined context text.
    """
    results = knowledge_collection.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    combined_context = "\n\n---\n\n".join(docs) if docs else ""
    return combined_context

def call_perplexity_api(query: str, context: str) -> str:
    """
    Call the Perplexity AI chat completions API with constructed prompt.
    """
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
You are an expert IIT JEE Physics tutor that answers questions with detailed explanations. Using the following context, answer the user's question.

Context:<br><br>
{context}<br><br>

User Question:<br><br>
{query}<br><br>

Provide a clear, structured, and detailed step-by-step explanation in your answer using HTML tags such as:<br>
<ul>
  <li><b>&lt;p&gt;</b> for paragraphs</li>
  <li><b>&lt;b&gt;</b> or <b>&lt;strong&gt;</b> for bold text</li>
  <li><b>&lt;i&gt;</b> or <b>&lt;em&gt;</b> for emphasis</li>
  <li><b>&lt;ul&gt;</b> and <b>&lt;li&gt;</b> for lists</li>
  <li><b>&lt;h3&gt;</b>, <b>&lt;h4&gt;</b> for headings</li>
</ul>
Do NOT use markdown syntax like **bold** or --- lines.<br><br>

Please provide the entire answer in valid HTML.
"""

    payload = {
        "model": "sonar",  # Make sure this model name is valid per Perplexity's documentation
        "messages": [
            {"role": "system", "content": "You are precise, concise and educational."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Perplexity API error: {response.status_code} {response.text}")

@app.post("/api/query", response_model=QueryResponse)
async def query_jee(request: QueryRequest):
    try:
        # Get context from ChromaDB
        context = query_chroma_collection(request.query, request.n_results)
        if not context.strip():
            return QueryResponse(
                thinking="",
                answer="Sorry, no relevant information found in the knowledge base for your query."
            )

        response_text = call_perplexity_api(request.query, context)

        # Optionally, parse thinking and answer parts if your prompt supports it
        return QueryResponse(thinking="", answer=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
