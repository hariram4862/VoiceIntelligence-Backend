from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
from typing import List, Optional
import pytz
import whisper
import shutil
import os
import requests
import google.generativeai as genai
from docx import Document
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base")  # Whisper for transcription

# MongoDB Setup
MONGO_URI = "mongodb+srv://hari123:hari123@cluster0.oxgsmkv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URI)

db = client.legal_assistant
users_collection = db.users
sessions_collection = db.sessions
files_collection = db.files

IST = pytz.timezone("Asia/Kolkata")

# --- Gemini setup ---
GEMINI_API_KEY = "AIzaSyA_ro-5MnFra6vuSJufp7LHiD4Tl0FslpQ"  # Replace this
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Qdrant & Embedding setup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_file(path, ext):
    try:
        if ext == '.pdf':
            doc = fitz.open(path)
            text = "".join([page.get_text() for page in doc])
            doc.close()
        elif ext == '.docx':
            document = Document(path)
            text = "\n".join(p.text for p in document.paragraphs)
        elif ext in ['.png', '.jpg', '.jpeg']:
            text = pytesseract.image_to_string(Image.open(path))
        elif ext == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            return None
        return text.strip()
    except:
        return None

def split_text(text, max_length=500, overlap=50):
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-(len(current_chunk)//2):]
            current_length = sum(len(w) for w in current_chunk)
        current_chunk.append(word)
        current_length += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

async def embed_and_store(file: UploadFile, email: str, session_id: str):
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_file(temp_path, ext)
    os.remove(temp_path)
    if not text:
        raise ValueError("No text extracted from file.")

    chunks = split_text(text)
    vectors = embed_model.encode(chunks, show_progress_bar=False)

    chunk_data = []
    for i in range(len(chunks)):
        chunk_data.append({
            "index": i,
            "text": chunks[i],
            "vector": vectors[i].tolist()
        })

    doc = {
        "email": email,
        "session_id": session_id,
        "filename": file.filename,
        "chunks": chunk_data
    }

    await files_collection.insert_one(doc)
    return len(chunks)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

@app.post("/query_or_upload")
async def query_or_upload(
    files: Optional[List[UploadFile]] = File(None),
    prompt: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    results = []

    if files:
        for file in files:
            try:
                chunk_count = await embed_and_store(file, email, session_id)
                results.append({"filename": file.filename, "chunks": chunk_count})
            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})

    if not prompt:
        return {"message": "✅ Documents processed.", "upload_results": results}

    query_vec = embed_model.encode([prompt])[0].tolist()

    documents = await files_collection.find({
        "email": email,
        "session_id": session_id
    }).to_list(length=100)

    similarities = []
    for doc in documents:
        for chunk in doc.get("chunks", []):
            sim = cosine_similarity(np.array(chunk["vector"]), np.array(query_vec))
            similarities.append({
                "text": chunk["text"],
                "score": sim
            })

    top_chunks = sorted(similarities, key=lambda x: x["score"], reverse=True)[:5]
    relevant_chunks = [c["text"] for c in top_chunks]

    if not relevant_chunks:
        return {
            "message": "⚠️ No relevant information found in session documents.",
            "upload_results": results,
            "response": "No relevant information found."
        }

    context = "\n\n".join(relevant_chunks[:3])
    final_prompt = f"""You are a legal assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{prompt}

Answer:"""

    payload = {"contents": [{"parts": [{"text": final_prompt}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    result = response.json()

    if "candidates" in result:
        answer = result["candidates"][0]["content"]["parts"][0].get("text", "")
        return {
            "message": "✅ Documents processed and response generated.",
            "upload_results": results,
            "response": answer.strip()
        }

    return {
        "message": "⚠️ Gemini failed to respond.",
        "upload_results": results,
        "response": "Gemini failed to generate a valid response."
    }




@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open("temp_audio.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model.transcribe("temp_audio.wav")
    os.remove("temp_audio.wav")
    return {"text": result.get("text", ""), "raw": result}


# @app.post("/upload_files")
# async def upload_files(files: List[UploadFile] = File(...)):
#     for file in files:
#         print(f"✅ Got the file: {file.filename}")
#     return {"message": f"✅ Received {len(files)} files successfully"}

# @app.post("/respond")
# async def respond(prompt: str = Form(...)):
#     try:
#         payload = {
#             "contents": [
#                 {
#                     "parts": [
#                         {"text": prompt}
#                     ]
#                 }
#             ]
#         }

#         headers = {"Content-Type": "application/json"}

#         response = requests.post(GEMINI_URL, headers=headers, json=payload)
#         result = response.json()

#         if "candidates" in result:
#             text = result["candidates"][0]["content"]["parts"][0].get("text", "")
#             return {"response": text}
#         else:
#             return {"response": f"❌ Gemini error: {result}"}

#     except Exception as e:
#         return {"response": f"❌ Exception: {str(e)}"}





# ========== ENDPOINTS ==========

@app.post("/register_user")
async def register_user(
    email: str = Form(...),
    full_name: str = Form(...),
    joined_at: str = Form(...) 
):
    existing = await users_collection.find_one({"_id": email})
    if existing:
        return {"message": "✅ User already exists."}

    new_user = {
        "_id": email,
        "full_name": full_name,
        "joined_at": joined_at,
        "last_login_at": joined_at

    }
    await users_collection.insert_one(new_user)
    return {"message": "✅ User registered", "user": new_user}

@app.post("/update_last_login")
async def update_last_login(
    email: str = Form(...),
    last_login_at: str = Form(...)
):
    result = await users_collection.update_one(
        {"_id": email},
        {"$set": {"last_login_at": last_login_at}}
    )
    
    if result.modified_count == 1:
        return {"message": "✅ Last login updated"}
    else:
        return {"message": "⚠️ User not found or already up-to-date"}


@app.post("/update_user_details/")
async def update_user_details(
    email: str = Form(...),
    full_name: str = Form(...)
):
    result = await users_collection.update_one(
        {"_id": email},
        {"$set": {"full_name": full_name}}
    )
    
    if result.modified_count == 1:
        return {"message": "✅ User name updated"}
    else:
        return {"message": "⚠️ User not found or already up-to-date"}



@app.get("/get_user_details/{email}")
async def get_user_details(email: str):
    user = await users_collection.find_one({"_id": email}, {"_id": 1, "full_name": 1})
    if not user:
        return {"message": "❌ User not found."}
    return {
        "message": "✅ User found.",
        "email": user["_id"],
        "full_name": user["full_name"]
    }

# Auto-generate a session name from prompt/response
def generate_session_name(prompt: str, response: str) -> str:
    
    try:
        instruction = "Suggest (only one suggestion...apt one..no other texts...not in bold format) a chat session name (3 to 4 words only) for the following prompt:\n"
        full_prompt = instruction + prompt

        payload = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ]
        }

        headers = {"Content-Type": "application/json"}

        gemini_resp = requests.post(GEMINI_URL, headers=headers, json=payload)
        gemini_data = gemini_resp.json()

        if "candidates" in gemini_data:
            session_name = gemini_data["candidates"][0]["content"]["parts"][0].get("text", "")
            return session_name.strip().title()  # Return Gemini's suggestion in title case

    except Exception as e:
        print(f"Gemini error: {e}")
   
    for phrase in [prompt, response]:
        for keyword in ["IPC", "Section", "Act", "law", "case", "rule", "rights", "punishment"]:
            if keyword.lower() in phrase.lower():
                return f"{keyword.upper()} Discussion"
    return "Legal Chat Session"


@app.post("/add_message")
async def add_message(
    email: str = Form(...),
    session_id: str = Form(None),  # Optional
    prompt: str = Form(...),
    response: str = Form(...)
):
    message = {
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.now(IST)
    }

    # If session_id is not provided, create a new session
    if not session_id:
        session_name = generate_session_name(prompt, response)
        session_data = {
            "user_email": email,
            "session_name": session_name,
            "created_at": datetime.now(IST),
            "messages": [message]
        }
        result = await sessions_collection.insert_one(session_data)
        return {
            "message": "✅ New session created with message",
            "session_id": str(result.inserted_id),
            "session_name": session_name
        }

    # If session_id provided, append message
    result = await sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$push": {"messages": message}}
    )

    if result.modified_count == 1:
        return {"message": "✅ Message added to session"}
    else:
        raise HTTPException(status_code=404, detail="❌ Session not found")


@app.get("/get_sessions/{email}")
async def get_sessions(email: str):
    sessions = await sessions_collection.find({"user_email": email}).to_list(length=100)
    for session in sessions:
        session["_id"] = str(session["_id"])
    return {"sessions": sessions}


@app.get("/get_session_chat/{session_id}")
async def get_session_chat(session_id: str):
    session = await sessions_collection.find_one({"_id": ObjectId(session_id)})
    if not session:
        raise HTTPException(status_code=404, detail="❌ Session not found")
    session["_id"] = str(session["_id"])
    return session
