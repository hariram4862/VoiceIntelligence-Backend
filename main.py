# # main.py
# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def root():
#     return {"message": "Hello,  World!"}


from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime, timezone
from fastapi import Body
from typing import Any, Dict, List, Optional
import pytz
import whisper
import shutil
from fastapi import Query
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel



load_dotenv()  
import requests
import google.generativeai as genai
from docx import Document
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
app = FastAPI()
from azure.storage.blob import BlobServiceClient

AZURE_CONNECTION_STRING =os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")



blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)

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
MONGO_URI = os.getenv("MONGO_URI")
# MongoDB Setup
from datetime import timezone
client = AsyncIOMotorClient(MONGO_URI, tz_aware=True, tzinfo=timezone.utc)


db = client.voice_intelligence
users_collection = db.users
sessions_collection = db.sessions
files_collection = db.files
shared_collection = db.shared_sessions
quiz_results_collection = db.quiz_results
import secrets
import random

IST = pytz.timezone("Asia/Kolkata")

# --- Gemini setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace this
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Qdrant & Embedding setup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:           # Mongo may give naive but UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


# ---------- Utils ----------
def extract_text_from_file(path: str, ext: str) -> Optional[str]:
    try:
        if ext == ".pdf":
            doc = fitz.open(path)
            text = "".join([p.get_text() for p in doc])
            doc.close()
        elif ext == ".docx":
            document = Document(path)
            text = "\n".join(p.text for p in document.paragraphs)
        elif ext in [".png", ".jpg", ".jpeg"]:
            text = pytesseract.image_to_string(Image.open(path))
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return None
        return text.strip()
    except:
        return None

def split_text(text: str, max_length: int = 500) -> List[str]:
    words, chunks, cur = text.split(), [], []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_length and cur:
            chunks.append(" ".join(cur))
            keep = max(1, int(len(cur) * 0.4))  # ~40% overlap
            cur = cur[-keep:]
            cur_len = sum(len(x) for x in cur) + len(cur) - 1
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

async def upload_to_blob(filename: str, content: bytes) -> str:
    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(content, overwrite=True)
    return blob_client.url

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

async def embed_and_store_bytes(filename: str, content: bytes, email: str, session_id: str, blob_url: str) -> int:
    ext = os.path.splitext(filename)[1].lower()
    tmp = f"tmp_{filename}"
    with open(tmp, "wb") as f:
        f.write(content)

    text = extract_text_from_file(tmp, ext)
    os.remove(tmp)
    if not text:
        raise ValueError("No text extracted from file.")

    chunks = split_text(text)
    vectors = embed_model.encode(chunks, show_progress_bar=False)
    chunk_data = [{"index": i, "text": chunks[i], "vector": vectors[i].tolist()} for i in range(len(chunks))]

    doc = {
        "email": email,
        "session_id": session_id,
        "filename": filename,
        "blob_url": blob_url,  # ✅ Store Azure Blob URL for download
        "chunks": chunk_data,
        "created_at": datetime.now(timezone.utc),
    }
    await files_collection.insert_one(doc)
    return len(chunks)


async def ensure_session(email: str, prompt: str, response: str, session_id: Optional[str]) -> Dict[str, Any]:
    if not session_id:
        session_name = await generate_session_name_async(prompt, response)
        session_data = {
            "user_email": email,
            "session_name": session_name,
            "created_at": datetime.now(timezone.utc),
            "messages": [{"prompt": prompt, "response": response, "timestamp": datetime.now(IST)}],
        }
        result = await sessions_collection.insert_one(session_data)
        return {"session_id": str(result.inserted_id), "session_name": session_name}
    else:
        await sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$push": {"messages": {"prompt": prompt, "response": response, "timestamp": datetime.now(IST)}}}
        )
        ses = await sessions_collection.find_one({"_id": ObjectId(session_id)}, {"session_name": 1})
        return {"session_id": session_id, "session_name": ses.get("session_name", "Untitled") if ses else "Untitled"}

async def top_k_chunks(email: str, session_id: str, query: str, k: int = 5) -> List[str]:
    query_vec = embed_model.encode([query])[0]
    docs = await files_collection.find({"email": email, "session_id": session_id}).to_list(length=1000)
    scored = []
    for d in docs:
        for ch in d.get("chunks", []):
            sim = cosine_similarity(np.array(ch["vector"]), np.array(query_vec))
            scored.append((sim, ch["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]

def build_prompt_with_context(context: List[str], user_q: Optional[str]) -> str:
    ctx = "\n\n".join(context[:3]) if context else ""
    if user_q:
        if ctx:
            return f"""You are a helpful assistant. Use ONLY the following context to answer.

Context:
{ctx}

Question:
{user_q}

Answer (be concise, bullet where helpful):"""
        else:
            return f"""You are a helpful assistant. Answer the following:

Question:
{user_q}

Answer (be concise):"""
    else:
        return f"""You are a helpful assistant. Summarize the following context into key points and action items.

Context:
{ctx}

Answer:"""

def call_gemini(prompt_text: str) -> str:
    try:
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY,   # << correct place for API key
        }
        r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=60)
        j = r.json()

        if "error" in j:
            return f"Gemini API error: {j['error'].get('message', 'Unknown error')}"

        if "candidates" in j and j["candidates"]:
            return j["candidates"][0]["content"]["parts"][0].get("text", "").strip()

        return "Gemini returned empty response."
    except Exception as e:
        return f"Gemini error: {e}"


async def generate_session_name_async(prompt: str, response: str) -> str:
    try:
        instruction = "Suggest (only one; 3-4 words; no extra text) a chat session name for the following prompt:\n"
        name = call_gemini(instruction + prompt)
        name = name.strip().title() if name else "Chat Session"
        return name[:50] or "Chat Session"
    except:
        return "Chat Session"

# ---------- New split routes ----------

@app.post("/v1/prompt-only")
async def route_prompt_only(
    prompt: str = Form(...),
    email: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    context: List[str] = []
    if session_id:
        # Use previously uploaded chunks in this session
        context = await top_k_chunks(email, session_id, query=prompt, k=7)

    final_prompt = build_prompt_with_context(context, user_q=prompt)
    answer = call_gemini(final_prompt)
    ses = await ensure_session(email=email, prompt=prompt, response=answer, session_id=session_id)
    return {"response": answer, **ses}

@app.post("/v1/files-only")
async def route_files_only(
    files: List[UploadFile] = File(...),
    email: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    if not session_id:
        tmp = await ensure_session(email, "Uploaded files", "Processing…", None)
        session_id = tmp["session_id"]

    results = []
    for f in files:
        try:
            content = await f.read()
            await f.seek(0)
            blob_url = await upload_to_blob(f.filename, content)
            chunk_count = await embed_and_store_bytes(f.filename, content, email, session_id, blob_url)
            results.append({"filename": f.filename, "chunks": chunk_count, "blob_url": blob_url})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})



    context = []
    if results and not all(("error" in r) for r in results):
        context = await top_k_chunks(email, session_id, query="summarize the uploaded documents", k=7)

    final_prompt = build_prompt_with_context(context, user_q=None)
    answer = call_gemini(final_prompt) if context else "Files uploaded and processed."

    fake_prompt = "Uploaded files: " + ", ".join([r["filename"] for r in results if "filename" in r])
    await ensure_session(email=email, prompt=fake_prompt, response=answer, session_id=session_id)

    return {
        "message": "✅ Documents processed.",
        "upload_results": results,
        "response": answer,
        "session_id": session_id
    }

@app.post("/v1/files-plus-prompt")
async def route_files_plus_prompt(
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    email: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    if not session_id:
        tmp = await ensure_session(email, prompt, "Processing…", None)
        session_id = tmp["session_id"]

    results = []
    for f in files:
        try:
            content = await f.read()
            await f.seek(0)
            blob_url = await upload_to_blob(f.filename, content)
            chunk_count = await embed_and_store_bytes(f.filename, content, email, session_id, blob_url)
            results.append({"filename": f.filename, "chunks": chunk_count, "blob_url": blob_url})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})



    context = await top_k_chunks(email, session_id, query=prompt, k=7)
    final_prompt = build_prompt_with_context(context, user_q=prompt)
    answer = call_gemini(final_prompt)

    ses = await ensure_session(email=email, prompt=prompt, response=answer, session_id=session_id)
    return {
        "message": "✅ Documents processed and response generated.",
        "upload_results": results,
        "response": answer,
        **ses
    }



@app.get("/")
async def root():
    return {"message": "FastAPI on Azure App Service!"}

@app.post("/rename_session")
async def rename_session(
    session_id: str = Body(...),
    new_name: str = Body(...)
):
    result = await sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"session_name": new_name}}
    )

    if result.modified_count == 1:
        return {"message": "✅ Session renamed successfully"}
    else:
        raise HTTPException(status_code=404, detail="❌ Session not found or already renamed")

@app.delete("/delete_session/{id}")
async def delete_session(id: str):
    session_result = await sessions_collection.delete_one({"_id": ObjectId(id)})
    files_result = await files_collection.delete_many({"session_id": id})

    if session_result.deleted_count == 1:
        return {
            "message": "✅ Session deleted successfully",
            "deleted_files": files_result.deleted_count
        }
    else:
        raise HTTPException(status_code=404, detail="❌ Session not found")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open("temp_audio.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model.transcribe("temp_audio.wav")
    os.remove("temp_audio.wav")
    return {"text": result.get("text", ""), "raw": result}



def generate_share_id(length=8):
    return ''.join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(length))

def generate_pin():
    return str(random.randint(1000, 9999))

@app.post("/share_session/{session_id}")
async def share_session(session_id: str):
    session = await sessions_collection.find_one({"_id": ObjectId(session_id)})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    share_id = generate_share_id()
    pin = generate_pin()

    await shared_collection.insert_one({
        "share_id": share_id,
        "pin": pin,
        "session_id": ObjectId(session_id),
        "created_at": datetime.now(timezone.utc),
    })

    return {"share_id": share_id, "pin": pin}



@app.get("/view_shared_session/{share_id}")
async def view_shared_session(share_id: str, pin: str = Query(...)):
    shared = await shared_collection.find_one({"share_id": share_id, "pin": pin})
    if not shared:
        raise HTTPException(status_code=403, detail="Invalid share ID or PIN")

    session = await sessions_collection.find_one({"_id": shared["session_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    formatted_messages = []
    for msg in session.get("messages", []):
        formatted_messages.append({"role": "user", "text": msg["prompt"]})
        formatted_messages.append({"role": "bot", "text": msg["response"]})

    return {
        "session_name": session.get("session_name", "Untitled"),
        "messages": formatted_messages
    }


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
    return "Chat Session"


@app.post("/add_message")
async def add_message(
    email: str = Form(...),
    session_id: str = Form(None),  # Optional
    prompt: str = Form(...),
    response: str = Form(...)
):
    now_utc = datetime.now(timezone.utc)
    message = {
        "prompt": prompt, 
        "response": response,
        "created_at": now_utc
    }

    # If session_id is not provided, create a new session
    if not session_id:
        session_name = generate_session_name(prompt, response)
        session_data = {
            "user_email": email,
            "session_name": session_name,
            "created_at": now_utc,
            "messages": [message]
        }
        result = await sessions_collection.insert_one(session_data)
        return {
            "message": "✅ New session created with message",
            "session_id": str(result.inserted_id),
            "session_name": session_name,
            "created_at":iso_utc(now_utc),
        }

    # If session_id provided, append message
    result = await sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$push": {"messages": message}}
    )

    if result.modified_count == 1:
        return {"message": "✅ Message added to session", "created_at": iso_utc(now_utc) }
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

# ---------- Helpers used by quiz endpoints ----------
def safe_json_parse(text: str):
    if not text:
        return None
    try:
        # Remove markdown code fences like ```json ... ```
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove optional "json" after ```
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        # Try parsing
        return json.loads(cleaned)
    except Exception as e:
        print("Parse error:", e)
        return None

# ---------- Quiz Routes ----------
class GenerateQuizRequest(BaseModel):
    prompt: Optional[str] = None
    email: str
    mode: str  # mcq | flashcard | teachback
    session_id: Optional[str] = None

@app.post("/generate_quiz")
async def generate_quiz(
    prompt: Optional[str] = Form(None),
    email: str = Form(...),
    mode: str = Form(...),  # "mcq","flashcard","teachback"
    session_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    # If files present, process & embed them (re-using existing code)
    if files:
        if not session_id:
            tmp = await ensure_session(email, "Uploaded files (for quiz)", "Processing…", None)
            session_id = tmp["session_id"]

        upload_results = []
        for f in files:
            try:
                content = await f.read()
                await f.seek(0)
                blob_url = await upload_to_blob(f.filename, content)
                chunk_count = await embed_and_store_bytes(f.filename, content, email, session_id, blob_url)
                upload_results.append({"filename": f.filename, "chunks": chunk_count, "blob_url": blob_url})
            except Exception as e:
                upload_results.append({"filename": f.filename, "error": str(e)})
    else:
        upload_results = []

    # Build context using top_k_chunks if we have a session_id
    context = []
    if session_id:
        context = await top_k_chunks(email, session_id, query=prompt or "learn", k=7)

    # Build instructions for LLM based on mode
    if mode.lower() == "mcq":
        instr = (
            "You are an assistant that *only* outputs JSON. Create 5 multiple-choice questions (or fewer if content small) "
            "based ONLY on the context and optional prompt. Return JSON array: "
            '[{"id":"q1","question":"...","options":["A","B","C","D"],"answer_index":0,"explanation":"..."}]\n\n'
        )
    elif mode.lower() == "flashcard":
        instr = (
            "You are an assistant that *only* outputs JSON. Create up to 10 flashcards based ONLY on the context/prompt. "
            'Return JSON array: [{"id":"f1","front":"...", "back":"..."}]\n\n'
        )
    elif mode.lower() == "teachback":
        instr = (
            "You are an assistant that *only* outputs JSON. Produce:\n"
            "{'lesson': 'short explanation (2-3 paragraphs)', 'prompts':[{'id':'t1','prompt':'...','rubric':['keyword1','keyword2']}]} \n"
            "Create up to 3 prompts for student to 'teach back'.\n\n"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    final_prompt = instr
    if context:
        final_prompt += "Context:\n" + "\n\n".join(context[:7]) + "\n\n"
    if prompt:
        final_prompt += "Focus: " + prompt + "\n\n"

    raw = call_gemini(final_prompt)

    parsed = safe_json_parse(raw) or {}
    if not parsed:
        # fallback: return raw text under 'raw' field and error
        parsed = {"error": "LLM output not parseable as JSON", "raw": raw}

    # Save as session message
    await ensure_session(email=email, prompt=prompt or f"Generated {mode} quiz", response=str(parsed), session_id=session_id)

    # Store quiz metadata to quiz_results (store but without score yet)
    quiz_doc = {
        "email": email,
        "session_id": session_id,
        "mode": mode.lower(),
        "generated_at": datetime.now(timezone.utc),
        "quiz_payload": parsed,
        "upload_results": upload_results
    }
    res = await quiz_results_collection.insert_one(quiz_doc)

    return {"quiz_id": str(res.inserted_id), "quiz": parsed, "mode": mode.lower(), "session_id": session_id}

# Request model for checking MCQ answers
class CheckAnswersRequest(BaseModel):
    quiz_id: Optional[str] = None
    email: str
    user_answers: Dict[str, Any]  # e.g. {"q1": 2, "q2": 0}
    correct_answers: Optional[Dict[str, Any]] = None  # optional, for safety

@app.post("/check_quiz_answers")
async def check_quiz_answers(payload: CheckAnswersRequest):
    # Retrieve quiz data if quiz_id passed
    quiz_doc = None
    if payload.quiz_id:
        quiz_doc = await quiz_results_collection.find_one({"_id": ObjectId(payload.quiz_id)})
    # Prefer correct_answers param if provided, else derive from quiz_doc
    correct_map = {}
    if payload.correct_answers:
        correct_map = payload.correct_answers
    elif quiz_doc and quiz_doc.get("quiz_payload"):
        q = quiz_doc["quiz_payload"]
        # If mcq format is array/dict
        if isinstance(q, list):
            # assume list of objects with id and answer_index
            for item in q:
                if "id" in item and "answer_index" in item:
                    correct_map[item["id"]] = item["answer_index"]
        elif isinstance(q, dict) and q.get("questions"):
            for item in q.get("questions", []):
                if "id" in item and "answer_index" in item:
                    correct_map[item["id"]] = item["answer_index"]
    # Score
    total = len(correct_map)
    correct = 0
    for qid, correct_idx in correct_map.items():
        user_sel = payload.user_answers.get(qid)
        # allow string numbers or ints
        try:
            if user_sel is None:
                continue
            if int(user_sel) == int(correct_idx):
                correct += 1
        except:
            # try string comparison
            if str(user_sel).strip().lower() == str(correct_idx).strip().lower():
                correct += 1

    percentage = round((correct / total) * 100, 2) if total > 0 else 0

    # Store result
    result_doc = {
        "quiz_id": payload.quiz_id,
        "email": payload.email,
        "score": correct,
        "total": total,
        "percentage": percentage,
        "user_answers": payload.user_answers,
        "evaluated_at": datetime.now(timezone.utc)
    }
    await quiz_results_collection.insert_one(result_doc)

    return {"score": correct, "total": total, "percentage": percentage}

# Teachback grading route - uses Gemini to grade by rubric or compares keywords
class TeachbackGradeRequest(BaseModel):
    quiz_id: Optional[str] = None
    email: str
    user_text: str
    rubric: Optional[List[str]] = None  # list of keywords or expected points

@app.post("/grade_teachback")
async def grade_teachback(payload: TeachbackGradeRequest):
    # If rubric provided, do simple keyword scoring, else ask Gemini to grade
    score = 0
    feedback = ""
    if payload.rubric:
        found = 0
        lowered = payload.user_text.lower()
        for kw in payload.rubric:
            if kw.lower() in lowered:
                found += 1
        total = len(payload.rubric)
        percentage = round((found / total) * 100, 2) if total>0 else 0
        score = int(round(percentage))
        feedback = f"Found {found}/{total} rubric items."
    else:
        # ask Gemini to grade freely (we will send grading instruction)
        grading_prompt = (
            "You are an assistant that MUST respond in JSON. Grade the following student answer on a scale 0-100 and provide "
            "short feedback. Return JSON: {\"score\": int, \"feedback\": \"...\"}\n\n"
            f"Student answer:\n{payload.user_text}\n\n"
            "Be concise."
        )
        raw = call_gemini(grading_prompt)
        parsed = safe_json_parse(raw)
        if parsed and isinstance(parsed, dict) and "score" in parsed:
            score = int(parsed.get("score", 0))
            feedback = str(parsed.get("feedback",""))
        else:
            # fallback naive: 0
            score = 0
            feedback = "Could not auto-grade (LLM parsing failed)."

    # Store result
    result_doc = {
        "quiz_id": payload.quiz_id,
        "email": payload.email,
        "teachback_score": score,
        "feedback": feedback,
        "user_text": payload.user_text,
        "evaluated_at": datetime.now(timezone.utc)
    }
    await quiz_results_collection.insert_one(result_doc)

    return {"score": score, "feedback": feedback}