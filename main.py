from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Setup MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.chat_db
collection = db.chat_logs

# Create FastAPI app
app = FastAPI()

# Pydantic model for incoming chat data
class ChatMessage(BaseModel):
    sender: str       # "user" or "bot"
    type: str         # "text", "audio"
    text: str = ""    # Optional
    blob_url: str = ""  # Optional

# POST endpoint to store a chat message
@app.post("/chat/")
async def store_chat(message: ChatMessage):
    doc = {
        "sender": message.sender,
        "type": message.type,
        "text": message.text,
        "blob_url": message.blob_url,
        "timestamp": datetime.utcnow()
    }
    result = await collection.insert_one(doc)
    if result.inserted_id:
        return {
            "status": "success",
            "id": str(result.inserted_id),
            "stored_at": doc["timestamp"].isoformat()
        }
    raise HTTPException(status_code=500, detail="Failed to store chat")
