import os
import threading
import time
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="./",
    embedding_function=embeddings
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful citizen service assistant for Pakistan.
Answer in simple plain language that anyone can understand.
If the user writes in Roman Urdu, reply in Roman Urdu.
If you don't know the answer say "Mujhe is baray mein maloomat nahi, please relevant department se rabta karein."
Keep answers concise and practical.
Context: {context}
Question: {question}
""")

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# keep-alive thread to prevent Railway cold starts
def keep_alive():
    while True:
        time.sleep(300)
        try:
            requests.get("https://awam-assist-production.up.railway.app/")
        except:
            pass

threading.Thread(target=keep_alive, daemon=True).start()

@app.get("/")
def root():
    return {"status": "Citizen Service Navigator API is running"}

@app.post("/ask")
async def ask(body: Question):
    try:
        async def generate():
            async for chunk in chain.astream(body.question):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.webm", audio_bytes, "audio/webm"),
            language="ur",
            prompt="sehat card, NADRA, CNIC, zakat, IESCO, passport, BISP, ehsaas, fard, NTN, bijli, pani, gas"
        )
        return {"transcript": transcript.text}
    except Exception as e:
        return {"transcript": f"Error: {str(e)}"}
