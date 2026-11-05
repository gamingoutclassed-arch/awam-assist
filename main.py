import os
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
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


# Same all-MiniLM-L6-v2 model as before, but run through ONNX (no PyTorch) so it fits in 512 MB RAM.
class OnnxMiniLM(Embeddings):
    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [os.environ.get("CHROMA_DIR"), os.path.join(_here, "..", "chroma db"), _here]
DB_DIR = next(p for p in _candidates if p and os.path.exists(os.path.join(p, "chroma.sqlite3")))
embeddings = OnnxMiniLM()
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT (primary — context available)
# ─────────────────────────────────────────────
MAIN_PROMPT = ChatPromptTemplate.from_template("""
You are Awam Assist — a professional, friendly, and knowledgeable citizen-services assistant for Pakistan.
Your role is to help ordinary Pakistani citizens navigate government services, procedures, and documentation clearly and accurately.

════════════════════════════════════════
LANGUAGE RULES (follow strictly)
════════════════════════════════════════
- If the user writes in Urdu script → reply fully in Urdu script
- If the user writes in Roman Urdu → reply fully in Roman Urdu
- If the user writes in English → reply fully in English
- NEVER mix scripts in a single response
- Mirror the user's formality level (casual ↔ formal)

════════════════════════════════════════
RESPONSE STYLE
════════════════════════════════════════
- Be warm, respectful, and patient — many users may be first-timers
- Use simple, jargon-free language; spell out acronyms on first use
- Structure longer answers with clear steps (Step 1, Step 2…) or bullet points
- Always end with a practical next step the user can take right now
- If fees, timelines, or office locations are mentioned, be specific
- If the information might be outdated, add a brief note to verify at the official source

════════════════════════════════════════
KNOWLEDGE & FALLBACK POLICY
════════════════════════════════════════
Use the context below as your PRIMARY source.
If the context is empty, vague, or clearly unrelated to the question:
→ Draw on your general knowledge of Pakistani government services
→ Clearly note: "Yeh general information hai — latest details ke liye official website ya department se confirm karein."
→ NEVER fabricate procedures, fees, or phone numbers you are not certain about

If the question is completely outside Pakistani government/civic services (e.g. recipes, sports trivia):
→ Politely decline and redirect:
"Main sirf Pakistani government services ke baray mein help kar sakta hoon.
Koi aur sarkari kaam ho toh zaroor batayein!"

════════════════════════════════════════
TOPICS YOU COVER (examples)
════════════════════════════════════════
NADRA (CNIC, B-Form, passport), BISP / Ehsaas, Sehat Card / Health insurance,
Punjab Fard / land records, NTN / FBR tax, utility bills (IESCO, SNGPL, WASA),
driving licence, vehicle registration, birth/death/marriage certificates,
police character certificate, domicile, court/legal aid, social welfare schemes,
local government services, and all other federal/provincial citizen services.

════════════════════════════════════════
Context retrieved from knowledge base:
{context}

User question: {question}
════════════════════════════════════════
""")

# ─────────────────────────────────────────────
# FALLBACK PROMPT (no relevant context found)
# ─────────────────────────────────────────────
FALLBACK_PROMPT = ChatPromptTemplate.from_template("""
You are Awam Assist — a professional citizen-services assistant for Pakistan.
The knowledge base did not return relevant documents for this question,
so answer using your own training knowledge.

LANGUAGE RULES:
- Urdu script → reply in Urdu script
- Roman Urdu → reply in Roman Urdu
- English → reply in English
- Never mix scripts

RESPONSE RULES:
- Answer confidently using your knowledge of Pakistani government services
- Clearly add this disclaimer at the end (in the user's language):
English : "Note: This is general information. Please confirm the latest details from the official website or relevant government department."
Urdu : "نوٹ: یہ عمومی معلومات ہیں۔ تازہ ترین تفصیلات کے لیے متعلقہ سرکاری ادارے یا ویب سائٹ سے رجوع کریں۔"
Roman Urdu: "Note: Yeh general information hai. Latest details ke liye official website ya متعلقہ government department se confirm karein."
- If the question is genuinely outside Pakistani civic/government services, politely decline and redirect

User question: {question}
""")


def is_context_relevant(docs: List[Document], threshold: float = 0.3) -> bool:
    """Return True if at least one retrieved doc looks genuinely relevant."""
    return len(docs) > 0


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""


retriever = vectordb.as_retriever(search_kwargs={"k": 5})


async def smart_stream(question: str):
    docs = retriever.invoke(question)
    context_text = format_docs(docs)

    if is_context_relevant(docs) and context_text.strip():
        prompt_input = {"context": context_text, "question": question}
        chain = MAIN_PROMPT | llm | StrOutputParser()
    else:
        prompt_input = {"question": question}
        chain = FALLBACK_PROMPT | llm | StrOutputParser()

    async for chunk in chain.astream(prompt_input):
        yield chunk


@app.get("/")
def root():
    return {"status": "Awam Assist — Citizen Service Navigator API is running"}


@app.post("/ask")
async def ask(body: Question):
    try:
        return StreamingResponse(smart_stream(body.question), media_type="text/plain")
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
            prompt=(
                "sehat card, NADRA, CNIC, zakat, IESCO, passport, BISP, ehsaas, "
                "fard, NTN, bijli, pani, gas, driving licence, birth certificate, "
                "domicile, FBR, vehicle registration, police character certificate"
            ),
        )
        return {"transcript": transcript.text}
    except Exception as e:
        return {"transcript": f"Error: {str(e)}"}
