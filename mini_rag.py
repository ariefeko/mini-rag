"""
MINI-RAG: Q&A atas satu dokumen menggunakan LangChain
=======================================================

Alur (persis seperti yang didiskusikan):
1. LOAD    -> baca file .txt
2. SPLIT   -> potong jadi chunk-chunk kecil
3. EMBED   -> ubah tiap chunk jadi vector (pakai nomic-embed-text via Ollama, LOKAL)
4. STORE   -> simpan vector di Chroma (in-memory, tidak perlu server terpisah)
5. RETRIEVE-> saat user tanya, cari chunk paling relevan
6. GENERATE-> masukkan chunk relevan sebagai context ke prompt, minta LLM (Groq) jawab

Prasyarat sebelum menjalankan:
--------------------------------
1. Ollama jalan di background dan model nomic-embed-text sudah ditarik:
     ollama pull nomic-embed-text
     ollama serve   (biasanya otomatis jalan di background setelah install)

2. Install dependency Python:
     pip install langchain langchain-text-splitters langchain-ollama langchain-chroma langchain-groq chromadb python-dotenv

3. Simpan API key Groq (gratis, dari https://console.groq.com) di file .env
   di folder yang sama dengan script ini:
     GROQ_API_KEY=isi_api_key_kamu

Cara jalankan:
     python mini_rag.py
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

load_dotenv()  # baca file .env di folder yang sama dan set sebagai env var

SPECIFIC_LEAVE_KEYWORDS = {
    "sakit",
    "melahirkan",
    "lahiran",
    "menikah",
    "nikah",
    "haji",
    "kematian",
    "bencana",
    "mendesak",
    "pendamping",
}


def prepare_question(question: str) -> dict:
    normalized = question.lower()

    is_generic_leave = (
        "cuti" in normalized
        and not any(
            keyword in normalized
            for keyword in SPECIFIC_LEAVE_KEYWORDS
        )
    )

    if is_generic_leave:
        interpreted_question = (
            f"{question}. Anggap permintaan cuti tanpa jenis khusus "
            "sebagai cuti tahunan."
        )
        search_query = f"kebijakan cuti tahunan {question}"
    else:
        interpreted_question = question
        search_query = question

    return {
        "question": question,
        "interpreted_question": interpreted_question,
        "search_query": search_query,
    }


def build_rag_pipeline(file_path: str):
    # 1. LOAD - baca dokumen
    # Catatan: file .txt sederhana dibaca langsung pakai Python biasa lalu
    # dibungkus jadi Document, supaya tidak perlu depend ke TextLoader dari
    # langchain-community (package tersebut sudah di-sunset/archived).
    print("[1/5] Loading dokumen...")
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    documents = [Document(page_content=text, metadata={"source": file_path})]

    # 2. SPLIT - potong jadi chunk kecil
    # chunk_size kecil karena dokumen contoh ini pendek.
    # Untuk dokumen panjang (misal codebase), biasanya 500-1000 karakter per chunk.
    print("[2/5] Splitting dokumen jadi chunk...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,  # overlap supaya konteks antar chunk tidak putus total
    )
    chunks = splitter.split_documents(documents)
    print(f"      -> Total {len(chunks)} chunk dihasilkan")

    # 3. EMBED - siapkan model embedding LOKAL (gratis, jalan di CPU kamu)
    print("[3/5] Menyiapkan embedding model (nomic-embed-text via Ollama)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. STORE - simpan ke vector store in-memory (Chroma)
    print("[4/5] Membuat vector store (Chroma, in-memory)...")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    # retriever = komponen yang nyari chunk paling relevan
    # k=2 artinya ambil 2 chunk paling mirip dengan pertanyaan
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. GENERATE - siapkan LLM (Groq, gratis tier, cepat karena LPU)
    print("[5/5] Menyiapkan LLM (Groq Llama 3.3 70B)...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_template(
    """Jawab berdasarkan context dan aturan interpretasi berikut.

Aturan:
- Permintaan "cuti" tanpa menyebut sakit, menikah, melahirkan, haji, kematian, atau kondisi khusus dianggap sebagai cuti tahunan.
- Durasi seperti 1 hari atau 2 hari adalah jumlah cuti yang ingin diambil, bukan jenis cutinya.
- Jangan mengatakan informasi tidak ditemukan apabila jenis cutinya dapat ditentukan menggunakan aturan di atas.
- Jangan menganggap permintaan sudah disetujui.
- Sebutkan persyaratan pengajuan yang tersedia di context.

Context:
{context}

Pertanyaan:
{question}

Pertanyaan terinterpretasi:
{interpreted_question}

Jawaban:"""
)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Rangkai semua jadi satu chain menggunakan LCEL (LangChain Expression Language)
    rag_chain = (
        RunnableLambda(prepare_question)
        | {
            "context": (
                RunnableLambda(lambda data: data["search_query"])
                | retriever
                | RunnableLambda(format_docs)
            ),
            "question": RunnableLambda(lambda data: data["question"]),
            "interpreted_question": RunnableLambda(
                lambda data: data["interpreted_question"]
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Pipeline siap!\n")
    return rag_chain


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY belum ditemukan.")
        print("Pastikan file .env ada di folder yang sama dengan script ini, isinya:")
        print("  GROQ_API_KEY=isi_api_key_kamu")
        return

    file_path = os.path.join(os.path.dirname(__file__), "kebijakan_cuti.txt")
    rag_chain = build_rag_pipeline(file_path)

    print("=== Mini-RAG Q&A siap. Ketik 'exit' untuk keluar. ===\n")
    while True:
        try:
            question = input("Pertanyaan: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAgent stopped.")
            break

        if question.lower() in ("exit", "quit", "keluar"):
            break
        if not question:
            continue

        try:
            answer = rag_chain.invoke(question)
        except KeyboardInterrupt:
            print("\nAgent stopped.")
            break

        print(f"\nJawaban: {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAgent stopped.")