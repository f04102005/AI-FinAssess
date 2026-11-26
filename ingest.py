import os
import time
import shutil
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
INDEX_DIR = "faiss_index"


def load_documents(data_dir: str) -> list[Document]:
    docs = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": filename, "page": f"p.{page_num}"},
                        )
                    )
        elif filename.endswith(".docx"):
            docx_file = DocxDocument(path)
            paras = [p.text.strip() for p in docx_file.paragraphs if p.text.strip()]
            batch_size = 15
            for i in range(0, len(paras), batch_size):
                chunk_paras = paras[i : i + batch_size]
                start_p = i + 1
                end_p = i + len(chunk_paras)
                docs.append(
                    Document(
                        page_content="\n".join(chunk_paras),
                        metadata={"source": filename, "page": f"Para {start_p}-{end_p}"},
                    )
                )
    return docs


def build_index():
    raw_docs = load_documents(DATA_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)
    print(f"Total chunks to index: {len(chunks)}")

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    batch_size = 40
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        done = min(i + batch_size, len(chunks))
        print(f"Indexed {done}/{len(chunks)} chunks...")
        if done < len(chunks):
            time.sleep(30)

    vectorstore.save_local(INDEX_DIR)
    print(f"Done. Saved '{INDEX_DIR}/' with {len(chunks)} chunks.")


if __name__ == "__main__":
    build_index()