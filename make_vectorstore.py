from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language
from langchain_community.vectorstores import FAISS
from fnmatch import fnmatch
import os

PROJECT = "./plantuml"
PERSIST = "./vectorstore/plantuml2"
MODEL   = "text-embedding-3-small"

TARGET_EXTS  = {".java", ".puml", ".pu", ".md", ".txt"}
EXCLUDE_FILES_PAT = {".*"}
EXCLUDE_DIRS_PAT = {"bin", "build", "gradle", ".*"}

java_sp = RecursiveCharacterTextSplitter.from_language(
    # language=Language.JAVA, chunk_size=1600, chunk_overlap=150, add_start_index=True
    language=Language.JAVA, chunk_size=900, chunk_overlap=150, add_start_index=True
)
puml_sp = RecursiveCharacterTextSplitter(
    separators=["\n@enduml", "@enduml", "@startuml", "\n\n", ""],
    # chunk_size=1800, chunk_overlap=150, add_start_index=True
    chunk_size=1200, chunk_overlap=50, add_start_index=True
)
md_hdr  = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
# generic = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=150, add_start_index=True)
generic = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, add_start_index=True)

def split_by_ext(text: str, meta: dict):
    ext = meta["ext"]
    if ext == ".java":
        return java_sp.create_documents([text], metadatas=[meta])
    if ext in (".puml", ".pu"):
        return puml_sp.create_documents([text], metadatas=[meta])
    if ext == ".md":
        docs = []
        for sec in md_hdr.split_text(text):
            docs += generic.create_documents([sec.page_content], metadatas=[{**meta, **sec.metadata}])
        return docs
    return generic.create_documents([text], metadatas=[meta])

def main():
    if not os.getenv("OPENAI_API_KEY"): raise SystemExit("ERROR: OPENAI_API_KEY is not set.")
    root = Path(PROJECT).resolve()
    if not root.exists(): raise SystemExit(f"ERROR: project path not found: {root}")

    print(f"[scan] {root}")
    docs = []
    for p in root.rglob("*"):
        if not p.is_file(): continue
        if any(fnmatch(seg, pat) for seg in p.parts[:-1] for pat in EXCLUDE_DIRS_PAT):
            continue
        if any(fnmatch(p.name, pat) for pat in EXCLUDE_FILES_PAT):
            continue

        ext = p.suffix.lower()
        if ext not in TARGET_EXTS: continue
        try:
            print(f"{p.relative_to(root)}")
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            print(f"  skip: read error")
            continue
        parts = split_by_ext(text, {"file": str(p.relative_to(root)), "ext": ext})
        for d in parts:
            if "start_index" in d.metadata and "offset" not in d.metadata:
                d.metadata["offset"] = d.metadata["start_index"]
        docs.extend(parts)

    if not docs: raise SystemExit("No target documents.")

    print(f"[faiss:start] n_docs={len(docs)} model={MODEL}")
    emb = OpenAIEmbeddings(model=MODEL, chunk_size=500)
    vs = FAISS.from_documents(documents=docs, embedding=emb)
    print("[faiss:build:done]")
    
    Path(PERSIST).mkdir(parents=True, exist_ok=True)
    vs.save_local(PERSIST)
    print(str(Path(PERSIST).resolve()))

if __name__ == "__main__":
    main()
