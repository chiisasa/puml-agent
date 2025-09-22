from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

PERSIST = "./vectorstore/plantuml"
MODEL   = "text-embedding-3-small"
TOPK    = 10  # 必要に応じて変更

def main():
    emb = OpenAIEmbeddings(model=MODEL, chunk_size=64)

    print("[faiss:load:start]")
    vs = FAISS.load_local(
        folder_path=PERSIST,
        embeddings=emb,
        allow_dangerous_deserialization=True,
    )
    print("[faiss:load:done]")
    print("Enter query (empty to quit):")

    try:
        while True:
            q = input("query> ").strip()
            if not q:
                break

            docs = vs.similarity_search(q, k=TOPK)
            if not docs:
                print("(no hits)")
                continue

            for i, d in enumerate(docs, 1):
                file = d.metadata.get("file", "-")
                off  = d.metadata.get("offset", "-")
                print("\n" + "-" * 80)
                print(f"[{i}] file={file} offset={off}")
                print("-" * 80)
                print(d.page_content)
            print("\n" + "=" * 80)
    except (EOFError, KeyboardInterrupt):
        pass

if __name__ == "__main__":
    main()
