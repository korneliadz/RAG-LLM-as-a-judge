import pickle
from rag import RAG

documents = [
    {
        "path": "source-documents/nowy_dokument.pdf",
        "title": "...",
        "date": "...",
        "document_id": "...",
        "document_type": "...",
        "issuing_authority": "..."
    }
]

def main():
    rag = RAG()

    try:
        chunks = rag.chunking(documents)
        chunks_with_embeddings = rag.generate_embeddings(chunks)

        rag.add_to_database(chunks_with_embeddings)

        try:
            with open("all_chunks.pkl", "rb") as f:
                all_chunks = pickle.load(f)
        except FileNotFoundError:
            all_chunks = []

        all_chunks.extend(chunks_with_embeddings)

        with open("all_chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)

    finally:
        if rag.client is not None:
            rag.client.close()

if __name__ == "__main__":
    main()