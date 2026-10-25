import pickle
from rag import RAG

documents = [
    {
        "path": "source-documents/Prawo o szkolnictwie wyzszym i nauce.pdf",
        "title": "Prawo o szkolnictwie wyższym i nauce",
        "date": "2018-07-20",
        "document_id": "1668",
        "document_type": "ustawa",
        "issuing_authority": "Sejm RP"
    },
    {
        "path": "source-documents/Regulamin studiów w PŁ (zmiany od 1.10.2026).pdf",
        "title": "Regulamin Studiów PŁ",
        "date": "2022-04-27",
        "document_id": "20/2022",
        "document_type": "regulamin",
        "issuing_authority": "Senat PŁ"
    }
]

def main():
    rag = RAG()

    chunks = rag.chunking(documents)
    chunks_with_embeddings = rag.generate_embeddings(chunks)

    with open("all_chunks.pkl", "wb") as f:
        pickle.dump(chunks_with_embeddings, f)
    
    try:
         rag.rebuild_database(chunks_with_embeddings)
    finally:
        if rag.client is not None:
            rag.client.close()

if __name__ == "__main__":
    main()