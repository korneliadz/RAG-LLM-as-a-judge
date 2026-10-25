from rag import RAG
import pickle

def main():
    '''Rebuilds the Weaviate database from the pickled chunks with embeddings.'''
    rag = RAG()
    with open("all_chunks.pkl", "rb") as f:
        chunks_with_embeddings = pickle.load(f)

    try:
        rag.rebuild_database(chunks_with_embeddings)
    finally:
        if rag.client is not None:
            rag.client.close()

if __name__ == "__main__":
    main()