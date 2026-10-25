from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import weaviate
import re

class RAG:
    def __init__(
            self, 
            host = 'localhost', 
            port = 8080, 
            grpc_port = 50051, 
            model_name = 'intfloat/multilingual-e5-base'
            ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.model_name = model_name

        self.model = None
        self.client = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _connect_database(self):
        if self.client is None:
            self.client = weaviate.connect_to_local(
                self.host,
                self.port,
                self.grpc_port
            )

    def _split_long_text(self, text, max_chars = 1500, overlap = 150):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                dot = text.rfind(".", start, end)
                if dot > start + 500:
                    end = dot + 1
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = end - overlap
        return chunks
    
    def _create_chunk(self, text, doc, paragraph, article):
        return {
            "text": text,
            "source": doc["title"],
            "document_id": doc["document_id"],
            "document_type": doc["document_type"],
            "date": doc["date"],
            "issuing_authority": doc["issuing_authority"],
            "paragraph": paragraph.group(0) if paragraph else None,
            "article": article.group(0) if article else None
        }
    
    def _batch(self, chunks, collection_name = "StudyChunks"):
        self._connect_database()
        collection = self.client.collections.get(collection_name)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                properties = chunk.copy()
                vector = properties.pop("embedding")
                batch.add_object(
                    properties = properties,
                    vector = vector
                )
    
    def chunking(self, documents, max_chars = 1500, overlap = 150):
        all_chunks = []

        for doc in documents:
            doc_chunks = []
            reader = PdfReader(doc["path"])
            full_text = ""
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    full_text += txt + "\n"

            full_text = re.sub(r"\s+", " ", full_text)

            if doc["document_type"] == "regulamin":
                sections = re.split(r"(?=§\s*\d+)", full_text)
            elif doc["document_type"] == "ustawa":
                sections = re.split(r"(?=Art\.\s*\d+[a-zA-Z]?)", full_text)
            else:
                sections = [full_text]

            for section in sections:
                section = section.strip()
                if len(section) < 50:
                    continue

                paragraph = re.search(r"§\s*\d+", section)
                article = re.search(r"Art\.\s*\d+[a-zA-Z]?", section)

                if len(section) <= max_chars:
                    doc_chunks.append(
                    self._create_chunk(
                        section,
                        doc,
                        paragraph,
                        article
                    )
                )
                    
                else:
                    subchunks = self._split_long_text(
                        section, 
                        max_chars=max_chars, 
                        overlap=overlap
                    )
                    for subchunk in subchunks:
                        doc_chunks.append(
                        self._create_chunk(
                            subchunk,
                            doc,
                            paragraph,
                            article
                        )
                    )
            for index, chunk in enumerate(doc_chunks):
                chunk["chunk_index"] = index

            all_chunks.extend(doc_chunks)
                        
        return all_chunks

    def generate_embeddings(self, chunks):
        self._load_model()
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks

    def add_to_database(self, chunks, collection_name = "StudyChunks"):
        self._connect_database()
        if not self.client.collections.exists(collection_name):
            self.client.collections.create(name=collection_name)
        self._batch(chunks, collection_name)

    def rebuild_database(self, chunks, collection_name = "StudyChunks"):
        self._connect_database()
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)

        self.client.collections.create(name=collection_name)
        self._batch(chunks, collection_name)