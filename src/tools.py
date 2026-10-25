from weaviate.classes.query import Filter
from weaviate.classes.query import MetadataQuery

class RAGTools:

    def __init__(self, collection, model):
        self.collection = collection
        self.model = model

    def _serialize_chunk(self, obj):
            return {
                "chunk_id": str(obj.uuid),
                "text": obj.properties["text"],
                "source": obj.properties["source"],
                "document_id": obj.properties["document_id"],
                "document_type": obj.properties["document_type"],
                "date": obj.properties.get("date"),
                "issuing_authority": obj.properties.get("issuing_authority"),
                "paragraph": obj.properties.get("paragraph"),
                "article": obj.properties.get("article")
            }

    
    def search_vector(self, query: str, top_k: int):
        '''
        Searches for the most relevant chunks based on the query.

        Args:
            query: The search query.
            top_k: The number of top results to return.

        Returns:
            A list of the most relevant chunks.
        '''
        query_embedded = self.model.encode(query)

        res = self.collection.query.near_vector(
            near_vector=query_embedded,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )

        return [self._serialize_chunk(obj) for obj in res.objects]


    def get_chunk(self, chunk_id: str):
        '''
        Retrieves a specific chunk of text.

        Args:
            chunk_id: The ID of the chunk to retrieve.

        Returns:
            The text of the chunk if found.
        '''
        res = self.collection.query.fetch_object_by_id(chunk_id)

        if res is None:
            return None

        return res.properties["text"]


    def get_document(self, document_id: str):
        '''
        Retrieves information about a source document.

        Args:
            document_id: The ID of the document.

        Returns:
            Metadata of the source document.
        '''
        res = self.collection.query.fetch_objects(
            filters=Filter.by_property("document_id").equal(document_id),
            limit=1
        )

        if not res.objects:
            return None

        properties = res.objects[0].properties

        return {
            "document_id": properties.get("document_id"),
            "title": properties.get("source"),
            "document_type": properties.get("document_type"),
            "date": properties.get("date"),
            "issuing_authority": properties.get("issuing_authority")
        }


    def get_neighboring_chunks(self, chunk_id: str, window: int):
        '''
        Retrieves neighboring chunks around a specified chunk.

        Args:
            chunk_id: The ID of the chunk.
            window: Number of neighboring chunks on each side.

        Returns:
            A list of neighboring chunks.
        '''
        chunk = self.collection.query.fetch_object_by_id(chunk_id)

        document_id = chunk.properties["document_id"]
        chunk_index = chunk.properties["chunk_index"]

        res = self.collection.query.fetch_objects(
            filters=(
                Filter.by_property("document_id").equal(document_id)
                & Filter.by_property("chunk_index").greater_or_equal(
                    chunk_index - window
                )
                & Filter.by_property("chunk_index").less_or_equal(
                    chunk_index + window
                )
            ),
            limit=100
        )

        return [self._serialize_chunk(obj) for obj in res.objects]


    def search_by_metadata(
        self,
        property_name: str,
        value: str
        ):
        '''
        Searches for chunks based on metadata.

        Args:
            property_name: Metadata property to search by, for example
                document_type, source, date or issuing_authority.
            value: Value that the metadata property should have.

        Returns:
            A list of chunks matching the metadata.
        '''
        filters = Filter.by_property(property_name).equal(value)

        res = self.collection.query.fetch_objects(
            filters=filters,
            limit=100
        )

        return [self._serialize_chunk(obj) for obj in res.objects]