# Importing required libraries and modules
from llama_index.llms.openai import OpenAI
import logging
from typing import Any, Dict, List, Optional, cast
import uuid
import os

# LlamaIndex internals for schema and vector store support
from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import(
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore

# Moorcheh SDK for backend vector storage
from moorcheh_sdk import MoorchehClient, MoorchehError
from moorcheh_sdk import MoorchehClient
from typing import ClassVar
from pydantic import Field, PrivateAttr, model_validator
from google.colab import userdata

# Logger for debug/info/error output
logger = logging.getLogger(__name__)

class MoorchehVectorStore():
    """Moorcheh Vector Store.

    In this vector store, embeddings and docs are stored within a Moorcheh namespace.
    During query time, the index uses Moorcheh to query for the top k most similar nodes.

    Args:
        api_key (Optional[str]): API key for Moorcheh.
            If not provided, will look for MOORCHEH_API_KEY environment variable.
        namespace (str): Namespace name to use for this vector store.
        namespace_type (str): Type of namespace - "text" or "vector".
        vector_dimension (Optional[int]): Vector dimension for vector namespace.
        batch_size (int): Batch size for adding nodes. Defaults to DEFAULT_EMBED_BATCH_SIZE.
        **kwargs: Additional arguments to pass to MoorchehClient.
    """

    # Default values and capabilities
    DEFAULT_NAMESPACE: ClassVar[str] = "llamaindex_default"
    DEFAULT_EMBED_BATCH_SIZE: ClassVar[int] = 64  # customize as needed

    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(self, api_key, namespace, namespace_type="text", vector_dimension=None, batch_size=64):
        # Initialize store attributes
        self.api_key=api_key
        self.namespace=namespace
        self.namespace_type=namespace_type
        self.vector_dimension=vector_dimension
        self.batch_size=batch_size

        # Initialize Moorcheh client
        self._client = MoorchehClient(api_key=self.api_key, base_url="https://wnc4zvnuok.execute-api.us-east-1.amazonaws.com/v1")
        self.is_embedding_query=False

    '''
    api_key: Optional[str] = Field(default=None)
    namespace: Optional[str] = Field(default="llamaindex_default")
    namespace_type: Optional[str] = Field(default="text")
    vector_dimension: Optional[int] = Field(default=None)
    batch_size: int = Field(default=64)
    '''
    # _client: MoorchehClient = PrivateAttr()


    def model_post_init(self):
        # Fallback to env var if API key not provided
        if not self.api_key:
            self.api_key = os.getenv("MOORCHEH_API_KEY")
        if not self.api_key:
            raise ValueError("`api_key` is required for Moorcheh client initialization")

        if not self.namespace:
            raise ValueError("`namespace` is required for Moorcheh client initialization")

        print("[DEBUG] Initializing MoorchehClient")

        print("[DEBUG] Listing namespaces...")
        try:
            namespaces = self._client.list_namespaces()
            print(f"[DEBUG] Found namespaces: {namespaces}")
        except Exception as e:
            print(f"[ERROR] Failed to list namespaces: {e}")
            raise

        if self.namespace not in namespaces:
            print(f"[DEBUG] Namespace '{self.namespace}' not found. Creating...")
            try:
                self._client.create_namespace(
                    namespace_name=self.namespace,
                    type=self.namespace_type,
                    vector_dimension=self.vector_dimension,
                )
            except Exception as e:
                print(f"[ERROR] Failed to create namespace: {e}")
                raise


        print("[DEBUG] MoorchehVectorStore initialization complete.")


    @property
    def client(self) -> MoorchehClient:
      """Return initialized Moorcheh client."""
      return self._client


    @classmethod
    def class_name(cls) -> str:
        """Return class name."""
        return "MoorchehVectorStore"


    @property
    def client(self) -> MoorchehClient:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to Moorcheh."""
        if not nodes:
            return []

        if self.namespace_type == "text":
            return self._add_text_nodes(nodes, **add_kwargs)
        else:
            return self._add_vector_nodes(nodes, **add_kwargs)

    def _add_text_nodes(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Add text documents to a text namespace."""
        documents = []
        ids = []

        for node in nodes:
            node_id = node.node_id or str(uuid.uuid4())
            ids.append(node_id)

            document = {
                "id": node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
            }

            # Add metadata if present
            if node.metadata:
                document["metadata"] = node.metadata

            documents.append(document)

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            try:
                result = self._client.upload_documents(
                    namespace_name=self.namespace, documents=batch
                )
                logger.debug(f"Uploaded batch of {len(batch)} documents")
            except MoorchehError as e:
                logger.error(f"Error uploading documents batch: {e}")
                raise

        logger.info(f"Added {len(documents)} text documents to namespace {self.namespace}")
        return ids

    def _add_vector_nodes(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Add vector nodes to vector namespace."""
        vectors = []
        ids = []

        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node {node.node_id} has no embedding for vector namespace")

            node_id = node.node_id or str(uuid.uuid4())
            ids.append(node_id)

            vector = {
                "id": node_id,
                "vector": node.embedding,
            }

            # Add metadata, including text content
            metadata = dict(node.metadata) if node.metadata else {}
            metadata["text"] = metadata.pop("text", node.get_content(metadata_mode=MetadataMode.NONE))
            vector["metadata"] = metadata

            vectors.append(vector)

        # Process in batches
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i : i + self.batch_size]
            try:
                result = self._client.upload_vectors(
                    namespace_name=self.namespace, vectors=batch
                )
                logger.debug(f"Uploaded batch of {len(batch)} vectors")
            except MoorchehError as e:
                logger.error(f"Error uploading vectors batch: {e}")
                raise

        logger.info(f"Added {len(vectors)} vectors to namespace {self.namespace}")
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        try:
            if self.namespace_type == "text":
                result = self._client.delete_documents(
                    namespace_name=self.namespace, ids=[ref_doc_id]
                )
            else:
                result = self._client.delete_vectors(
                    namespace_name=self.namespace, ids=[ref_doc_id]
                )
            logger.info(f"Deleted document {ref_doc_id} from namespace {self.namespace}")
        except MoorchehError as e:
            logger.error(f"Error deleting document {ref_doc_id}: {e}")
            raise


    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query Moorcheh vector store.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            logger.warning(
                f"Moorcheh does not support query mode {query.mode}. "
                "Using default mode instead."
            )

        # Prepare search parameters
        search_kwargs = {
            "namespaces": [self.namespace],
            "top_k": query.similarity_top_k,
        }

        # Add similarity threshold if provided
        #if query.similarity_top_k is not None:
        #    search_kwargs["threshold"] = query.similarity_top_k

        # Handle query input
        if query.query_str is not None:
            search_kwargs["query"] = query.query_str
        elif query.query_embedding is not None:
            search_kwargs["query"] = query.query_embedding
        else:
            raise ValueError("Either query_str or query_embedding must be provided")

        # TODO: Add metadata filter support when available in Moorcheh SDK
        if query.filters is not None:
            logger.warning("Metadata filters are not yet supported by Moorcheh integration")

        try:
            # Execute search
            search_result = self._client.search(**search_kwargs)

            # Parse results
            nodes = []
            similarities = []
            ids = []

            results = search_result.get("results", [])
            for result in results:
                node_id = result.get("id")
                score = result.get("score", 0.0)

                if node_id is None:
                    logger.warning("Found result with no ID, skipping")
                    continue

                ids.append(node_id)
                similarities.append(score)

                # Extract text and metadata
                if self.namespace_type == "text":
                    text = result.get("text", "")
                    metadata = result.get("metadata", {})
                else:
                    # For vector namespace, text is stored in metadata
                    metadata = result.get("metadata", {})
                    text = metadata.pop("text", "")  # Remove text from metadata

                # Create node
                node = TextNode(
                    text=text,
                    id_=node_id,
                    metadata=metadata,
                )
                nodes.append(node)

            return VectorStoreQueryResult(
                nodes=nodes,
                similarities=similarities,
                ids=ids,
            )

        except MoorchehError as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_generative_answer(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> str:
        """Get a generative AI answer using Moorcheh's built-in RAG capability.

        This method leverages Moorcheh's information-theoretic approach
        to provide context-aware answers directly from the API.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to use for context.
            **kwargs: Additional keyword arguments passed to Moorcheh.

        Returns:
            str: Generated answer string.
        """
        try:
            result = self._client.get_generative_answer(
                namespace=self.namespace,
                query=query,
                top_k=top_k,
                **kwargs,
            )
            return result.get("answer", "")
        except MoorchehError as e:
            logger.error(f"Error getting generative answer: {e}")
            raise



if __name__ == "__main__":
    print("MoorchehVectorStore loaded successfully.")
