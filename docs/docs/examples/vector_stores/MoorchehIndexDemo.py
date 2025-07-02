# --- Welcome to the Demo of the Moorcheh Vector Store ---
# --- Import the following packages --
import logging
import sys
import os
from moorcheh_sdk import MoorchehClient
from IPython.display import Markdown, display
from typing import Any, Callable, Dict, List, Optional, cast
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


# --- Logging Setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# --- Check if the API Keys are set in Enviornment Variables ---
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError(f"Environment variable OPENAI_API_KEY is not set")
if "MOORCHEH_API_KEY" not in os.environ:
    raise EnvironmentError(f"Environment variable MOORCHEH_API_KEY is not set")

# --- Set the values of tbe API Keys in your Environment Variables ---
api_key = os.environ["MOORCHEH_API_KEY"]
openai_key = os.environ["OPENAI_API_KEY"]


# --- Load Documents ---
documents = SimpleDirectoryReader("./your_documents_folder").load_data()


# --- Initialize the Moorcheh Vector Store ---
__all__ = ["MoorchehVectorStore"]

# Creates a Moorcheh Vector Store with the following parameters
# For text-based namespaces, set namespace_type to "text" and vector_dimension to None
# For vector-based namespaces, set namespace_type to "vector" and vector_dimension to the dimension of your uploaded vectors
vector_store = MoorchehVectorStore(api_key=api_key, namespace="namespace_name", namespace_type="text", vector_dimension=None, add_sparse_vector=True, batch_size=100)



# --- Create a Vector Store Index using the Vector Store and given Documents ---
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)


# --- Generate Response ---
# --- Set Logging to DEBUG for more Detailed Outputs ---
query_engine = index.as_query_engine()
response = query_engine.query("Add your query here.")

display(Markdown(f"<b>{response}</b>"))
print("\n\n================================\n\n", response, "\n\n================================\n\n")


# --- Filters for Metadata ---
filter = MetadataFilters(
    filters=[
        MetadataFilter(
            key="file_path",
            value="insert the file path to the document here",
            operator=FilterOperator.EQ,
        )
    ],
    condition=FilterCondition.AND,
)
