# https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo/
import time
import torch
from neo4j_utils.neo4j_server import reset_neo4j_server, get_neo4j_credentails, wait_for_neo4j_server, is_neo4j_server_up
from llama_index.core import Settings
from huggingface_hub import hf_hub_download, snapshot_download
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from IPython.display import Markdown, display

if not is_neo4j_server_up():
    reset_neo4j_server()
    wait_for_neo4j_server()

supported_embed_models = ["thenlper/gte-large"]

supported_llm_models = {
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    "microsoft/Phi-3-mini-4k-instruct-gguf": "Phi-3-mini-4k-instruct-q4.gguf",
}

model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
embed_model_name="thenlper/gte-large"
temperature=0.0
max_new_tokens=256
context_window=4096
gpu_layers=20
dim=1024
memory_token_limit=4096
sentense_embedding_percentile_cutoff=0.8
similarity_top_k=2
hf_token="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"

MODELS_PATH = "./models"
EMBED_PATH = "./embed_models"

n_gpu_layers = 0
if torch.cuda.is_available():
    print("It is a GPU node, setup GPU.")
    n_gpu_layers = gpu_layers

def get_model_path(model_name):
    filename = supported_llm_models[model_name]
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=filename,
        resume_download=True,
        cache_dir=MODELS_PATH,
        local_files_only=False,
        token=hf_token,
    )
    return model_path

def get_embed_model_path( embed_model):
    embed_model_path = snapshot_download(
        repo_id=embed_model,
        resume_download=True,
        cache_dir=EMBED_PATH,
        local_files_only=False,
        token=hf_token,
    )
    return embed_model_path

llm = LlamaCPP(
    model_path=get_model_path(model_name),
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=context_window,
    # kwargs to pass to __call__()
    # generate_kwargs={"temperature": 0.0, "top_k": 5, "top_p": 0.95},
    generate_kwargs={"temperature": temperature},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": n_gpu_layers},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = HuggingFaceEmbedding(
    model_name=embed_model_name,
    cache_folder=EMBED_PATH,
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024

documents = SimpleDirectoryReader("./example_data").load_data()

graph_store = Neo4jGraphStore(
    username=get_neo4j_credentails()["username"],
    password=get_neo4j_credentails()["password"],
    url=get_neo4j_credentails()["uri"],
    database=get_neo4j_credentails()["database"],
)
graph_store.query("MATCH (n) DETACH DELETE n")

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    include_embeddings=True,
    show_progress=True,    
)

query_engine1 = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
query_engine2 = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
query_engine3 = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=3,
)

response = query_engine1.query("Tell me more about Interleaf")
display(Markdown(f"<b>{response}</b>"))

response = query_engine2.query(
    "Tell me more about what the author worked on at Interleaf"
)
display(Markdown(f"<b>{response}</b>"))

response = query_engine3.query(
    "Tell me more about what the author worked on at Interleaf"
)
display(Markdown(f"<b>{response}</b>"))
