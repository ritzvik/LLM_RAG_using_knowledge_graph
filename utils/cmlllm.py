import os
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.readers.file import UnstructuredReader, PDFReader
from llama_index.readers.nougat_ocr import PDFNougatOCR
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from huggingface_hub import hf_hub_download, snapshot_download
import time
import torch
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from utils.duplicate_preprocessing import DuplicateRemoverNodePostprocessor
import torch
import logging
import sys
import subprocess
import gradio as gr
import atexit
from neo4j_utils.neo4j_server import get_neo4j_service_name, get_current_namespace
import utils.vectordb as vectordb
from llama_index.core.memory import ChatMemoryBuffer
from dotenv import load_dotenv
from llama_index.core.chat_engine import ContextChatEngine

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

QUESTIONS_FOLDER = "questions"


def exit_handler():
    print("cmlllmapp is exiting!")
    vectordb.stop_vector_db()


atexit.register(exit_handler)


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])

supported_embed_models = ["thenlper/gte-large"]


def get_supported_embed_models():
    embedList = list(supported_embed_models)
    return embedList


supported_llm_models = {
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    "microsoft/Phi-3-mini-4k-instruct-gguf": "Phi-3-mini-4k-instruct-q4.gguf",
}
chat_engine_map = {}


def get_supported_models():
    llmList = list(supported_llm_models)
    return llmList


active_collection_available = {"default_collection": False}


def get_active_collections():
    return list(active_collection_available)


print("resetting the questions")
print(subprocess.run([f"rm -rf {QUESTIONS_FOLDER}"], shell=True))

milvus_start = vectordb.reset_vector_db()
print(f"milvus_start = {milvus_start}")


def infer2(msg, history, collection_name):
    query_text = msg
    print(f"query = {query_text}, collection name = {collection_name}")

    if len(query_text) == 0:
        yield "Please ask some questions"
        return

    if (
        collection_name in active_collection_available
        and active_collection_available[collection_name] != True
    ):
        yield "No documents are processed yet. Please process some documents.."
        return

    if collection_name not in chat_engine_map:
        yield f"Chat engine not created for collection {collection_name}.."
        return

    chat_engine = chat_engine_map[collection_name]

    try:
        streaming_response = chat_engine.stream_chat(query_text)
        generated_text = ""
        for token in streaming_response.response_gen:
            generated_text = generated_text + token
            yield generated_text
    except Exception as e:
        op = f"failed with exception {e}"
        print(op)
        yield op


class CMLLLM:
    MODELS_PATH = "./models"
    EMBED_PATH = "./embed_models"

    questions_folder = QUESTIONS_FOLDER

    def __init__(
        self,
        model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        embed_model_name="thenlper/gte-large",
        temperature=0.0,
        max_new_tokens=256,
        context_window=3900,
        gpu_layers=20,
        dim=1024,
        collection_name="default_collection",
        memory_token_limit=3900,
        sentense_embedding_percentile_cutoff=0.8,
        similarity_top_k=2,
        progress=gr.Progress(),
    ):
        if len(model_name) == 0:
            model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        if len(embed_model_name) == 0:
            embed_model_name = "thenlper/gte-large"
        self.active_model_name = model_name
        self.active_embed_model_name = embed_model_name
        n_gpu_layers = 0
        if torch.cuda.is_available():
            print("It is a GPU node, setup GPU.")
            n_gpu_layers = gpu_layers

        self.node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=128)

        progress((1, 4), desc="setting the global parameters")

        self.set_global_settings(
            model_name=model_name,
            embed_model_path=embed_model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            n_gpu_layers=n_gpu_layers,
            node_parser=self.node_parser,
            progress=progress,
        )
        progress((1.5, 4), desc="done, setting the global parameters")
        self.dim = dim
        self.similarity_top_k = similarity_top_k
        self.sentense_embedding_percentile_cutoff = sentense_embedding_percentile_cutoff
        self.memory_token_limit = memory_token_limit

    def get_active_model_name(self):
        print(f"active model is {self.active_model_name}")
        return self.active_model_name

    def get_active_embed_model_name(self):
        print(f"active embed model is {self.active_embed_model_name}")
        return self.active_embed_model_name

    def delete_collection_name(self, collection_name, progress=gr.Progress()):
        print(f"delete_collection_name : collection = {collection_name}")

        if collection_name is None or len(collection_name) == 0:
            return None

        active_collection_available.pop(collection_name, None)
        chat_engine_map.pop(collection_name, None)
        progress((1, 1), desc=f"Successfully deleted the collection {collection_name}")

    def set_collection_name(
        self,
        collection_name,
        progress=gr.Progress(),
    ):
        print(f"set_collection_name : collection = {collection_name}")

        if collection_name is None or len(collection_name) == 0:
            return None

        print(f"adding new collection name {collection_name}")

        if not collection_name in active_collection_available:
            active_collection_available[collection_name] = False

        if collection_name in chat_engine_map:
            print(
                f"collection {collection_name} is already configured and chat_engine is set"
            )
            progress(
                (1, 1),
                desc=f"collection {collection_name} is already configured and chat_engine is set.",
            )
            return

        progress(
            (1, 4),
            desc=f"creating or getting the vector db collection {collection_name}",
        )

        progress((2, 4), desc="setting the vector db")

        vector_store = MilvusVectorStore(
            dim=self.dim,
            collection_name=collection_name,
        )
        vector_store = Neo4jVectorStore(
            username="neo4j",
            password="password",
            uri=f"bolt://{get_neo4j_service_name()}.{get_current_namespace()}.svc.cluster.local:7687",
            embedding_dimension=self.dim,
            
        )

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        progress((3, 4), desc="setting the chat engine")

        chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            verbose=True,
            postprocessor=[
                SentenceEmbeddingOptimizer(
                    percentile_cutoff=self.sentense_embedding_percentile_cutoff
                ),
                DuplicateRemoverNodePostprocessor(),
            ],
            memory=ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit),
            system_prompt=(
                "You are an expert Q&A assistant that is trusted around the world.\n"
                "Always answer the query using the Context provided and not prior knowledge or General knowledge."
                "Avoid statements like 'Based on the context' or 'The context information'.\n"
                "If the provided context dont have the information, answer 'I dont know'.\n"
                "Please cite the source along with your answers."
            ),
            similarity_top_k=self.similarity_top_k,
        )
        progress(
            (4, 4),
            desc=f"successfully updated the chat engine for the collection name {collection_name}",
        )

        chat_engine_map[collection_name] = chat_engine

    def ingest(self, files, questions, collection_name, progress=gr.Progress()):
        if not (collection_name in active_collection_available):
            return f"Some issues with the llm and colection {collection_name} setup. please try setting up the llm and the vector db again."

        file_extractor = {
            ".html": UnstructuredReader(),
            ".pdf": PDFReader(),
            ".txt": UnstructuredReader(),
        }

        if torch.cuda.is_available():
            file_extractor[".pdf"] = PDFNougatOCR()

        print(f"collection = {collection_name}, questions = {questions}")

        progress(0.3, desc="loading the documents")

        filename_fn = lambda filename: {"file_name": os.path.basename(filename)}

        active_collection_available[collection_name] = False

        try:
            start_time = time.time()
            op = "Questions\n"
            i = 1
            for file in files:
                progress(0.4, desc=f"loading document {os.path.basename(file)}")
                reader = SimpleDirectoryReader(
                    input_files=[file],
                    file_extractor=file_extractor,
                    file_metadata=filename_fn,
                )
                document = reader.load_data(num_workers=1, show_progress=True)

                progress(0.4, desc=f"done loading document {os.path.basename(file)}")

                vector_store = MilvusVectorStore(
                    dim=self.dim,
                    collection_name=collection_name,
                )

                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                )

                progress(
                    0.4, desc=f"start indexing the document {os.path.basename(file)}"
                )
                nodes = self.node_parser.get_nodes_from_documents(document)

                index = VectorStoreIndex(
                    nodes, storage_context=storage_context, show_progress=True
                )

                progress(
                    0.4, desc=f"done indexing the document {os.path.basename(file)}"
                )

                ops = (
                    "Completed data ingestion. took "
                    + str(time.time() - start_time)
                    + " seconds."
                )

                print(f"{ops}")
                progress(0.4, desc=op)

                start_time = time.time()
                print(
                    f"start dataset generation from the document {os.path.basename(file)}."
                )
                progress(
                    0.4,
                    desc=f"start dataset generation from the document {os.path.basename(file)}.",
                )

                data_generator = DatasetGenerator.from_documents(documents=document)

                dataset_op = (
                    f"Completed data set generation for file {os.path.basename(file)}. took "
                    + str(time.time() - start_time)
                    + " seconds."
                )
                print(f"{dataset_op}")
                progress(0.4, desc=dataset_op)
                print(
                    f"start generating questions from the document {os.path.basename(file)}"
                )
                progress(
                    0.4,
                    desc=f"generating questions from the document {os.path.basename(file)}",
                )
                eval_questions = data_generator.generate_questions_from_nodes(
                    num=questions
                )

                for q in eval_questions:
                    op += str(q) + "\n"
                    i += 1

                print(
                    f"done generating questions from the document {os.path.basename(file)}"
                )
                progress(
                    0.4,
                    desc=f"done generating questions from the document {os.path.basename(file)}",
                )
                print(subprocess.run([f"rm -f {file}"], shell=True))

            progress(0.9, desc=f"done processing the documents {collection_name}...")
            print(f"done processing the documents {collection_name}...")
            active_collection_available[collection_name] = True

        except Exception as e:
            print(e)
            ops = f"ingestion failed with exception {e}"
            progress(0.9, desc=ops)
        return op

    def upload_document_and_ingest(self, files, questions, progress=gr.Progress()):
        if len(files) == 0:
            return "Please add some files..."
        return self.ingest(files, questions, progress)

    def set_global_settings(
        self,
        model_name,
        embed_model_path,
        temperature,
        max_new_tokens,
        context_window,
        n_gpu_layers,
        node_parser,
        progress=gr.Progress(),
    ):
        self.set_global_settings_common(
            model_name=model_name,
            embed_model_path=embed_model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            n_gpu_layers=n_gpu_layers,
            progress=progress,
        )

        Settings.callback_manager = callback_manager
        Settings.node_parser = node_parser

    def set_global_settings_common(
        self,
        model_name,
        embed_model_path,
        temperature,
        max_new_tokens,
        context_window,
        n_gpu_layers,
        progress,
    ):
        print(
            f"Enter set_global_settings_common. model_name = {model_name}, embed_model_path = {embed_model_path}"
        )
        self.active_model_name = model_name
        self.active_embed_model_name = embed_model_path
        model_path = self.get_model_path(model_name)
        print(f"model_path = {model_path}")
        progress(0.1, f"Starting the model {model_path}")

        Settings.llm = LlamaCPP(
            model_path=model_path,
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
        progress(0.3, f"Setting the embed model {embed_model_path}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_path,
            cache_folder=self.EMBED_PATH,
        )

    def get_model_path(self, model_name):
        filename = supported_llm_models[model_name]
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            resume_download=True,
            cache_dir=self.MODELS_PATH,
            local_files_only=False,
            token=hf_token,
        )
        return model_path

    def get_embed_model_path(self, embed_model):
        embed_model_path = snapshot_download(
            repo_id=embed_model,
            resume_download=True,
            cache_dir=self.EMBEDS_PATH,
            local_files_only=False,
            token=hf_token,
        )
        return embed_model_path

    def clear_chat_engine(self, collection_name):
        if collection_name in chat_engine_map:
            chat_engine = chat_engine_map[collection_name]
            chat_engine.reset()
