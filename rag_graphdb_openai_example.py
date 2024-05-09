# https://cookbook.openai.com/examples/rag_with_graph_db
# https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146

import os
import torch
import json 
import pandas as pd

from neo4j_utils.neo4j_server import reset_neo4j_server, get_neo4j_credentails, wait_for_neo4j_server, is_neo4j_server_up

from langchain.graphs import Neo4jGraph
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.vectorstores.neo4j_vector import Neo4jVector
from huggingface_hub import hf_hub_download, snapshot_download

from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline
)
from transformers import BitsAndBytesConfig

hf_token="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"
os.environ['HF_TOKEN'] = hf_token

MODELS_PATH = "./models"
EMBED_PATH = "./embed_models"

model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
embed_model_name="thenlper/gte-large"

supported_embed_models = ["thenlper/gte-large"]

supported_llm_models = {
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    "microsoft/Phi-3-mini-4k-instruct-gguf": "Phi-3-mini-4k-instruct-q4.gguf",
}


# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

from dotenv import load_dotenv
load_dotenv()

if not is_neo4j_server_up():
    reset_neo4j_server()
    wait_for_neo4j_server()

file_path = './data/amazon_product_kg.json'

with open(file_path, 'r') as file:
    jsonData = json.load(file)

df =  pd.read_json(file_path)
df.head()

graph = Neo4jGraph(
    username=get_neo4j_credentails()["username"],
    password=get_neo4j_credentails()["password"],
    url=get_neo4j_credentails()["uri"],
)
graph.query("MATCH (n) DETACH DELETE n")

def sanitize(text):
    text = str(text).replace("'","").replace('"','').replace('{','').replace('}', '')
    return text

i = 1
for obj in jsonData:
    print(f"{i}. {obj['product_id']} -{obj['relationship']}-> {obj['entity_value']}")
    i+=1
    query = f'''
        MERGE (product:Product {{id: {obj['product_id']}}})
        ON CREATE SET product.name = "{sanitize(obj['product'])}", 
                       product.title = "{sanitize(obj['TITLE'])}", 
                       product.bullet_points = "{sanitize(obj['BULLET_POINTS'])}", 
                       product.size = {sanitize(obj['PRODUCT_LENGTH'])}

        MERGE (entity:{obj['entity_type']} {{value: "{sanitize(obj['entity_value'])}"}})

        MERGE (product)-[:{obj['relationship']}]->(entity)
        '''
    graph.query(query)


embedding = HuggingFaceEmbeddings(model_name=embed_model_name, cache_folder=EMBED_PATH)

vector_index = Neo4jVector.from_existing_graph(
    embedding=embedding,
    url=get_neo4j_credentails()["uri"],
    username=get_neo4j_credentails()["username"],
    password=get_neo4j_credentails()["password"],
    index_name='products',
    node_label="Product",
    text_node_properties=['name', 'title'],
    embedding_node_property='embedding',
)

def embed_entities(entity_type):
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embedding,
        url=get_neo4j_credentails()["uri"],
        username=get_neo4j_credentails()["username"],
        password=get_neo4j_credentails()["password"],
        index_name=entity_type,
        node_label=entity_type,
        text_node_properties=['name', 'title'],
        embedding_node_property='embedding',
    )

entities_list = df['entity_type'].unique()


for t in entities_list:
    embed_entities(t)


entity_types = {
    "product": "Item detailed type, for example 'high waist pants', 'outdoor plant pot', 'chef kitchen knife'",
    "category": "Item category, for example 'home decoration', 'women clothing', 'office supply'",
    "characteristic": "if present, item characteristics, for example 'waterproof', 'adhesive', 'easy to use'",
    "measurement": "if present, dimensions of the item", 
    "brand": "if present, brand of the item",
    "color": "if present, color of the item",
    "age_group": "target age group for the product, one of 'babies', 'children', 'teenagers', 'adults'. If suitable for multiple age groups, pick the oldest (latter in the list)."
}

relation_types = {
    "hasCategory": "item is of this category",
    "hasCharacteristic": "item has this characteristic",
    "hasMeasurement": "item is of this measurement",
    "hasBrand": "item is of this brand",
    "hasColor": "item is of this color", 
    "isFor": "item is for this age_group"
 }

entity_relationship_match = {
    "category": "hasCategory",
    "characteristic": "hasCharacteristic",
    "measurement": "hasMeasurement", 
    "brand": "hasBrand",
    "color": "hasColor",
    "age_group": "isFor"
}

system_prompt = f'''
    ### Context:
    You are a helpful agent designed to fetch information from a graph database. 
    
    The graph database links products to the following entity types:
    {json.dumps(entity_types)}
    
    Each link has one of the following relationships:
    {json.dumps(relation_types)}

    Depending on the user prompt, determine if it possible to answer with the graph database.
        
    The graph database can match products with multiple relationships to several entities.
    
    Example user input:
    "Which blue clothing items are suitable for adults?"
    
    There are three relationships to analyse:
    1. The mention of the blue color means we will search for a color similar to "blue"
    2. The mention of the clothing items means we will search for a category similar to "clothing"
    3. The mention of adults means we will search for an age_group similar to "adults"
    
    ### Instruction:
    Return a json object following the following rules:
    For each relationship to analyse, add a key value pair with the key being an exact match for one of the entity types provided, and the value being the value relevant to the user query.
    
    For the example provided, the expected output would be:
    {{
        "color": "blue",
        "category": "clothing",
        "age_group": "adults"
    }}
    
    If there are no relevant entities in the user prompt, return an empty json object.
'''

system_prompt += """

    ### Question:
    {question}
"""

print(system_prompt)

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


# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    # model_name,
    # quantization_config=bnb_config,
    token=hf_token,
)

text_generation_pipeline = pipeline(
    model=model,
    # tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=300,
    token=hf_token,
)
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
prompt_template = PromptTemplate(
    input_variables=["question"],
    template= system_prompt,
)
llm_chain = prompt_template | mistral_llm

def define_query(prompt_text):
    llm_chain.invoke(prompt_text)

example_queries = [
    "Which pink items are suitable for children?",
    "Help me find gardening gear that is waterproof",
    "I'm looking for a bench with dimensions 100x50 for my living room"
]

for q in example_queries:
    print(f"Q: '{q}'\n{define_query(q)}\n")