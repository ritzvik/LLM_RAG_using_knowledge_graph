# https://cookbook.openai.com/examples/rag_with_graph_db
# https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146

import os
from typing import Dict, List
import torch
import json 
import pandas as pd

from neo4j_utils.neo4j_server import reset_neo4j_server, get_neo4j_credentails, wait_for_neo4j_server, is_neo4j_server_up

from langchain.graphs import Neo4jGraph
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.vectorstores.neo4j_vector import Neo4jVector
from huggingface_hub import hf_hub_download, snapshot_download

from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline
)
from transformers import BitsAndBytesConfig

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

supported_embed_models = ["thenlper/gte-large"]

supported_llm_models = {
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    "microsoft/Phi-3-mini-4k-instruct-gguf": "Phi-3-mini-4k-instruct-q4.gguf",
}

"""
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
"""

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
        text_node_properties=['value'],
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
    
    If there are no relevant entities in the user prompt, return an empty json object. The output should contain just the json object, no additional text.
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

mistral_llm = LlamaCpp(
    model_path=get_model_path(model_name),
    n_gpu_layers=n_gpu_layers,
    temperature=temperature,
    n_ctx=context_window,
    f16_kv=True,
    verbose=True,
)

def parse_json_garbage(s) -> Dict[str, str]:
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])

def define_query(prompt_text: str) -> Dict[str, str]:
    text_for_llm = system_prompt.replace("{question}", prompt_text)
    response = parse_json_garbage(mistral_llm.invoke(text_for_llm))
    return response

example_queries = [
    "Which pink items are suitable for children?",
    "Help me find gardening gear that is waterproof",
    "I'm looking for a bench with dimensions 100x50 for my living room"
]

for q in example_queries:
    print(f"Q: '{q}'\n{define_query(q)}\n")

def create_embedding(text)->List[float]:
    result = embedding.embed_query(text)
    return result

def create_query(query_dict: Dict[str, str], threshold=0.81):
    # Creating embeddings
    embeddings_data = []
    for key, val in query_dict.items():
        if key != 'product':
            embeddings_data.append(f"${key}Embedding AS {key}Embedding")
    query = "WITH " + ",\n".join(e for e in embeddings_data)
    # Matching products to each entity
    query += "\nMATCH (p:Product)\nMATCH "
    match_data = []
    for key, val in query_dict.items():
        if key != 'product':
            relationship = entity_relationship_match[key]
            match_data.append(f"(p)-[:{relationship}]->({key}Var:{key})")
    query += ",\n".join(e for e in match_data)
    similarity_data = []
    for key, val in query_dict.items():
        if key != 'product':
            similarity_data.append(f"gds.similarity.cosine({key}Var.embedding, ${key}Embedding) > {threshold}")
    query += "\nWHERE "
    query += " AND ".join(e for e in similarity_data)
    query += "\nRETURN p"
    return query

def query_graph(query_dict: Dict[str, str]):
    embeddingsParams = {}
    query = create_query(query_dict)
    for key, val in query_dict.items():
        embeddingsParams[f"{key}Embedding"] = create_embedding(val)
    result = graph.query(query, params=embeddingsParams)
    return result

example_response = {
    "category": "clothes",
    "color": "blue",
    "age_group": "adults"
}

query_graph(example_response)


# Adjust the relationships_threshold to return products that have more or less relationships in common
def query_similar_items(product_id, relationships_threshold = 3):
    
    similar_items = []
        
    # Fetching items in the same category with at least 1 other entity in common
    query_category = '''
            MATCH (p:Product {id: $product_id})-[:hasCategory]->(c:category)
            MATCH (p)-->(entity)
            WHERE NOT entity:category
            MATCH (n:Product)-[:hasCategory]->(c)
            MATCH (n)-->(commonEntity)
            WHERE commonEntity = entity AND p.id <> n.id
            RETURN DISTINCT n;
        '''
    

    result_category = graph.query(query_category, params={"product_id": int(product_id)})
    #print(f"{len(result_category)} similar items of the same category were found.")
          
    # Fetching items with at least n (= relationships_threshold) entities in common
    query_common_entities = '''
        MATCH (p:Product {id: $product_id})-->(entity),
            (n:Product)-->(entity)
            WHERE p.id <> n.id
            WITH n, COUNT(DISTINCT entity) AS commonEntities
            WHERE commonEntities >= $threshold
            RETURN n;
        '''
    result_common_entities = graph.query(query_common_entities, params={"product_id": int(product_id), "threshold": relationships_threshold})
    #print(f"{len(result_common_entities)} items with at least {relationships_threshold} things in common were found.")

    for i in result_category:
        similar_items.append({
            "id": i['n']['id'],
            "name": i['n']['name']
        })
            
    for i in result_common_entities:
        result_id = i['n']['id']
        if not any(item['id'] == result_id for item in similar_items):
            similar_items.append({
                "id": result_id,
                "name": i['n']['name']
            })
    return similar_items

product_ids = ['1519827', '2763742']

for product_id in product_ids:
    print(f"Similar items for product #{product_id}:\n")
    result = query_similar_items(product_id)
    print("\n")
    for r in result:
        print(f"{r['name']} ({r['id']})")
    print("\n\n")

def get_formatted_product_response(query_result)->List:
    response = []
    for r in query_result:
        response.append({
            "id": r['p']['id'],
            "name": r['p']['name']
        })
    return response

def query_db(params):
    # Querying the db
    result = query_graph(params)
    return get_formatted_product_response(result)   

def similarity_search(prompt, threshold=0.8):
    embedding = create_embedding(prompt)
    query = '''
            WITH $embedding AS inputEmbedding
            MATCH (p:Product)
            WHERE gds.similarity.cosine(inputEmbedding, p.embedding) > $threshold
            RETURN p
            '''
    result = graph.query(query, params={'embedding': embedding, 'threshold': threshold})
    return get_formatted_product_response(result)

prompt_similarity = "I'm looking for nice curtains"
print(similarity_search(prompt_similarity))

print(get_formatted_product_response(query_graph(define_query("Show me Adidas clothing for kids"))))
