from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from utils.cmlllm import supported_llm_models, supported_embed_models

MODELS_PATH = "./models"
EMBEDS_PATH = "./embed_models"

# supported_llm_models = {
#     "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
#     "microsoft/Phi-3-mini-4k-instruct-gguf": "Phi-3-mini-4k-instruct-q4.gguf",
# }

# supported_embed_models = ["thenlper/gte-large"]

for supported_llm_model in supported_llm_models:
    print(
        f"download model {supported_llm_model} file {supported_llm_models[supported_llm_model]}"
    )
    hf_hub_download(
        repo_id=supported_llm_model,
        filename=supported_llm_models[supported_llm_model],
        resume_download=True,
        cache_dir=MODELS_PATH,
        local_files_only=False,
    )


for embed_model in supported_embed_models:
    print(f"download embed model {embed_model}")
    snapshot_download(
        repo_id=embed_model,
        resume_download=True,
        cache_dir=EMBEDS_PATH,
        local_files_only=False,
    )
