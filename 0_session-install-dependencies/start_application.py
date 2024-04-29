import os
import cmlapi
import json
from utils.check_dependency import check_gpu_enabled

client = cmlapi.default_client(
    url=os.getenv("CDSW_API_URL").replace("/api/v1", ""),
    cml_api_key=os.getenv("CDSW_APIV2_KEY"),
)
available_runtimes = client.list_runtimes(
    search_filter=json.dumps(
        {"kernel": "Python 3.10", "edition": "Nvidia GPU", "editor": "PBJ Workbench"}
    )
)
print(available_runtimes)

## Set available runtimes to the latest runtime in the environment (iterator is the number that begins with 0 and advances sequentially)
## The JOB_IMAGE_ML_RUNTIME variable stores the ML Runtime which will be used to launch the job
print(available_runtimes.runtimes[0])
print(available_runtimes.runtimes[0].image_identifier)
APP_IMAGE_ML_RUNTIME = available_runtimes.runtimes[0].image_identifier

## Store the ML Runtime for any future jobs in an environment variable so we don't have to do this step again
os.environ["APP_IMAGE_ML_RUNTIME"] = APP_IMAGE_ML_RUNTIME
project = client.get_project(project_id=os.getenv("CDSW_PROJECT_ID"))


if check_gpu_enabled() == False:
    print("Start AI Chat with your documents without GPU")
    application_request = cmlapi.CreateApplicationRequest(
        name="AI Chat with your documents",
        description="AI Chat with your documents",
        project_id=project.id,
        subdomain="ai-chat-with-doc",
        script="3_app-run-python-script/front_end_app.py",
        cpu=6,
        memory=24,
        runtime_identifier=os.getenv("APP_IMAGE_ML_RUNTIME"),
        bypass_authentication=True,
        environment={"CML": "yes", "TOKENIZERS_PARALLELISM": "false"},
    )

else:
    print("Start AI Chat with your documents with GPU")
    application_request = cmlapi.CreateApplicationRequest(
        name="AI Chat with your documents",
        description="AI Chat with your documents",
        project_id=project.id,
        subdomain="ai-chat-with-doc",
        script="3_app-run-python-script/front_end_app.py",
        cpu=2,
        memory=16,
        nvidia_gpu=1,
        runtime_identifier=os.getenv("APP_IMAGE_ML_RUNTIME"),
        bypass_authentication=True,
        environment={"CML": "yes", "TOKENIZERS_PARALLELISM": "false"},
    )

app = client.create_application(project_id=project.id, body=application_request)
