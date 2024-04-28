import os
import requests


# Check that the current workspace allows workloads to use GPUs
def check_gpu_enabled():
    APIv1 = os.getenv("CDSW_API_URL")
    PATH = "site/config/"
    API_KEY = os.getenv("CDSW_API_KEY")

    url = "/".join([APIv1, PATH])
    res = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        auth=(API_KEY, ""),
    )
    max_gpu_per_engine = res.json().get("max_gpu_per_engine")

    if max_gpu_per_engine < 1:
        print("GPU's are not enabled for this workspace")
        return False
    print("GPUs are enabled in this workspace.")
    return True


def check_unauthenticated_access_to_app_enabled():
    APIv1 = os.getenv("CDSW_API_URL")
    PATH = "site/config/"
    API_KEY = os.getenv("CDSW_API_KEY")

    url = "/".join([APIv1, PATH])
    res = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        auth=(API_KEY, ""),
    )
    allow_unauthenticated_access_to_app = res.json().get(
        "allow_unauthenticated_access_to_app"
    )

    return allow_unauthenticated_access_to_app


# # Check that there are available GPUs or autoscalable GPUs available
# def check_gpu_launch():
#     # Launch a worker that uses GPU to see if any gpu is available or autoscaling is possible
#     worker = cdsw.launch_workers(
#         n=1, cpu=2, memory=4, nvidia_gpu=1, code="print('GPU Available')"
#     )

#     # Wait for 10 minutes to see if worker pod reaches success state
#     worker_schedule_status = cdsw.await_workers(
#         worker, wait_for_completion=True, timeout_seconds=600
#     )
#     if len(worker_schedule_status["failures"]) == 1:
#         cdsw.stop_workers(worker_schedule_status["failures"][0]["id"])
#         # Failure at this point is due to not enough GPU resources at the time of launch.
#         # Ask your admin about quota and autoscaling rules for GPU
#         print("Unable to allocate GPU resource within 10 minutes")
#         return False

#     print("Launched workers successfully. Shutting down the worker...")
#     cdsw.stop_workers()
#     # Wait for 10 minutes to see if worker pod reaches success state
#     worker_schedule_status = cdsw.await_workers(
#         worker, wait_for_completion=True, timeout_seconds=600
#     )
#     if len(worker_schedule_status["failures"]) == 1:
#         cdsw.stop_workers(worker_schedule_status["failures"][0]["id"])
#         # Failure at this point is due to not enough GPU resources at the time of launch.
#         # Ask your admin about quota and autoscaling rules for GPU
#         print("Unable to shutdown workers within 10 minutes")
#         return False

#     return True


if __name__ == "__main__":
    print("Checking the enablement and availibility of GPU in the workspace")
    check_gpu_enabled()
    # check_gpu_launch()
