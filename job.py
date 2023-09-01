# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential

# For Compute 
from azure.ai.ml.entities import AmlCompute

# For Job
from azure.ai.ml import command
from azure.ai.ml import Input

import yaml

# Load the config.yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# Assign the values to the respective variables
subscription_id = config['subscription_id']
resource_group_name = config['resource_group_name']
workspace_name = config['workspace_name']
gpu_compute_target = config['gpu_compute_target']
custom_env_name = config['custom_env_name']
azure_ds_path = config['azure_ds_path']
data_yaml_path = config['data']
experiment_name = config['project']
display_name = config['name']

credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
)

job = command(
    inputs = dict(
        data=Input(
            type = "uri_folder", 
            path = azure_ds_path
        ),
        data_yaml_path = data_yaml_path
    ),
    compute=gpu_compute_target,
    environment=custom_env_name,
    code=".",  # location of source code
    command="""
    echo "The data asset path is ${{ inputs.data }}" && 
    sed -i "s|path:.*$|path: ${{ inputs.data }}|" ${{ inputs.data_yaml_path }} &&
    python train.py
     """,
    experiment_name=experiment_name,
    display_name=display_name,
)
# for multi gpu, add the following in config.yaml
# --device 0,1,2,3 \
# --sync-bn \

ml_client.create_or_update(job)