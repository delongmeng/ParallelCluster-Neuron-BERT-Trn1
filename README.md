## Distributed Large Language Model Pre-training Using ParallelCluster and AWS Trainium Instances

### Introduction

In this repo we show how to train the BERT large language model from scratch in a distributed environment using AWS ParallelCluster, an AWS supported open source cluster management tool that helps deploy and manage high performance computing (HPC) clusters in the AWS Cloud, with AWS Trn1 [Trainium](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html) instances and AWS [Neuron SDK](https://aws.amazon.com/machine-learning/neuron/). This repo largely follows this [Phase 1 BERT-Large pretraining tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/bert.html#phase-1-bert-large-pretrainingg).


### Create a Cluster Using ParallelCluster

- ParallelCluster can be installed following the instruction [here](https://docs.aws.amazon.com/parallelcluster/latest/ug/install-v3-virtual-environment.html).  

- A cluster can be configured and created following this [instruction](https://docs.aws.amazon.com/parallelcluster/latest/ug/install-v3-configuring.html). Here the configuration file `config/config_ml_neuron_trn1.yaml` was used to create a cluster using the following CLI command:

```
pcluster create-cluster --cluster-name myCluster --cluster-configuration config/config_ml_neuron_trn1.yaml
```

- Instances: This cluster has one head node with the instance type of `c5.4xlarge` and a compute queue with 16 so-called "dynamic" nodes of `trn1.32xlarge` instance type (a dynamic node only gets turned on when needed and automatically turned off when the job is finished). 

- Storage: here we use Amazon Elastic File System (EFS) for storage.

- Bootstrap configuration: A OnNodeConfigured script (see `scripts/install_neuron.sh`), stored in a S3 bucket, is used as a "custom action" in the bootstrap process. This script will be executed at the end of the instance bootstrap process to set up the environment needed for the model training, including:

  - Setting up Python virtual environment
  - Install Neuron SDK and EFA  

- SSM (or ssh) into the head node of the cluster.   

- In the head node, you can try simple Slurm commands such as:

```
sinfo
squeue
srun -N 1 hostname
srun -N 2 hostname
sbatch -N 1 --wrap "sleep 10"
sbatch -N 2 --wrap "sleep 10"
scontrol show job --details
```


### Pre-train the BERT Model

You can follow the steps in [this tutorial](https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/dp-bert-launch-job.md) for the training part. Briefly:

- Activate the virtual environment

```
source ~/aws_neuron_venv_pytorch/bin/activate
```

- Download scripts and training dataset. These scripts can also be found in the `scripts` directory in this repo.   

```
mkdir -p ~/examples/dp_bert_hf_pretrain
cd ~/examples/dp_bert_hf_pretrain
wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/dp_bert_hf_pretrain/run_dp_bert_large_hf_pretrain_bf16_s128.sh
chmod +x ./run_dp_bert_large_hf_pretrain_bf16_s128.sh
wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/dp_bert_hf_pretrain/dp_bert_large_hf_pretrain_hdf5.py
wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/dp_bert_hf_pretrain/requirements.txt
python3 -m pip install -r requirements.txt

mkdir -p ~/examples_datasets/
pushd ~/examples_datasets/
aws s3 cp s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar .  --no-sign-request
# 48 G

tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
# this takes ~15min

rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
popd
```

- Compile and train the model. On a multi-node distributed environment, we can use Slurm in the head node to run a job on multiple compute nodes. Here we use the `sbatch` command to submit a `srun` job and request 16 nodes. The `srun` job uses the `run_dp_bert_large_hf_pretrain_bf16_s128.sh` script, which uses `torchrun` (see reference [here](https://pytorch.org/docs/stable/elastic/run.html)) to execute the `dp_bert_large_hf_pretrain_hdf5.py` script with settings related to the distributed environment.

```
# Compile the model. This step takes around 30 minutes.
cd ~/examples/dp_bert_hf_pretrain
sbatch --exclusive --nodes=16 --wrap "srun neuron_parallel_compile ./run_dp_bert_large_hf_pretrain_bf16_s128.sh"
```
 
```
# Train the model. This step takes around 5 hours.
cd ~/examples/dp_bert_hf_pretrain
sbatch --exclusive --nodes=16 --wrap "srun ./run_dp_bert_large_hf_pretrain_bf16_s128.sh"
```


- Note that you can monitor the job progress by executing the `tail -f slurm-<job-id>.out` command. You can also ssh into a compute node and run `neuron-top` to monitor NeuronCore and vCPU utilization, memory usage, loaded models, and Neuron applications. More details can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-top-user-guide.html).


- If you are interested in how to use a pre-trained LLM model for inference with or without fine-tuning, please take a look at this repo: https://github.com/delongmeng/Machine-Translation-LLM-Finetuning.


### Troubleshooting

- If you get "Insufficient Capacity Errors" (ICEs) when creating the cluster, you may consider EC2 [capacity reservation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-using.html).

- The output of the OnNodeConfigured script for head node can be found here: `/var/log/cfn-init-cmd.log` and `/var/log/cfn-init.log`.

- The output of the OnNodeConfigured script for compute node can be found here: `/var/log/cloud-init-output.log`.

