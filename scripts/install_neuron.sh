#!/bin/bash
set -e

echo "[POST-INSTALL] Neuron SDK Release 2.6.0"
# Configure Linux for Neuron repository updates
. /etc/os-release

sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
echo "[POST-INSTALL] updating OS packages..."
sudo apt-get update -y

# Install git
echo "[POST-INSTALL] installing git..."
sudo apt-get install git -y

# Remove preinstalled packages and Install Neuron Driver and Runtime
echo "[POST-INSTALL] removing pre-installed packages..."
sudo apt-get remove aws-neuron-dkms -y
sudo apt-get remove aws-neuronx-dkms -y
sudo apt-get remove aws-neuronx-oci-hook -y
sudo apt-get remove aws-neuronx-runtime-lib -y
sudo apt-get remove aws-neuronx-collectives -y

echo "[POST-INSTALL] updating OS packages again..."
sudo apt-get update -y
echo "[POST-INSTALL] installing aws-neuronx packages..."
sudo apt-get install aws-neuronx-dkms=2.6.33.0 -y
sudo apt-get install aws-neuronx-oci-hook=2.1.14.0 -y
sudo apt-get install aws-neuronx-runtime-lib=2.10.30.0* -y
sudo apt-get install aws-neuronx-collectives=2.10.37.0* -y

# Install EFA Driver(only required for multiinstance training)
echo "[POST-INSTALL] installing EFA driver..."
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
cat aws-efa-installer.key | gpg --fingerprint
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig &&  gpg --verify ./aws-efa-installer-latest.tar.gz.sig

tar -xvf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && sudo bash efa_installer.sh --yes
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

# Remove pre-installed package and Install Neuron Tools
echo "[POST-INSTALL] removing pre-installed packages..."
sudo apt-get remove aws-neuron-tools  -y
sudo apt-get remove aws-neuronx-tools  -y
echo "[POST-INSTALL] installing aws-neuronx-tools..."
sudo apt-get install aws-neuronx-tools=2.6.1.0 -y

export PATH=/opt/aws/neuron/bin:$PATH

PYTHON_VERSION=$(python --version 2>&1)
echo "[POST-INSTALL] Python version: $PYTHON_VERSION" 

# Install Python venv and activate Python virtual environment to install
# Neuron pip packages.
echo "[POST-INSTALL] installing python3.8-venv..."
sudo apt install python3.8-venv -y

cd /home/ubuntu

. "/etc/parallelcluster/cfnconfig"
echo "[POST-INSTALL] the node type is $cfn_node_type"

if [[ $cfn_node_type == "HeadNode" ]]; then
  echo "[POST-INSTALL] Detected this is head node"
  python3.8 -m venv aws_neuron_venv_pytorch
  source aws_neuron_venv_pytorch/bin/activate
  pip install -U pip

  # Install packages from repos
  python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
  python -m pip install torch-neuronx=="1.12.0.1.4.0" neuronx-cc=="2.3.0.4" torchvision

  chown ubuntu:ubuntu -R aws_neuron_venv_pytorch
  echo "[POST-INSTALL] Finished setting up the head node"
else
  echo "[POST-INSTALL] Detected this is compute node"
  DNS_SERVER=""
  grep Ubuntu /etc/issue &>/dev/null && DNS_SERVER=$(resolvectl dns | awk '{print $4}' | sort -r | head -1)
  IP="$(host $HOSTNAME $DNS_SERVER | tail -1 | awk '{print $4}')"
  DOMAIN=$(jq .cluster.dns_domain /etc/chef/dna.json | tr -d \")
  sudo sed -i "/$HOSTNAME/d" /etc/hosts
  sudo bash -c "echo '$IP $HOSTNAME.${DOMAIN::-1} $HOSTNAME' >> /etc/hosts"
  echo "[POST-INSTALL] Finished setting up the compute node"
fi

