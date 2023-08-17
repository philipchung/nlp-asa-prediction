#!/usr/bin/env bash

# Maintainer: Philip Chung
# Date: 11-25-2022

# The following script installs pyenv and poetry on an Azure Compute Instance.
# Pyenv gives us ability to control python version despite other python versions
# installed on this machine (e.g. system, conda, etc.).  
# Poetry is used for package dependency resolution and virtualenv creation.

### --- Script Defaults & Debug Settings --- ###
# If script fails, exit script
set -o errexit
# If access unset variable, fail script & exit (deactivate since we will access $CONDA_SHLVL)
# set -o nounset
# If pipe command fails, exit script
set -o pipefail
# Trace for debugging
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace;
fi

### --- Deactivate Conda --- ###
# Azure Compute Instances are docker containers that have preconfigured conda environments
# that are used by default.  We deactivate all conda environments so we don't use a
# version of python managed by conda.
eval "$(conda shell.bash hook)"
for i in $(seq ${CONDA_SHLVL}); do
    conda deactivate
done

### --- Pyenv --- ###
# Install pyenv dependencies
sudo apt-get update && sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git -y

# Install pyenv to install & manage different versions of python on machine
curl https://pyenv.run | bash
# Configure bashrc to load pyenv automatically
echo '
### --- pyenv --- ###
# Add pyenv to $PATH
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
### --- pyenv --- ###
' >> ~/.bashrc

# Install a specific Python Version with pyenv & Use it Globally.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
export GLOBAL_PYTHON_VERSION="3.10.10"
pyenv install ${GLOBAL_PYTHON_VERSION}
pyenv global ${GLOBAL_PYTHON_VERSION}

### --- Poetry --- ###
# Install Poetry using the global python version managed by pyenv
curl -sSL https://install.python-poetry.org | ${PYENV_ROOT}/versions/${GLOBAL_PYTHON_VERSION}/bin/python -
# Add Poetry installation to $PATH
export PATH="/home/azureuser/.local/bin:$PATH"

# Environment Variables
# Azure CloudFiles are mounted & symlinked as a dir within $HOME.  This path is referenced as CLOUDFILES_HOME
# Note: default user in Azure Compute Instance is `azureuser`, not USERNAME.
export USERNAME="chungph"
export PROJECT_NAME="presurgnlp-az"
export CLOUDFILES_HOME="/home/azureuser/cloudfiles/code/Users/${USERNAME}"
export PROJECT_PATH="${CLOUDFILES_HOME}/${PROJECT_NAME}"

# Go to Project Directory
cd ${PROJECT_PATH}
# Use pyenv to install & specify local python version (only used in this Directory)
LOCAL_PYTHON_VERSION="3.10.10"
if [[ "${LOCAL_PYTHON_VERSION}" != "${GLOBAL_PYTHON_VERSION}" ]]; then
    pyenv install ${LOCAL_PYTHON_VERSION};
fi
pyenv local ${LOCAL_PYTHON_VERSION}
# Have poetry pick-up local python version
poetry env use python

# Install Python Packages with Poetry
poetry install

# Note: Poetry currently cannot handle the --extra-index-url command which allows
# specifying different python wheels that target different CUDA versions.
# - The version of pytorch in pyproject.toml & poetry.lock uses CUDA 10.2.
# - The Azure Compute Instance has CUDA Version 11.4 (view version with `nvidia-smi` command)
# - To use pytorch with CUDA 11.3, we use Poe the Poet to run a predefined
# install command, which is defined in pyproject.toml under [tool.poe.tasks]
# If `poetry install` or `poetry update` is called, this line needs to be re-run.
poetry run poe force-pytorch-cuda113
# poetry run poe install-ray-tune210

# Export environment as standard requirements.txt using pip
poetry run pip freeze > requirements.txt

# Activate Poetry Shell
# poetry shell
# Note: If using VSCode, to make the poetry virtual environment discoverable, you must add 
# the default poetry virtual env location to the settings 
# {"python.venvPath": "~/.cache/pypoetry/virtualenvs"}.
# Then you can use the /bin/python in this virtualenv as a jupyter kernel.

# Configure .bashrc to go to project directory
echo '
### --- deactivate conda by default --- ###
conda config --set auto_activate_base false
### --- deactivate conda by default --- ###

### --- poetry custom config --- ###

# Add Poetry Installation to PATH
export PATH="/home/azureuser/.local/bin:$PATH"

# Navigate to project directory
export USERNAME='"${USERNAME}"'
export PROJECT_NAME='"${PROJECT_NAME}"'
export CLOUDFILES_HOME='"${CLOUDFILES_HOME}"'
export PROJECT_PATH='"${PROJECT_PATH}"'
cd ${PROJECT_PATH}
### --- poetry custom config --- ###
' >> ~/.bashrc

# Restart Shell
exec "$SHELL"