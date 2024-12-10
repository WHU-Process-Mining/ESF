# A Two-Stage Predictive Process Monitoring Approach with Enabled State Filtering for High-Variability Suffixes

## Installation

1. Create a python environment

    ```bash
    conda create -n ESF python=3.8.0
    conda activate ESF 
    ```

2. Install pytorch

    Following the official website's guidance (<https://pytorch.org/get-started/locally/>), install the corresponding PyTorch version based on your CUDA version. For our experiments, we use torch 1.12.1+cu116. The installation command is as follows:

    ```bash
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
    ```

3. Install other related dependencies

    ```bash
    pip install -r requirements.txt
    ```

## Train
First, you need to specify data_path and dataset in configs/ESF_Model.yaml. 

Here, Two training methods are provided here:

1. Specify Hyperparameters:
    Specify model_parameters in   configs/ESF_Model.yaml.
    ```bash
    python execute/ESF/train_ESF.py
    ```
2. Use Optuna for Hyperparameter Optimization:
    ```bash
    python execute/ESF/run_ESF.py
    ```
## Test
    ```bash
    python execute/ESF/test_ESF.py
    ```

