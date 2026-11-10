1. Environment Requirements
To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

# Create and activate conda environment
conda create -n CySAF python=3.8
conda activate CySAF
# Install required packages
pip install -r requirements.txt


2. Download Data
All the datasets needed for CySAF can be obtained from the [https://drive.google.com/drive/folders/1dfnzGafiaxo6BUsCMZbmlE0N6G5_yFqK].

Setup Instructions:

Create a folder named ./dataset in the project root
Place all the CSV files in this directory
Note: Place the CSV files directly into this directory (e.g., ./dataset/ETTh1.csv)


3. Training Example
You can reproduce the experiment results as the following examples:

    bash ./scripts/etth1.sh
    bash ./scripts/ettm2.sh