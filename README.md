# A Multimodel Approach to Dermatology VQA

This project implements multimodal question-answering models using **BERT** and **DistilBERT** as language models, combined with different fusion mechanisms: **UNITER, Cross Attention, and ViLT**.

## Installation & Requirements

To set up the project, ensure you have the following dependencies installed:

### **1️ Install Python & Dependencies**
```bash
# Clone the repository
git clone git@github.com:asarthaks/FoundationModelsProject.git
cd FoundationModelsProject

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2️ Required Packages**
The main dependencies are:
- `torch`
- `transformers`
- `evaluate`
- `tqdm`
- `pandas`
- `numpy`

Ensure all required packages are installed using:
```bash
pip install torch transformers evaluate tqdm pandas numpy
```

## Running the Training Pipeline

### **Command Line Usage**
To train the model, use:
```bash
python train.py <LanguageModel> <FusionModel>
```
Where:
- `<LanguageModel>`: Choose between `BERT` or `DistilBERT`.
- `<FusionModel>`: Choose between `UNITER`, `CrossAttention`, or `ViLT`.

#### **Example Run**
```bash
python train.py DistilBERT UNITER
```
This command trains the model using **DistilBERT** as the language model and **UNITER** as the fusion mechanism.


## Results
The training logs and results are saved in the **results/** directory. After training, you can find:
- Model checkpoints in `models/`
- Detailed outputs in `results/outputDetails_<FusionModel>.txt`

#### **Example of Model Output:**
```
Predictions: "The lesion appears benign..."
References: "The lesion is benign..."
Questions: "What is the diagnosis of this lesion?"
Filename: ISIC_0024329
```

## Notes
- Ensure you have the dataset in the correct location (`data/` folder).
- The script assumes the dataset follows the `HAM_clean.csv` format.
- GPU support is enabled automatically if available.


---

