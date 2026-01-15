# ServImage

ServImage is a benchmark and model for evaluating the quality of generated images in real-world design tasks. It includes outputs from a wide range of mainstream text-to-image and image-to-image generation models, along with corresponding human annotations and task pricing information, enabling the study of generalization across different generators and varying levels of task complexity.

---

## Project Overview

- **ServImage_Bench**  
  An evaluation subset used for experiments in our paper, covering outputs from multiple mainstream image generation models (e.g., Flux, Gemini, Qwen, Stable Diffusion). Each sample is annotated with human quality labels.

- **ServImage_Dataset**  
  A full task-level dataset that includes task descriptions, pricing information (as a proxy for task complexity), and generated images, designed for model training and more comprehensive analysis.

- **ServImageModel**  
  A quality assessment model built on multimodal large language models such as Qwen3-VL, which determines whether a generated image satisfies the requirements of a given task (accept / reject).

---

## Clone the Repository

```bash
# Clone using a personal access token
# (replace YOUR_TOKEN with your GitHub Personal Access Token)
git clone https://YOUR_TOKEN@github.com/NAME/ServImage.git

# Enter the project directory
cd ServImage
---

## Data Structure

All data are organized under the ServImage/data/ directory:
	•	ServImage/data/ServImage_Bench/
	•	A benchmark image collection for evaluation, with subdirectories grouped by image generation models (e.g., flux-1.1-pro/, qwen-text-to-image/, etc.).
	•	ServImage/data/ServImage_Dataset/
	•	The full dataset, containing images and metadata organized by task categories (C-Digital / C-Portrait / C-Product).
	•	ServImage/data/gemini-banana-pro_input_with_label.jsonl
	•	An example annotation file in JSONL format, where each line is a JSON object recording the task ID, generation model, category, and human annotation results.
	•	ServImage/data/Label_data/
	•	Scripts and documentation for constructing different data splits, including standard splits, LOGO splits, and price-based splits (see readme.md for details).

Example annotation file (single-line JSON)

{
  "model": "qwen-text-to-image",
  "task_id": "portrait-001",
  "category": "C-Portrait",
  "annotations": {
    "subtask_1": true,
    "subtask_2": false
  }
}


## Environment Setup

We recommend using Conda to create an isolated environment and install the required dependencies.

```bash
# 1. Create a Conda environment
# (you may change the environment name and Python version as needed)
conda create -n servimage python=3.10 -y

# 2. Activate the environment
conda activate servimage

# 3. Install dependencies
# (make sure requirements.txt is located in the project root directory)
pip install -r requirements.txt



## Download the Model

Download the ServImageModel-4B-Think model.




## Training

Before training, make sure your data are properly prepared.

---

### Dataset Download

Download **ServImage_Dataset** from Hugging Face:

```bash
# 1. Log in to Hugging Face (requires an access token)
huggingface-cli login
# or (new CLI)
hf auth login

# 2. (Optional) Configure git credential helper to store credentials
git config --global credential.helper store

# 3. Clone the dataset repository into the data directory
cd ServImage/data
git clone https://huggingface.co/datasets/NAME/ServImage_Dataset

# 4. Install git-lfs (required for large files)
apt-get update && apt-get install git-lfs -y
git lfs install

# 5. Pull LFS files (image data)
cd ServImage_Dataset
git lfs pull

### Download model weights
cd /path/to/ServImage
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 \
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir weights/Qwen3-VL-2B-Instruct
  
1.	Qwen3-VL-2B-Instruct
2.	Qwen3-VL-4B-Instruct
3.	Qwen3-VL-4B-Thinking



### Stage 1 Training (5 epochs)

```bash
cd /workspace/workspace2/ServImage
bash scripts/train_stage1_4bthink.sh


### Stage 2 Training (2 epochs)

```bash
cd /workspace/workspace2/ServImage
export STAGE1_MODEL_PATH="output/ServImage_Model-think4b/checkpoint-3040"
bash scripts/train_stage2_4bthink.sh# ServImage
