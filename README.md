# Easy_LoRA
Implementing simple model retraining using LoRA
This project contains scripts for fine-tuning the Qwen2.5-1.5B-Instruct model on 1C:Enterprise.Element data using LoRA (Low-Rank Adaptation) technique, along with an interactive chat interface for testing the trained model.

📁 Project Structure
text
project/
├── train_LoRA.py          # Main script for model training
├── chat_LoRA.py           # Interactive chat interface script
├── result_LoRA.json       # JSON file with training data
├── models/                # Cache for downloaded models
├── qwen-lora-finetuned_2.0/  # Directory with saved model
└── README.md              # This file
🚀 Quick Start
1. Install Dependencies
bash
pip install torch transformers datasets peft accelerate
2. Prepare Data
Place your JSON file with training data in the following format:

json
```
{
  "data": [
    {
      "instruction": "User question",
      "input": "Context (optional)",
      "output": "Correct answer"
    }
  ]
}
```
3. Start Training
bash
python train_LoRA.py
4. Start Chat Interface
bash
python chat_LoRA.py
