# Easy_LoRA
Implementing simple model retraining using LoRA
This project contains scripts for fine-tuning the Qwen2.5-1.5B-Instruct model on 1C:Enterprise.Element data using LoRA (Low-Rank Adaptation) technique, along with an interactive chat interface for testing the trained model.

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
```
python train_LoRA.py
```
5. Start Chat Interface
bash
```
python chat_LoRA.py
```
*🛠 Technical Details*
Model
Base Model: Qwen/Qwen2.5-1.5B-Instruct

Fine-tuning Type: LoRA (Low-Rank Adaptation)

LoRA Parameters:

r: 4-16 (depending on mode)

alpha: 32

dropout: 0.05

target_modules: all projection layers

Optimization
Learning rate: 5e-5 to 3e-4 (depending on mode)

Batch size: 1-4 (with gradient accumulation)

Context length: 256-1024 tokens

Regularization: weight decay 0.01, gradient clipping 1.0

Generation
Temperature: 0.1-0.4 for different answer types

Top-p sampling: 0.7-0.9

Repetition penalty: 1.15-1.3

Beam search: up to 5 beams for technical answers

💡 Recommendations
For Training:
Use at least 100+ examples for quality training

"BALANCED" mode is recommended for most tasks

Check quality on test prompts after training

Save different model versions for comparison

For Usage:
For technical questions, use keywords: "code", "syntax", "how to make"

For explanations: "what is", "explain", "tell about"

The system automatically detects question type and selects optimal generation parameters

⚠️ Limitations
Training requires significant computational resources (minimum 16GB RAM recommended)

Model trained only on Russian language

Focus on 1C:Enterprise.Element topic

Check base model license for commercial use

🔧 Troubleshooting
Problem: "Out of memory"
Solution: Decrease batch_size or max_length in settings

Problem: Model generates meaningless text
Solution:

Increase training data amount

Use "QUALITY" mode

Decrease learning rate

Problem: Answers are too short
Solution: Increase max_new_tokens in generation function

📈 Quality Assessment
The script automatically tests the model on the following prompts:

"What is 1C:Enterprise.Element?"

"How to create a new application?"

"Tell about the control panel"

Element language code examples

Syntax comparative analysis
