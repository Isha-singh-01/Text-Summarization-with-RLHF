## Dialogue Summarization with Flan-T5 (Instruction-Tuned, PEFT, RLHF Detoxification)

This project demonstrates the development and optimization of a Flan-T5 model for dialogue summarization, integrating instruction-based fine-tuning, parameter-efficient fine-tuning (PEFT), and Reinforcement Learning with Human Feedback (RLHF) for output detoxification.

## Project Overview

The goal of this project is to generate concise, high-quality summaries of dialogue while minimizing toxicity in outputs. The workflow progresses through three main stages:

1. **Instruction-Based Fine-Tuning**  
   - Fine-tuned Flan-T5 on dialogue datasets using instruction prompts.  
   - Achieved improved ROUGE scores over zero-shot baselines (ROUGE-1 +18%, ROUGE-2 +10%, ROUGE-L +13%).

2. **Parameter-Efficient Fine-Tuning (PEFT) with LoRA**  
   - Reduced trainable parameters to 1.4% while maintaining near-original performance (ROUGE drop ≤1.7%).  
   - Enabled scalable training and inference on single CPU/GPU environments.  

3. **RLHF Detoxification**  
   - Applied Proximal Policy Optimization (PPO) with a hate speech reward model to minimize toxicity in generated summaries.  
   - Used KL-divergence with a reference model to prevent reward hacking and maintain semantic alignment.  
   - Evaluated models using quantitative ROUGE metrics and mean toxicity scores, as well as qualitative pre/post-detoxification comparisons.

## Features

- Automated data preprocessing, tokenization, and instruction prompt wrapping.  
- Integration of PEFT and LoRA for efficient training.  
- Toxicity evaluation pipeline leveraging Hugging Face `transformers` and Facebook RoBERTa hate speech classifier.  
- PPO-based RLHF for detoxification, ensuring safer outputs without compromising summary quality.  
- Evaluation of model performance both quantitatively (ROUGE, toxicity scores) and qualitatively (human review).

## Tech Stack

- **Python 3.9+**  
- **PyTorch**  
- **Hugging Face Transformers** (`Flan-T5`, `AutoModelForSeq2SeqLM`, `AutoModelForSequenceClassification`)  
- **PEFT + LoRA**  
- **TRL** (for PPO and RLHF training)  
- **Evaluate** (ROUGE & toxicity metrics)  
- **tqdm** (progress bars for long-running jobs)  
