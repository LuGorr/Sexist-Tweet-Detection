# Sexism Detection Projects: NLP Assignments 1 & 2

This repository contains two distinct approaches to the task of sexism detection in social media text (tweets), developed as part of the Master’s Degree in Artificial Intelligence at the University of Bologna.

---

## Project 1: Traditional Deep Learning & Transformers
**Focus:** Evaluating LSTM architectures and fine-tuning encoder-based Transformer models.

### System Overview
* **Task:** Four-class sexism classification.
* **Models Evaluated:**
    * **Baselines:** Single-layer LSTM and Stacked LSTM using pretrained GloVe Twitter embeddings.
    * **Transformer A:** Fine-tuned `twitter-roberta-base-hate`.
    * **Transformer B:** Fine-tuned `bertweet-large-sexism-detector` (task-specific).
    * **Hierarchical Classifier** A binary classifier first separates non-sexist from sexist text, followed by a multi-class classifier distinguishing among intents.

### Key Findings
* **Best Performance:** Both the task-specific **Transformer B** and the **Hierarchical Model** achieved the top Macro F1 score of **0.55** on the test set.
* **Transformers vs. RNNs:** Transformer models significantly outperformed LSTMs due to superior contextualized representations.
* **Augmentation Impact:** Data augmentation improved the Macro F1 of Transformer A from 0.46 to 0.52 and significantly improved model generalization.

---

## Project 2: Large Language Models (LLMs) & Prompting
**Focus:** Comparing open-weight LLMs using Zero-Shot and Few-Shot prompting strategies.

### System Overview
* **Task:** Five-category classification: *not-sexist, threats, derogation, animosity,* and *prejudiced*.
* **Models:** `Mistral-7B-Instruct-v0.3` and `Meta-Llama-3.1-8B-Instruct`.
* **Inference:** 4-bit quantization using `bitsandbytes` to run on limited hardware (Google Colab).
* **Strategies:**
    * **Zero-Shot (ZS):** Task description and definitions only.
    * **Few-Shot (FS):** Injection of 2 demonstration examples per class (10 total).
    * **Thinking Prompts (T):** Encouraging Chain-of-Thought reasoning before the final label.

### Key Findings
* **Best Model:** **Mistral-7B (Few-Shot)** achieved the highest Macro F1 of **0.53**.
* **Prompting Variance:** Mistral showed massive improvement with few-shot examples, while Llama-3.1 experienced a performance regression.
* **Reasoning Noise:** "Thinking" prompts generally **degraded** performance, suggesting that added verbosity can introduce noise in short-text classification.
* **Safety Measures:** Llama-3.1 initially exhibited "Refuse-to-Reply" behavior due to safety filters; this was successfully bypassed using a "DAN" prompt injection to reduce the fail ratio from 2% to 0%.

---

## Authors
* Faezeh Sarlakifar
* Ludovico Gorrieri
* Alessandro Capialbi
* Giacomo Antonelli 
(University of Bologna )
