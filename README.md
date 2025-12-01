# English-to-Punjabi Neural Machine Translator (V2)

A character-level Transformer model built from scratch in PyTorch to translate English to Punjabi. This project implements the architecture proposed in the paper *"Attention Is All You Need"* (Vaswani et al.), adapted to handle the specific linguistic challenges of the Punjabi language.

![licensed-image](https://github.com/user-attachments/assets/6f1bb856-d1c9-4e92-bfe2-79554ccd2d0a)


## 📜 Project Evolution: The Story

This repository hosts **Version 2** of the translator. To understand why this version exists, we must look at where we started.

### Version 0: "The Bureaucrat" (Previous Iteration)

  * **The Setup:** A 3-layer Transformer trained exclusively on the **Samanantar** dataset (Government reports).
  * **The Result:** The model had excellent spelling but a severe personality bias. It "thought" like a government officer.
      * *Input:* "I eat an apple."
      * *Translation:* "I receive/obtain this." (Because government data talks about receiving grants, not eating fruit).
  * **The Verdict:** Intelligent, but too formal and sheltered.

### Version 2: "The Hybrid" (Current Code)

  * **The Fix:** We built a **Data Mixer** pipeline. We combine **Opus-100** (Casual/Conversational data) with **Samanantar** (Formal data) to balance the model's vocabulary.
  * **The Upgrade:**
      * **Deeper Network:** Increased from 3 layers to **6 Layers**.
      * **Better Training:** Added Learning Rate Schedulers (`ReduceLROnPlateau`) and Checkpoint Resuming.
      * **Result:** A more robust model capable of handling sentence structures better, moving away from purely bureaucratic translations.

-----

## 🏗️ Technical Architecture

This is not a fine-tuned BERT/GPT model; it is a **Transformer trained from scratch**.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **d\_model** | 512 | Size of the vector representing each character. |
| **Heads** | 8 | Number of parallel attention heads. |
| **Layers** | 6 | Stacked Encoder and Decoder layers (increased from 3). |
| **Max Sequence** | 200 | Maximum length of a sentence. |
| **Tokenizer** | Character-Level | Custom JSON vocabularies mapping chars to integers. |

### Key Components

#### 1\. The Data Pipeline (The "Mixer")

Unlike standard loaders, this custom `DataLoader` performs a specific mix to cure dataset bias:

1.  **Loads Casual Data:** Fetches \~100k sentences from `Helsinki-NLP/opus-100`.
2.  **Fills Gaps:** Fills the remaining capacity with formal data from `ai4bharat/samanantar`.
3.  **Shuffles:** Ensures the model doesn't learn "Casual" first and "Formal" last.

#### 2\. Masking Strategy

The Transformer relies on masks to handle variable sentence lengths and prevent "cheating" during training.

  * **Padding Mask:** Ignores the `<PAD>` tokens so they don't influence attention scores.
  * **Look-Ahead Mask:** A triangular matrix used in the Decoder. It ensures that when predicting character \#4, the model can only see characters \#1, \#2, and \#3.

-----

## 🚀 Setup and Usage

### Prerequisites

  * Python 3.8+
  * PyTorch (CUDA recommended)
  * HuggingFace `datasets` library

<!-- end list -->

```bash
pip install torch datasets numpy
```

### File Structure

Ensure your directory looks like this:

```
/project-root
│
├── Transformer-trainer-translator.ipynb          # The main V2 training script (provided in this repo)
├── my-transformer-scripts/
|          ├──  Transformer.py   # Contains the Transformer class, Encoder, Decoder blocks
|          |──  english_vocab.json
|          |──  punjabi_vocab.json
├── Translator_model_weights/
│   ├── translator_model.pth
│   └── translator_engtopuj_epoch_2.pth
└── README.md
```

### Training

To start training the model (or resume from a checkpoint):

```python
# In your Python environment or Notebook
Transformer-trainer-translator.ipynb run every block
```

*Note: The script is configured to look for a checkpoint (`translator_engtopuj_epoch_X.pth`). If found, it automatically resumes training. If not, it starts from scratch.*

### Inference (Translation)

To use the model to translate text:

```python
# Load your model and dataset first, then:
sentence = "How are you?"
prediction = translate(sentence, model, train_dataset)
print(f"English: {sentence}")
print(f"Punjabi: {prediction}")
```

-----

## 📊 Performance & Observations

**Training Metrics (Epoch 10/30):**

  * **Loss:** \~1.26
  * **Convergence:** Steady decrease in loss, indicating the model is learning character relationships and grammar rules.

**Qualitative Results (V2):**
The V2 model shows significant improvement in grammar structure (Subject-Object-Verb order) compared to V0.

| English Input | V2 Prediction (Punjabi) | Analysis |
| :--- | :--- | :--- |
| *He is playing* | ਉਹ ਚਲ ਰਿਹਾ ਹੈ | **Good.** Correctly identifies subject and action. |
| *My name is Rahul* | ਮੇਰੇ ਨਾਂ ਹਾਈ ਰਾਹੁਲ | **Passable.** Transliterates "Rahul" correctly. |
| *Good morning* | ਗੂ ਦੇ ਮਾਪਿਆਂ ਦੀ ਜਾਂਚ ਕਰੋ | **Hallucination.** (Still struggles with idioms). |

*Current Limitation:* While the architecture is strong, the model requires more epochs (30+) to fully converge and reduce hallucinations on idiomatic phrases like "Good Morning".

-----

## 🔮 Future Improvements

1.  **Word-Level Tokenization:** Switching from Character-level to Sub-word (BPE) tokenization would significantly improve semantic understanding.
2.  **Beam Search:** Implementing Beam Search decoding instead of Greedy decoding (argmax) for smoother sentence generation.
3.  **More Data:** Increasing the casual dataset size to further dilute the formal bias.

-----

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. Specifically looking for contributions in **Beam Search implementation** or **BPE Tokenizer integration**.

-----

**Author:** Naman
**License:** MIT
