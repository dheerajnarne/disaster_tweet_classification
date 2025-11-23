#  Disaster Tweet Classification: From TF-IDF to Fine-Tuned BERT

## 📌 Project Overview
This project focuses on **Natural Language Processing (NLP)** to classify tweets into two categories: **Real Disasters (1)** and **Not Real Disasters (0)**. 

Twitter is a vital source of real-time communication during emergencies. However, the use of metaphorical language (e.g., "This movie bombed," "My heart is on fire") makes it difficult for automated systems to distinguish between actual emergencies and casual conversation. This project explores multiple techniques ranging from traditional statistical methods to state-of-the-art Deep Learning models to solve this ambiguity.

---

## 📂 Dataset Description
The model was trained on a combined dataset sourced from Kaggle competitions and social media repositories.

* **Sources:**
    1.  Kaggle "NLP Getting Started" `train.csv`.
    2.  [cite_start]`socialmedia-disaster-tweets-DFE.csv`[cite: 2].
* [cite_start]**Total Data Points:** The datasets were concatenated to form a robust corpus of **18,489 rows**[cite: 4].
* **Class Balance:**
    * **Not Disaster (0):** ~57%
    * [cite_start]**Disaster (1):** ~43%[cite: 7].
* [cite_start]**Features:** The primary input is the `text` column, and the target variable is `target`[cite: 4].

---

## ⚙️ Preprocessing Pipeline
[cite_start]Before feeding data into the models, the text underwent standard NLP preprocessing steps[cite: 6]:

1.  **Lowercasing:** Converting all text to lowercase to ensure uniformity.
2.  **URL Removal:** Removing hyperlinks (`http`, `www`) using regex.
3.  **Punctuation Removal:** Stripping special characters.
4.  **Tokenization:** Splitting text into individual words using NLTK's `word_tokenize`.

---

## 🚀 Methodologies Implemented

This project implemented and compared four distinct approaches:

### 1. TF-IDF + Logistic Regression (Baseline)
* [cite_start]**Feature Extraction:** Used `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) with a maximum of 5,000 features and standard English stop words removed[cite: 10].
* **Model:** Logistic Regression.
* **Logic:** Relies on the frequency of specific keywords (e.g., "fire", "kill", "bomb") to determine the class.

### 2. Word2Vec Embeddings + Logistic Regression
* **Feature Extraction:** Utilized pre-trained **Google News Vectors (Negative 300)**.
* [cite_start]**Method:** Each sentence was represented by averaging the 300-dimensional word vectors of its constituent tokens[cite: 15].
* **Model:** Logistic Regression.
* **Logic:** Captures semantic meaning and relationships between words, rather than just keyword matching.

### 3. BERT Feature Extraction + Logistic Regression
* [cite_start]**Feature Extraction:** Used the `bert-base-uncased` model to extract embeddings. specifically the `last_hidden_state` of the `[CLS]` token[cite: 26].
* **Model:** Logistic Regression (max iter 500).
* **Logic:** Uses the contextual understanding of a Transformer model but keeps the model weights frozen, using it only as a static feature generator.

### 4. Fine-Tuned BERT (State-of-the-Art)
* [cite_start]**Model:** `BertForSequenceClassification` loaded with `bert-base-uncased`[cite: 39].
* **Training:** Fine-tuned the entire model (all layers) using the Hugging Face `Trainer` API.
* **Hyperparameters:**
    * Epochs: 10
    * Batch Size: 16 (Train) / 64 (Eval)
    * Warmup Steps: 500
    * [cite_start]Weight Decay: 0.01[cite: 39].
* [cite_start]**Tracking:** Experiments were tracked using **Weights & Biases (WandB)**[cite: 24].

---

## 📊 Results & Comparison

The Fine-Tuned BERT model significantly outperformed all other approaches.

| Model Approach | Accuracy | Precision (Disaster) | Recall (Disaster) | F1-Score (Disaster) |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + LogReg** | [cite_start]**83.86%** [cite: 11] | 0.87 | 0.73 | 0.79 |
| **Word2Vec + LogReg** | [cite_start]**81.10%** [cite: 16] | 0.82 | 0.71 | 0.76 |
| **BERT (Static) + LogReg** | [cite_start]**80.67%** [cite: 26] | N/A | N/A | N/A |
| **Fine-Tuned BERT** | [cite_start]**92.75%** [cite: 39] | **High** | **High** | **High** |

### Key Observations:
* **TF-IDF Strength:** Surprisingly, the simple TF-IDF model outperformed the Word2Vec and Static BERT approaches. [cite_start]This suggests that "disaster" tweets are highly keyword-dependent (words like "earthquake" or "flood" are strong standalone indicators)[cite: 13].
* [cite_start]**Static BERT Limitations:** Using BERT embeddings with a simple Logistic Regression failed to converge effectively, resulting in the lowest accuracy (80.67%)[cite: 26].
* [cite_start]**The Power of Fine-Tuning:** By updating the weights of the BERT model during training, accuracy jumped to **92.75%** by Epoch 8[cite: 39]. This demonstrates the necessity of contextual learning for this specific task.

---

## 🔍 Error Analysis: Why do models fail?

[cite_start]An in-depth look at misclassified samples reveals several recurring themes[cite: 40]:

### 1. Metaphorical Language
The model struggles when disaster words are used in non-literal contexts.
* *Text:* "katramsland yes im a bleeding heart liberal"
* *True Label:* Not Disaster (0) | [cite_start]*Predicted:* Disaster (1) (due to "bleeding")[cite: 40].
* *Text:* "the dress memes have officially exploded on the internet"
* *True Label:* Not Disaster (0) | [cite_start]*Predicted:* Disaster (1) (due to "exploded")[cite: 40].

### 2. Contextual Ambiguity
Some tweets are vague without external context.
* *Text:* "looks like a war zone outside whats going on"
* *True Label:* Not Disaster (0) | [cite_start]*Predicted:* Disaster (1)[cite: 40].

### 3. Sarcasm and Slang
* *Text:* "burning bridges is my forte"
* *True Label:* Not Disaster (0) | [cite_start]*Predicted:* Disaster (1)[cite: 40].

### 4. Keyword Interference
Top words indicating positive (disaster) classes were strong triggers like "derailment," "suicide," and "bombing." [cite_start]Conversely, words like "love," "new," and "panic" (surprisingly) leaned towards negative classes[cite: 13].

---

## 🛠️ Tech Stack & Dependencies
* **Python 3.10**
* [cite_start]**Libraries:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `gensim`, `transformers`, `torch`, `wandb`[cite: 1, 23, 25].
* [cite_start]**Hardware:** Trained on GPU (CUDA) enabled environment[cite: 26].

## 🔮 Conclusion
While keyword-based models (TF-IDF) provide a strong baseline for disaster classification due to the specificity of danger-related vocabulary, they fail to grasp sarcasm and metaphors. **Fine-tuning Large Language Models (BERT)** bridges this gap, providing a ~9% improvement in accuracy over the baseline by understanding the semantic context of the tweet.
