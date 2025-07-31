# 🎬 MovieAnalysis_NLP

This is a simple Natural Language Processing (NLP) project that analyzes **Turkish movie reviews** and predicts whether they are **positive or negative** using a **BERT-based model**.

---

## 🧩 What This Project Does

- Loads and cleans Turkish movie reviews (removes stopwords, punctuation, etc.)
- Tokenizes text using Hugging Face's pre-trained Turkish BERT model
- Trains a binary classification model (positive/negative)
- Visualizes model performance (accuracy, loss, confusion matrix)
- Saves the trained model for future use

---

## 📌 Why I Made This

I'm currently learning NLP and deep learning. Since my computer has limited resources, I worked with a **smaller dataset** and still achieved decent results by using **transfer learning**.

This project helped me understand:
- How to preprocess and clean text data
- How to use pre-trained Transformer models (BERT)
- How to evaluate NLP classification models

---

## 🔍 Tools and Libraries Used

| Library | Purpose |
|--------|---------|
| `pandas`, `numpy` | Data handling and manipulation |
| `re`, `string` | Text cleaning |
| `nltk` | Stopword removal and tokenization |
| `scikit-learn` | Evaluation metrics like confusion matrix, classification report |
| `matplotlib`, `seaborn` | Visualization of model training and results |
| `transformers`, `datasets` | Pre-trained BERT model, tokenization, training |
| `gc`, `random` | Resource management and reproducibility |

Install required libraries:
---
📊 Results
Although I trained the model on a limited dataset, the performance was promising:
Accuracy: about 0.84%
F1-score: class 0: 0.83%, class 1: %0.85
Loss & Accuracy Graphs: See /results/Loss_Accuracy_Graph.png
Confusion Matrix: See /results/Confusion_Matrix.png

Note: Due to hardware limitations, I didn't train on the full dataset or apply heavy hyperparameter tuning. With better resources, performance can be further improved.
---
📁 Dataset
The dataset I used is a Turkish movie review dataset available on Kaggle. Due to license concerns, I haven’t uploaded it to this repository.

https://www.kaggle.com/datasets/mustfkeskin/turkish-movie-sentiment-analysis-dataset 

You can replace this link with the specific dataset link you used once you upload the project.
---
🚀 How to Improve
Use a larger and more diverse dataset
Apply better hyperparameter tuning
Try more advanced models like RoBERTa or LLaMA variants (if GPU is available)
Include sentiment intensity (not just binary)
Add experiment tracking (like WandB)
---
```bash
pip install pandas numpy scikit-learn matplotlib transformers datasets nltk
