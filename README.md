# 🎬 MovieAnalysis_NLP

This is a simple Natural Language Processing (NLP) project that analyzes **Turkish movie reviews** and predicts whether they are **positive or negative**. I used a **BERT-based model** and trained it on a small dataset.

---

## 🧩 What this project does

- Loads and cleans movie reviews
- Uses a pre-trained Turkish BERT model
- Trains the model on labeled reviews (positive / negative)
- Evaluates with accuracy, F1-score, and confusion matrix
- Saves the trained model for later use

---

## 🔍 Tools and Libraries

- Python
- PyTorch (via Hugging Face Transformers)
- scikit-learn
- matplotlib
- pandas
- numpy

Install required libraries with:


📊 Results
Even though the training was performed on a limited dataset due to hardware limitations, the model achieved promising results:
📈 Accuracy: 84%
🎯 F1-score (Negative): 0.83
🎯 F1-score (Positive): 0.85
⚖️ Macro F1-score: 0.84
📌 Precision (Negative): 0.85
📌 Recall (Positive): 0.87
🧪 Samples Evaluated: 1089


📦 Dataset
The dataset was downloaded from Kaggle, but due to licensing concerns, it is not included in this repository.
https://www.kaggle.com/datasets/mustfkeskin/turkish-movie-sentiment-analysis-dataset


🧠 Future Improvements
Use a larger and more diverse dataset for better generalization
Deploy the model as a web API or streamlit app
Try different models such as DistilBERT or ELECTRA
Perform hyperparameter tuning
Add more explainability (e.g., word importance, attention weights)

👩‍🎓 About the Author
This project was created by a student interested in NLP and machine learning. The goal was to explore sentiment analysis in Turkish language using BERT models and improve deep learning skills.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main script
python main.py

