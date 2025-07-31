import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import random
import gc

#  NLTK SETUP 

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

#  DATA LOADING 
csv_path = ""  #update for yourself 
df = pd.read_csv(csv_path)

#  DATA CLEANING 
def temizle(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[0-9]+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, language='turkish')
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

df["clean_comment"] = df["comment"].apply(temizle)

#  DATA REDUCTION ( I did not have sufficient resources to train the entire dataset.) 
df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)  # %5 veriyle daha hızlı eğitim
print(f"Kullanılan veri seti boyutu: {len(df)}")

#  LABEL CREATION 
df['point'] = df['point'].astype(str).str.replace(',', '.').astype(float)
df['label'] = df['point'].apply(lambda x: 1 if x >= 3.0 else 0)

#  DATA AUGMENTATION 
def augment_text(text):
    words = text.split()
    if len(words) < 2:
        return text
    i = random.randint(0, len(words)-1)
    j = random.randint(0, len(words)-1)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)

minority_df = df[df['label'] == 0]
augmented = minority_df.sample(frac=1.0, random_state=42).copy()
augmented['clean_comment'] = augmented['clean_comment'].apply(augment_text)
df = pd.concat([df, augmented]).reset_index(drop=True)
del minority_df, augmented
gc.collect()

#  TOKENIZATION 
df = df[["clean_comment", "label"]].dropna().reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
dataset = Dataset.from_pandas(df)
del df
gc.collect()

def tokenize(example):
    return tokenizer(example["clean_comment"], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
del dataset
gc.collect()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
test_dataset = train_test["test"]
del tokenized_dataset, train_test
gc.collect()

#  MODEL & TRAINING (Since I did not have sufficient resources, I had to make some reductions here too.) 
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,       
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

#  EVALUATION 
preds = trainer.predict(test_dataset)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print("\n REPORT:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"])
plt.xlabel("Prediction")
plt.ylabel("Reality")
plt.title("Confusion Matrix")
plt.show()

#  LOSS & ACCURACY PLOTS 
logs = trainer.state.log_history
train_loss = [log["loss"] for log in logs if "loss" in log]
eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
eval_acc = [log["accuracy"] for log in logs if "accuracy" in log]

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss, label="Training Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.xlabel("Adım")
plt.ylabel("Loss")
plt.title("Loss Graph")
plt.legend()

plt.subplot(1,2,2)
plt.plot(eval_acc, label="Validation Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Graph")
plt.legend()

plt.tight_layout()
plt.show()

#  SAVE MODEL 
model_path = "./MovieAnalysis"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("Model ve tokenizer başarıyla 'MovieAnalysis' klasörüne kaydedildi.")

#  RAM CLEANİNG 
del train_dataset, test_dataset, trainer, preds, y_true, y_pred, cm, logs, train_loss, eval_loss, eval_acc
gc.collect()
