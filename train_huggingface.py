import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score
import numpy as np

# Vérifier le contenu brut du fichier intents.json
try:
    with open('data/intents.json', 'r', encoding='utf-8-sig') as f:
        raw_data = json.load(f)
    print('Contenu brut de intents.json :', raw_data)
except Exception as e:
    print(f'Erreur lors de la lecture de intents.json : {e}')
    exit(1)

# Charger les données
try:
    dataset = Dataset.from_pandas(pd.DataFrame(raw_data))
    print('Structure des données :', dataset)
    print('Exemple de données :', dataset[0])
except Exception as e:
    print(f'Erreur lors du chargement du dataset : {e}')
    exit(1)

# Créer un mappage des étiquettes
labels = list(set(dataset['intent']))
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
print('Labels :', labels)

# Initialiser le tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

# Préparer les données
def preprocess_function(examples):
    result = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    result['labels'] = [label2id[example] for example in examples['intent']]
    return result

dataset = dataset.map(preprocess_function, batched=True)

# Initialiser le modèle
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-multilingual-uncased',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Paramètres d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Calculer les métriques
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# Initialiser le trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics
)

# Entraîner le modèle
trainer.train()

# Évaluer
eval_results = trainer.evaluate()
print('Résultats de l\'évaluation :', eval_results)
accuracy = eval_results.get('eval_accuracy', 0.0)
print(f'Précision : {accuracy:.2f}')

# Sauvegarder le modèle
model.save_pretrained('models/intent_classifier')
tokenizer.save_pretrained('models/intent_classifier')

# Tester quelques exemples
test_texts = ['What is the Power BI course about?', 'Merci beaucoup!', 'Contact details for BI-GEEK?']
for text in test_texts:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
    predicted_intent = id2label[predicted_id]
    print(f'Texte : {text}, Prédiction : {predicted_intent}')