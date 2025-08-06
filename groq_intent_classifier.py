import json
import os
from groq import Groq
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

# Charger la clé API depuis .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY non trouvée dans .env")

# Initialiser le client Groq
client = Groq(api_key=GROQ_API_KEY)


# Charger les données depuis intents.json
def load_intents(file_path="data/intents.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Fonction pour classer une intention avec Groq
def classify_intent(text):
    prompt = f"""
    Classifiez l'intention du texte suivant en une des catégories suivantes : 
    'course_info', 'certifications', 'contact', 'enroll', 'thanks', ou 'other'.
    Texte : "{text}"
    Répondez uniquement avec le nom de l'intention (par exemple, 'course_info').
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Vous êtes un classificateur d'intentions pour un chatbot."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


# Tester et évaluer la précision
def evaluate_intents():
    intents_data = load_intents()
    y_true = [item["intent"] for item in intents_data]
    y_pred = [classify_intent(item["text"]) for item in intents_data]

    # Calculer la précision
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Précision : {accuracy:.2f}")

    # Afficher les prédictions
    for item, pred in zip(intents_data, y_pred):
        print(f"Texte : {item['text']}, Intention réelle : {item['intent']}, Prédiction : {pred}")


if __name__ == "__main__":
    try:
        evaluate_intents()
    except Exception as e:
        print(f"Erreur : {e}")