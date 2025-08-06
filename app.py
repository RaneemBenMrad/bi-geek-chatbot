from flask import Flask, request, jsonify, render_template, send_from_directory
import psycopg2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
import os
from dotenv import load_dotenv
from groq import Groq
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import CORS

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), 'chatbot.log')
handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console_handler)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
CORS(app)

# Initialize Groq client
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    timeout=120.0,
    max_retries=3
)

# Load Hugging Face model and tokenizer
try:
    model_path = 'models/intent_classifier'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    logging.info("Hugging Face model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Hugging Face model: {str(e)}")
    model = None
    tokenizer = None

# SYSTEM PROMPT for Groq
SYSTEM_PROMPT = """
You are a helpful assistant for BI-GEEK, a leading academy in Tunisia specializing in Business Intelligence (BI), Data Science, and Big Data training. ONLY respond to questions about BI-GEEK’s courses, enrollment, instructors, or related services. If the query is unrelated to BI-GEEK, respond with: "Sorry, I can only assist with BI-GEEK training programs. Please ask about our courses or contact contact@bi-geek.net."
Available courses:
- Power BI (8 weeks, 700,000 DT, online available)
- Data Engineering Bootcamp (20 weeks, 2,500,000 DT)
- Microsoft Certifications (PL-300, DP-900, DP-203, 8 weeks, 700,000 DT each, online available)
- Python for Data Science (8 weeks, 700,000 DT, online available)
- Big Data (20 weeks, 3,000,000 DT)
- 3-day workshops (700,000 DT)
Courses start quarterly (January, April, July, October). For enrollment, direct users to www.bi-geek.net ('Inscription' section), contact@bi-geek.net, or +216 58 611 283. Instructors are certified experts. If unsure, suggest contacting BI-GEEK.
"""

# -------------------- PostgreSQL Connection --------------------
def get_pg_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "bigeek"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        logging.info(f"Connected to PostgreSQL - Host: {os.getenv('POSTGRES_HOST', 'localhost')}, Database: {os.getenv('POSTGRES_DB', 'bigeek')}, Schema: public")
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        logging.info(f"Tables in public schema: {[table[0] for table in tables]}")
        cur.close()
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to PostgreSQL: {e}")
        raise

# -------------------- SQLite for Chat History --------------------
def init_db():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY, user_message TEXT, bot_response TEXT)''')
    conn.commit()
    conn.close()
    logging.info("SQLite database initialized")

def insert_chat_history(user_message, bot_response):
    try:
        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (user_message, bot_response) VALUES (?, ?)", (user_message, bot_response))
        conn.commit()
        conn.close()
        logging.info(f"Inserted chat history: {user_message[:50]}...")
    except Exception as e:
        logging.error(f"Failed to insert chat history: {e}")

def get_recent_chat_history(limit=5):
    try:
        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()
        c.execute("SELECT user_message, bot_response FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
        history = c.fetchall()
        conn.close()
        bi_geek_terms = [
            "course", "training", "power bi", "data science", "big data", "certification", "python",
            "microsoft", "bi-geek", "courses", "formations", "subscribe", "enroll", "contact", "how",
            "join", "inscription", "phone", "email", "support", "joindre", "téléphone", "numéro",
            "adresse", "help", "informations", "infos", "contacter", "coordonnee", "connect", "reach"
        ]
        filtered = [
            msg for user, bot in reversed(history)
            if any(term in user.lower() or term in bot.lower() for term in bi_geek_terms)
            for msg in [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]
        ]
        logging.info(f"Retrieved {len(filtered)} relevant chat history entries")
        return filtered[:limit * 2]
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        return []

# -------------------- Formation Lookup in PostgreSQL --------------------
def get_course_by_keyword(user_input):
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        query = """
            SELECT DISTINCT f.title, f.duration_weeks, f.price_dt, f.description, f.modality
            FROM public.formations f
            LEFT JOIN public.keywords k ON f.id = k.formation_id
            WHERE LOWER(k.keyword) LIKE %s
               OR LOWER(f.title) LIKE %s
               OR LOWER(f.description) LIKE %s
            ORDER BY f.title
            LIMIT 3;
        """
        words = user_input.strip().lower().split()
        results = []
        for word in words:
            like_input = f"%{word}%"
            cur.execute(query, (like_input, like_input, like_input))
            results.extend(cur.fetchall())
        results = list(set(results))
        logging.info(f"Query executed for '{user_input}', results: {results}")
        cur.close()
        conn.close()
        if results:
            response = "\n".join(
                [f"{title} is a {duration}-week course ({modality}) for {price} DT. Description: {desc}" for
                 title, duration, price, desc, modality in results])
            logging.info(f"DB match for '{user_input}': {response[:100]}...")
            return response
        logging.info(f"No DB match for '{user_input}'")
        return None
    except Exception as e:
        logging.error(f"DB error for '{user_input}': {e}")
        return None

# -------------------- Contact Lookup in PostgreSQL --------------------
def get_contact_info():
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        logging.info("Executing contact query on public.contacts")
        query = "SELECT label, value FROM public.contacts;"
        cur.execute(query)
        results = cur.fetchall()
        logging.info(f"Contact query results: {results}")
        cur.close()
        conn.close()
        email = phone = website = None
        for label, value in results:
            if label.lower() == "email":
                email = value
            elif label.lower() == "phone":
                phone = value
            elif label.lower() == "website":
                website = value
        contact_text = []
        if email:
            contact_text.append(f"📧 Email: {email}")
        if phone:
            contact_text.append(f"📞 Phone: {phone}")
        if website:
            contact_text.append(f"🌐 Website: {website}")
        if contact_text:
            response = "Here is the contact information for BI-GEEK:\n" + "\n".join(contact_text)
        else:
            response = "Contact information: Email: contact@bi-geek.net, Phone: +216 58 611 283, Website: www.bi-geek.net"
        logging.info(f"Contact response: {response[:100]}...")
        return response
    except Exception as e:
        logging.error(f"DB error in get_contact_info: {e}")
        try:
            conn = get_pg_connection()
            cur = conn.cursor()
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'contacts';")
            table_exists = cur.fetchone()
            logging.info(f"Table 'contacts' exists in public schema: {bool(table_exists)}")
            cur.close()
            conn.close()
        except Exception as e2:
            logging.error(f"Error checking table existence: {e2}")
        return "Contact information: Email: contact@bi-geek.net, Phone: +216 58 611 283, Website: www.bi-geek.net"

# -------------------- Intent Classification with Hugging Face --------------------
def classify_intent(text):
    try:
        if not model or not tokenizer:
            logging.error("Hugging Face model or tokenizer not loaded")
            return "other"
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=-1).item()
        intent = model.config.id2label[predicted_id]
        logging.info(f"Hugging Face intent for '{text}': {intent}")
        return intent
    except Exception as e:
        logging.error(f"Error in Hugging Face intent classification: {e}")
        return "other"

# -------------------- Main Chat Response Logic --------------------
def get_response(user_input):
    try:
        logging.info(f"Processing user input: '{user_input}'")
        clean = user_input.lower().strip()
        logging.info(f"Normalized input: '{clean}'")

        intent = classify_intent(clean)

        if intent == "contact":
            response = get_contact_info()
            insert_chat_history(user_input, response)
            logging.info(f"Contact response for '{user_input}': {response[:100]}...")
            return response

        if intent == "certifications":
            response = (
                "BI-GEEK offers certified courses including:\n"
                "- PL-300 (Power BI Data Analyst)\n"
                "- DP-900 (Azure Data Fundamentals)\n"
                "- DP-203 (Azure Data Engineering)\n"
                "Each certification runs for 8 weeks and costs 700,000 DT.\n"
                "Courses are online and prepare for official Microsoft exams."
            )
            insert_chat_history(user_input, response)
            logging.info(f"Certification response for '{user_input}'")
            return response

        if intent == "enroll":
            response = "To subscribe, visit www.bi-geek.net (Inscription section), email contact@bi-geek.net, or call +216 58 611 283. Courses start quarterly (January, April, July, October)."
            insert_chat_history(user_input, response)
            logging.info(f"Subscription response for '{user_input}': {response[:100]}...")
            return response

        if intent == "thanks":
            response = "You're welcome! How else can I assist you with BI-GEEK courses?"
            insert_chat_history(user_input, response)
            logging.info(f"Thank you response for '{user_input}'")
            return response

        if intent == "course_info":
            course_info = get_course_by_keyword(clean)
            if course_info:
                insert_chat_history(user_input, course_info)
                logging.info(f"DB match for '{user_input}': {course_info[:100]}...")
                return course_info

        chat_history = get_recent_chat_history()
        course_context = get_course_by_keyword(clean) or "No course data found in the database."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\nRelevant data:\n" + course_context},
            *chat_history,
            {"role": "user", "content": user_input}
        ]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=500
        )
        response = chat_completion.choices[0].message.content
        insert_chat_history(user_input, response)
        logging.info(f"Groq response for '{user_input}': {response[:100]}...")
        return response

    except Exception as e:
        logging.error(f"Error processing '{user_input}': {e}")
        response = "Sorry, I couldn't process that request. Please try again or contact contact@bi-geek.net."
        insert_chat_history(user_input, response)
        return response

# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)
    return jsonify({'response': response})

@app.route('/chatbot/history', methods=['GET'])
def chat_history():
    history = get_recent_chat_history(limit=10)
    return jsonify({'history': history})

@app.route('/test-course/<mot>', methods=['GET'])
def test_course(mot):
    info = get_course_by_keyword(mot)
    return jsonify({'result': info})

@app.route('/test-db')
def test_db():
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM public.formations;")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        logging.info(f"Test DB: {count} formations found")
        return f"✅ Connected to PostgreSQL! {count} formations found."
    except Exception as e:
        logging.error(f"Test DB failed: {e}")
        return f"❌ Failed to connect to PostgreSQL: {e}"

@app.route('/test-queries', methods=['GET'])
def test_queries():
    test_cases = [
        "Power BI course",
        "price of data science",
        "next course start",
        "what is python",
        "unrelated topic"
    ]
    results = []
    for query in test_cases:
        response = get_response(query)
        results.append({"query": query, "response": response})
    logging.info(f"Test queries executed: {len(results)} results")
    return jsonify({"test_results": results})

@app.route('/test-contact', methods=['GET'])
def test_contact():
    response = get_contact_info()
    return jsonify({"response": response})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# -------------------- Run --------------------
if __name__ == '__main__':
    try:
        conn = get_pg_connection()
        print("Database connection successful!")
        conn.close()
    except Exception as e:
        print(f"Database connection failed: {e}")
    init_db()
    print("Starting Flask server on http://127.0.0.1:5001...")
    app.run(host='127.0.0.1', port=5001, debug=True)