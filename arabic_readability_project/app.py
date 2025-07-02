# app.py

import os
import torch
import pandas as pd
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. INITIALIZATION ---

# Initialize the Flask app
app = Flask(__name__)

# --- 2. LOAD MODELS AND DATA (This happens only ONCE when the app starts) ---

print("Loading models and data...")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths
MODEL_PATH = './model'
SIMPLIFY_DICT_PATH = 'hard_easy.xlsx'

# Load the fine-tuned BERT model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("BERT model loaded successfully.")
except Exception as e:
    print(f"Error loading BERT model: {e}")
    model = None # Set to None to handle errors gracefully

# Load the simplification dictionary
try:
    df = pd.read_excel(SIMPLIFY_DICT_PATH)
    df = df.rename(columns={df.columns[0]: "hard", df.columns[1]: "simple"})
    df = df[["hard", "simple"]].dropna()
    simplify_dict = dict(zip(df["hard"], df["simple"]))
    print("Simplification dictionary loaded successfully.")
except Exception as e:
    print(f"Error loading simplification dictionary: {e}")
    simplify_dict = {}

# Define the label maps (consistent with your training)
label_map = {'easy': 0, 'medium': 1, 'hard': 2}
inverse_label_map = {v: k for k, v in label_map.items()}

# --- 3. CORE LOGIC FUNCTION (Adapted from your notebook) ---

def analyze_and_simplify_sentence(sentence):
    if not model or not tokenizer:
        return {"error": "Model not loaded."}

    words = sentence.split()
    word_predictions = []
    modified_words = []
    sentence_contains_hard = False
    sentence_contains_medium = False

    for word in words:
        # Predict difficulty
        inputs = tokenizer(
            word, max_length=32, padding='max_length',
            truncation=True, return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class_id = torch.argmax(probs).item()
            predicted_difficulty = inverse_label_map[pred_class_id]
        
        word_predictions.append({'word': word, 'difficulty': predicted_difficulty})
        
        # Check for sentence level
        if predicted_difficulty == 'hard':
            sentence_contains_hard = True
        elif predicted_difficulty == 'medium':
            sentence_contains_medium = True
        
        # Simplify if needed
        if predicted_difficulty in ['hard', 'medium']:
            simplified_word = simplify_dict.get(word, word) # Keep original if not in dict
            modified_words.append(simplified_word)
        else:
            modified_words.append(word)

    # Determine overall sentence level based on SAMER logic
    if sentence_contains_hard:
        sentence_level = 5
    elif sentence_contains_medium:
        sentence_level = 4
    else:
        sentence_level = 3 # Or lower, based on your scale. Level 3 is "Easy"

    modified_sentence = " ".join(modified_words)
    
    return {
        "original_sentence": sentence,
        "word_predictions": word_predictions,
        "sentence_level": sentence_level,
        "modified_sentence": modified_sentence
    }

# --- 4. FLASK ROUTES ---

# Route for the main page
@app.route('/')
def home():
    # This will look for 'index.html' in the 'templates' folder
    return render_template('index.html')

# Route to handle the analysis request from the web page
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = analyze_and_simplify_sentence(text)
    return jsonify(result)

# --- 5. RUN THE APP ---

if __name__ == '__main__':
    # The app will run on http://127.0.0.1:5000
    app.run(debug=True)