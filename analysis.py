import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- 1. LOAD DATA ---
# Make sure language.csv is in the same folder as this script
df = pd.read_csv("language.csv")

print("Step 1: Data loaded successfully.")
print(df.head())  # Optional: See first few rows


# --- 2. PREPROCESSING ---
def clean_text(text):
    text = str(text)
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'\[\]', ' ', text)
    text = text.lower()
    return text

df['cleaned_text'] = df['Text'].apply(clean_text)
print("Step 2: Text cleaning complete.")


# --- 3. ENCODING & VECTORIZATION ---
le = LabelEncoder()
y = le.fit_transform(df['language'])

cv = CountVectorizer(analyzer='char', ngram_range=(1,2))
X = cv.fit_transform(df['cleaned_text'])

print("Step 3: Vectorization complete.")


# --- 4. TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

print("Step 4: Training complete.")


# --- 5. EVALUATION ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Final Accuracy: {accuracy * 100:.2f}%")


# --- 6. PREDICTION FUNCTION ---
def predict_language(text):
    cleaned = clean_text(text)
    vectorized = cv.transform([cleaned])
    prediction = model.predict(vectorized)
    return le.inverse_transform(prediction)[0]


# --- 7. TEST ---
test_phrase = "नमस्ते"
print(f"Test Prediction for '{test_phrase}': {predict_language(test_phrase)}")