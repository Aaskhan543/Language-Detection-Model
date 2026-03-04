# Language-Detection-Model
A Machine Learning and NLP project that identifies the language of a given text using natural language processing, feature engineering, and classification algorithms.
# 🌍 Language Detection Model

## 📌 Project Overview

The **Language Detection Model** is a Machine Learning and Natural Language Processing (NLP) project that automatically predicts the language of a given text input.

The model is trained using a multilingual dataset and applies text vectorization and the Naive Bayes classification algorithm to identify languages accurately.

---

## 🚀 Features

* Detects language from user input text
* Uses NLP text vectorization
* Machine Learning based classification
* Interactive user prediction system
* Implemented using Python & Scikit-learn
* Jupyter Notebook implementation included

---

## 🧠 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Natural Language Processing (NLP)
* Multinomial Naive Bayes

---

## 📂 Project Structure

```
Language-Detection-Model/
│
├── language_detection.ipynb
├── language_detection.py
├── language.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ How the Model Works

### 1️⃣ Import Libraries

The project starts by importing required libraries:

* NumPy → numerical operations
* Pandas → dataset handling
* Scikit-learn → machine learning tools

---

### 2️⃣ Load Dataset

```
data = pd.read_csv("language.csv")
```

The dataset contains:

* Text sentences
* Corresponding language labels

Total samples: **22,000 rows**

---

### 3️⃣ Text Vectorization (NLP)

Machine learning models cannot understand text directly.

```
CountVectorizer()
```

CountVectorizer converts text into numbers by counting word frequency.

Example:

```
"love data science"
"love machine learning"
```

Becomes:

```
['data','learning','love','machine','science']
```  

---

### 4️⃣ Dataset Checking

```
data.isnull().sum()
```

No missing values are found.

Each language contains equal samples, making the dataset balanced.

---

### 5️⃣ Prepare Input and Output Data

```
x = np.array(data["Text"])
y = np.array(data["language"])
```

* **x** → text data
* **y** → language labels

---

### 6️⃣ Convert Text to Numerical Features

```
cv = CountVectorizer()
x = cv.fit_transform(x)
```

All sentences are converted into a sparse numerical matrix.

---

### 7️⃣ Train-Test Split

```
train_test_split(x, y, test_size=0.33, random_state=42)
```

Dataset division:

* 67% Training Data
* 33% Testing Data

---

### 8️⃣ Model Training

```
model = MultinomialNB()
model.fit(X_train, y_train)
```

Algorithm Used:
**Multinomial Naive Bayes**

Why?

* Fast
* Efficient for text classification
* High performance on NLP tasks

---

### 9️⃣ Model Evaluation

```
model.score(X_test, y_test)
```

Model Accuracy:
**≈ 95% Accuracy**

---

### 🔟 Language Prediction

```
user = input("Enter a text")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
```

Steps:

1. User enters text
2. Text converts into vector
3. Model predicts language
4. Predicted language is displayed

Example:

```
Input: Hallo wereld
Output: Dutch
```

---

## ▶️ Installation & Setup

Clone repository:

```
git clone https://github.com/YOUR_USERNAME/Language-Detection-Model.git
cd Language-Detection-Model
```

Install dependencies:

```
pip install -r requirements.txt
```

Run project:

```
python language_detection.py
```

---

## 📊 Model Performance

* Algorithm: Multinomial Naive Bayes
* NLP Method: CountVectorizer
* Accuracy: ~95%

---

## 📈 Future Improvements

* Deep Learning models (LSTM / Transformers)
* Web application deployment
* REST API integration
* Real-time language detection system

---

## 👨‍💻 Author

**Mohd Aas Khan**
Machine Learning & Ai Enthusiast

---

⭐ If you like this project, give it a star on GitHub!
