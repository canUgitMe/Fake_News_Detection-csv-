ENTIRE CODE:
(Accuracy:99%)


import pandas as pd

url_fake = "https://media.githubusercontent.com/media/canUgitMe/Fake_News_Detection-csv-/refs/heads/main/Fake.csv"
url_true = "https://media.githubusercontent.com/media/canUgitMe/Fake_News_Detection-csv-/refs/heads/main/True.csv"

true = pd.read_csv(url_true)
fake = pd.read_csv(url_fake)

print("Fake news shape:", fake.shape)
print("True news shape:", true.shape)

fake.head()

# Cell 3: combine and label
def get_text_column(df):
    for c in ['text','article','content','body','full_text']:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]

fake_text_col = get_text_column(fake)
true_text_col = get_text_column(true)

def make_content(df, text_col):
    if 'title' in df.columns:
        return df['title'].fillna('') + ' . ' + df[text_col].fillna('')
    else:
        return df[text_col].fillna('')

fake['content'] = make_content(fake, fake_text_col)
true['content'] = make_content(true, true_text_col)

fake['label'] = 1
true['label'] = 0

df = pd.concat([fake[['content','label']], true[['content','label']]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print("Combined shape:", df.shape)
df.head()





print(df['label'].value_counts())
print("\nMissing content count:", df['content'].isna().sum())

print("\nExample REAL article (label=0):")
display(df[df['label']==0].head(1))
print("\nExample FAKE article (label=1):")
display(df[df['label']==1].head(1))

df['char_len'] = df['content'].str.len()
df[['label','char_len']].groupby('label').describe().T


!pip install -q nltk
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text, remove_stopwords=True, do_lemma=True):
    text = clean_text(text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    if do_lemma:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df['clean'] = df['content'].astype(str).apply(preprocess)
df[['content','clean']].head()


from sklearn.model_selection import train_test_split

X = df['clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))  # unigrams + bigrams
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

print("TF-IDF shape (train):", Xtr.shape)

# Cell 8: train simple models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

nb = MultinomialNB()
nb.fit(Xtr, y_train)

lr = LogisticRegression(max_iter=1000, class_weight='balanced')  # class_weight helps if imbalance
lr.fit(Xtr, y_train)

# Cell 9: evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def evaluate(model, X_test_vec):
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, digits=4))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_vec)[:,1]
        try:
            print("ROC AUC:", roc_auc_score(y_test, probs))
        except:
            pass

print("=== Naive Bayes ===")
evaluate(nb, Xte)
print("\n=== Logistic Regression ===")
evaluate(lr, Xte)

import joblib
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(nb, 'nb_model.joblib')
joblib.dump(lr, 'lr_model.joblib')
print("Saved tfidf_vectorizer.joblib, nb_model.joblib, lr_model.joblib")

def predict_text(model, vectorizer, raw_text):
    processed = preprocess(raw_text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0][1]
    return {"prediction": int(pred), "probability": float(prob) if prob is not None else None}

# Example
print(predict_text(nb, tfidf, "Toxic metals found in dugongs in Tamil Nadu, study blames polluted seagrass diet"))
