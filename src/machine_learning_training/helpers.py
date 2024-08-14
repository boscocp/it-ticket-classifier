from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import PorterStemmer

nltk.data.path.append("/tmp")
nltk.download("stopwords", quiet=True, download_dir="/tmp")
nltk.download("punkt", download_dir="/tmp")
stopwords = nltk.corpus.stopwords.words("english")


def preprocess_text(text: str) -> str:
    stemmer = PorterStemmer()
    text = str(text).lower()
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stopwords]
    text = " ".join(words)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text


class TFIDFExtrator(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, max_features: int = None) -> None:
        if max_features is None:
            max_features = 128
        self.column = column
        self.tfidf = TfidfVectorizer(max_features=max_features)

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        return self.tfidf.transform(X)
