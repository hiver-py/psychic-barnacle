import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')

def remove_html_tags(text:str) -> str:
    """Function to clean HTML tags."""
    return re.sub(r'<.*?>', '', text)


def remove_special_characters(text:str) -> str:
    """Function to remove special characters and punctuation."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def to_lowercase(text:str) -> str:
    """Function to lowercase all text."""
    return text.lower()


def remove_stopwords(text:str) -> str:
    """Function to remove stop words."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return ' '.join(word for word in words if word not in stop_words)

# Function to clean the dataset
def clean_text(text:str) -> str:
    """Cleans the text from html tags and special characters, converts to lower case and returns it."""
    text = remove_html_tags(text)
    text = to_lowercase(text) 
    text = remove_special_characters(text)
    # Leaving stopwords in for now.
    # print('Remove stopwords')
    # text = remove_stopwords(text)  # Remove stop words
    return text
