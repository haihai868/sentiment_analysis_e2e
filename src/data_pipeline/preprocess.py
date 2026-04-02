import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Any
import logging
import html
# from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class TextPreprocessor:
    """Text preprocessing for Twitter data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """Clean Twitter text"""
        if not isinstance(text, str):
            return ""
        
        text = html.unescape(text)
        # Remove HTML tags
        # text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove numbers and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return nltk.word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing pipeline"""
        logger.info("Starting text preprocessing...")
        
        # Clean text
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        # Tokenize, remove stopwords, lemmatize
        df['tokens'] = df['clean_text'].apply(self.tokenize)
        df['tokens'] = df['tokens'].apply(self.remove_stopwords)
        df['tokens'] = df['tokens'].apply(self.lemmatize)
        
        # Join tokens back to text
        df['clean_text'] = df['tokens'].apply(' '.join)
        
        # Remove empty texts
        df = df[df['clean_text'].str.len() > 0]
        
        logger.info(f"Preprocessing complete. {len(df)} samples remaining")
        
        return df
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        texts = [self.clean_text(text) for text in texts]
        texts_tokens = [self.tokenize(text) for text in texts]
        texts_tokens = [self.remove_stopwords(tokens) for tokens in texts_tokens]
        texts_tokens = [self.lemmatize(tokens) for tokens in texts_tokens]
        texts = [' '.join(tokens) for tokens in texts_tokens]
        return texts
    
if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('data/raw/Twitter_Data.csv')
    
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_df(df)
    processed_df.to_csv('data/processed/processed_twitter_data.csv', index=False)