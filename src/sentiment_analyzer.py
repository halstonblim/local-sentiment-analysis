import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
import pandas as pd

class SentimentAnalyzer:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model_name = model_name
        logging.info(f"Initialized sentiment analyzer with model: {model_name}")

    def analyze_texts(self, texts, batch_size=32):
        """Analyze sentiment for a list of texts."""
        logits_neg, logits_pos, preds = [], [], []
        skipped = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i + batch_size]
            batch_texts_cleaned = [text if isinstance(text, str) and text.strip() else "[EMPTY]" for text in batch_texts]

            inputs = self.tokenizer(
                batch_texts_cleaned, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_logits = outputs.logits
                batch_probs = softmax(batch_logits, dim=1)
                batch_preds = torch.argmax(batch_probs, dim=1)

                logits_neg.extend(batch_logits[:, 0].cpu().tolist())
                logits_pos.extend(batch_logits[:, 1].cpu().tolist())
                preds.extend(batch_preds.cpu().tolist())

        return {
            "logit_negative": logits_neg,
            "logit_positive": logits_pos,
            "predicted_label": ["negative" if p == 0 else "positive" for p in preds],
            "model_version": self.model_name
        }

    def analyze_dataframe(self, df):
        """Analyze sentiment for all texts in a DataFrame."""
        texts = df["text"].astype(str).tolist()
        results = self.analyze_texts(texts)
        
        for key, value in results.items():
            df[key] = value
            
        return df 