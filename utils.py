import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

def load_models():
    """Memuat model dengan penyesuaian untuk Bahasa Indonesia"""
    from config import Config
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME)
    
    # Tambahkan token khusus untuk bahasa informal
    new_tokens = ['sangat', 'banget', 'gak', 'ga', 'nggak', 'enggak', 'tdk', 'tak']
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    """Analisis sentimen dengan penanganan khusus Bahasa Indonesia"""
    # Preprocessing teks
    text = text.lower().replace('gak', 'tidak').replace('ga', 'tidak')
    
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128  # Lebih pendek untuk bahasa informal
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    
    # Aturan khusus untuk kata-kata negatif
    negative_words = ['buruk', 'jelek', 'tidak', 'jangan', 'gagal', 'kecewa']
    if any(word in text for word in negative_words) and pred.item() != 2:
        return 2, 0.9  # Paksa Negatif dengan confidence tinggi
    
    # Aturan khusus untuk kata-kata positif
    positive_words = ['bagus', 'baik', 'puas', 'senang', 'mantap', 'recommend']
    if any(word in text for word in positive_words) and pred.item() != 0:
        return 0, 0.9  # Paksa Positif dengan confidence tinggi
    
    return pred.item(), confidence.item()

def get_sentiment_label(pred, confidence):
    """Mengkonversi prediksi ke label dengan threshold yang lebih rendah"""
    from config import Config
    labels = {0: 'Positif', 1: 'Netral', 2: 'Negatif'}
    
    # Threshold yang lebih rendah untuk Bahasa Indonesia
    if confidence < 0.5:  # Diubah dari 0.7 ke 0.5
        return 'Netral', confidence
    
    return labels.get(pred, 'Netral'), confidence