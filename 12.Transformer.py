from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Muat model dan tokenizer BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 kelas: negatif, netral, positif

def predict_sentiment(text):
    """Fungsi untuk prediksi sentimen."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        if not text.strip():
            return render_template('trasformer.html', error="Masukkan teks terlebih dahulu!")
        
        try:
            prediction = predict_sentiment(text)
            labels = {0: '😠 Negatif', 1: '😐 Netral', 2: '😊 Positif'}
            result = labels.get(prediction, 'Tidak diketahui')
            return render_template('trasformer.html', text=text, result=result)
        except Exception as e:
            return render_template('trasformer.html', error=f"Error: {str(e)}")
    
    return render_template('trasformer.html')

if __name__ == '__main__':
    app.run(debug=True)