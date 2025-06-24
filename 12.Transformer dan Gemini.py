from flask import Flask, render_template, request
import google.generativeai as genai
from config import Config
from utils import load_models, analyze_sentiment, get_sentiment_label
import time

app = Flask(__name__)

# Inisialisasi model
tokenizer, model = load_models()
genai.configure(api_key=Config.GEMINI_API_KEY)
gemini = genai.GenerativeModel('gemini-2.0-flash')

def analyze_with_rules(text, tokenizer, model):
    """Analisis dengan aturan tambahan"""
    from config import Config
    
    # Analisis dasar
    pred, confidence = analyze_sentiment(text, tokenizer, model)
    
    # Aturan tambahan berdasarkan kata kunci
    text_lower = text.lower()
    
    # Jika mengandung kata negatif, paksa ke Negatif
    if any(word in text_lower for word in Config.NEGATIVE_WORDS):
        return 2, max(confidence, 0.8)  # Pastikan confidence tinggi
    
    # Jika mengandung kata positif, paksa ke Positif
    if any(word in text_lower for word in Config.POSITIVE_WORDS):
        return 0, max(confidence, 0.8)
    
    return pred, confidence

def generate_ai_response(text, sentiment, confidence):
    """Membuat respon AI yang selaras dengan hasil analisis"""
    prompt = f"""
    [Instruksi]
    Berikan analisis untuk teks berikut dengan ketentuan:
    1. Hasil analisis sentimen: {sentiment} (Tingkat Kepercayaan: {confidence:.0%})
    2. Teks: "{text}"
    
    Format respons:
    - Analisis: [Jelaskan mengapa teks dikategorikan seperti hasil analisis]
    - Catatan: [Berikan catatan jika ada kemungkinan ketidaksesuaian]
    - Saran: [Berikan saran terkait sentimen yang terdeteksi]
    
    Gunakan Bahasa Indonesia yang formal dan jelas. Maksimal 5 kalimat.
    """
    
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            return render_template('gemini.html', error="Mohon masukkan teks untuk dianalisis")
        
        try:
            start_time = time.time()
            
            # Analisis dengan aturan tambahan
            pred, confidence = analyze_with_rules(text, tokenizer, model)
            sentiment, final_confidence = get_sentiment_label(pred, confidence)
            
            # Dapatkan respon AI
            ai_response = generate_ai_response(text, sentiment, final_confidence)
            
            processing_time = round(time.time() - start_time, 2)
            
            return render_template('gemini.html',
                                text=text,
                                sentiment=sentiment,
                                confidence=f"{final_confidence:.0%}",
                                ai_response=ai_response,
                                processing_time=processing_time)
        
        except Exception as e:
            return render_template('gemini.html', 
                                error=f"Terjadi kesalahan: {str(e)}",
                                text=text)
    
    return render_template('gemini.html')

if __name__ == '__main__':
    app.run(debug=True)