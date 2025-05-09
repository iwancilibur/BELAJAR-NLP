import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Download semua resource yang diperlukan
nltk.download('punkt')
nltk.download('punkt_tab')

# Stopword bahasa Indonesia (sederhana)
stopwords_id = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'atau', 
    'tidak', 'dengan', 'untuk', 'pada', 'adalah', 'juga'
}

def preprocess_text_id(text):
    # Case folding
    text = text.lower()
    # Hapus tanda baca/angka
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenisasi
    tokens = word_tokenize(text, language='english')  # Gunakan tokenizer Inggris
    # Stopword removal
    tokens = [word for word in tokens if word not in stopwords_id]
    
    # Hitung statistik
    stats = {
        'jumlah_kata': len(tokens),
        'kata_unik': len(set(tokens)),
        'kata_terbanyak': Counter(tokens).most_common(3),
        'daftar_kata': list(set(tokens))
    }
    return " ".join(tokens), stats

# Contoh teks berita
berita = """
Pemerintah hari ini mengumumkan bantuan sosial sebesar Rp 10 triliun untuk UMKM. 
Bantuan ini diberikan kepada pelaku usaha yang terdampak krisis ekonomi. 
Menteri Keuangan mengatakan bantuan akan cair mulai bulan depan.
"""

# Proses teks
cleaned_text, hasil = preprocess_text_id(berita)

# Hasil
print("=== TEKS BERITA ASLI ===")
print(berita)

print("\n=== HASIL PREPROCESSING ===")
print(cleaned_text)

print("\n=== ANALISIS TEKS ===")
print(f"Jumlah kata total: {hasil['jumlah_kata']}")
print(f"Jumlah kata unik: {hasil['kata_unik']}")
print(f"3 kata terbanyak: {hasil['kata_terbanyak']}")
print("Daftar kata unik:", ", ".join(hasil['daftar_kata']))