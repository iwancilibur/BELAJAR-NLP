from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Persiapan Data
corpus = [
    "Saya belajar NLP",
    "NLP digunakan untuk pemrosesan bahasa",
    "Python adalah bahasa pemrograman populer untuk NLP"
]

# 2. Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# 3. Transformasi Teks ke Bentuk Numerik
X = tfidf.fit_transform(corpus)

# 4. Output Hasil
print("Vocabulary (Daftar Kata Unik):", tfidf.get_feature_names_out())
print("\nMatriks TF-IDF:\n", X.toarray())