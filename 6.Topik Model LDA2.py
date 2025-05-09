import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import string
from nltk.tokenize import word_tokenize
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download semua resources yang diperlukan dengan penanganan error
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Panggil fungsi download resources
download_nltk_resources()

# Data baru dari teks fiktif
data = """
Kecerdasan buatan (AI) telah membawa revolusi dalam berbagai sektor, termasuk kesehatan, pendidikan, dan transportasi.
Dengan algoritma pembelajaran mesin, sistem mampu mendeteksi pola dan membuat prediksi yang akurat.

Contohnya, rumah sakit kini menggunakan AI untuk menganalisis hasil radiologi dengan lebih cepat dan akurat.
Di bidang pendidikan, platform e-learning mengadaptasi materi berdasarkan gaya belajar masing-masing siswa.

Sementara itu, kendaraan otonom menggunakan sensor dan model AI untuk menghindari kecelakaan dan menavigasi lalu lintas dengan aman.
Penggunaan AI di masa depan diprediksi akan semakin meluas, didukung oleh peningkatan daya komputasi dan ketersediaan data besar.

Namun demikian, penerapan AI juga menimbulkan tantangan etis, seperti privasi data dan bias algoritmik, yang harus diatasi melalui regulasi yang bijak.
#AI #MachineLearning #Teknologi #InovasiDigital
"""

def preprocess_text(text):
    """Membersihkan dan memproses teks untuk bahasa Indonesia/Inggris"""
    try:
        tokens = word_tokenize(text.lower())
    except:
        tokens = text.lower().split()

    tokens = [token for token in tokens 
              if token not in string.punctuation 
              and not token.isdigit()
              and not token.startswith('#')]

    stop_words = set(stopwords.words('indonesian') + stopwords.words('english') +
                     ['contoh', 'seperti', 'dengan', 'untuk', 'pada', 'adalah',
                      'yang', 'di', 'dan', 'dalam', 'ke', 'nya'])
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        pass

    return tokens

def apply_lda(documents, num_topics=3, passes=10):
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                random_state=100,
                                passes=passes,
                                alpha='auto')

    return lda_model, corpus, dictionary

def main():
    print("Memulai proses analisis topik...\n")
    
    processed_docs = [preprocess_text(data)]
    
    lda_model, corpus, dictionary = apply_lda(processed_docs, num_topics=3)
    
    print("=== Hasil Analisis Topik ===")
    for idx, topic in lda_model.print_topics(-1):
        print(f"\nTopik {idx + 1}:")
        topic_terms = topic.split(' + ')
        for term in topic_terms[:10]:
            print(f"  {term}")
    
    print("\n=== Distribusi Topik dalam Dokumen ===")
    for doc_topics in lda_model[corpus]:
        for topic_num, prop_topic in doc_topics:
            print(f"Topik {topic_num + 1}: {prop_topic:.2%}")

if __name__ == "__main__":
    main()
