import re
from collections import defaultdict
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# 1. Data Preparation
documents = {
    1: "Python adalah bahasa pemrograman populer untuk NLP",
    2: "NLP menggunakan Python dan pemrosesan bahasa alami",
    3: "Information retrieval adalah sistem pencari dokumen",
    4: "Pemrograman Python dan NLP digunakan untuk text mining"
}

# 2. Preprocessing (FIXED THE SYNTAX ERROR HERE)
stop_words = set(stopwords.words('indonesian')).union({'untuk', 'dan', 'adalah'})
processed_docs = {}
for doc_id, text in documents.items():
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    processed_docs[doc_id] = tokens

# 3. Inverted Index
inverted_index = defaultdict(list)
for doc_id, tokens in processed_docs.items():
    for token in set(tokens):
        inverted_index[token].append(doc_id)

print("=== INVERTED INDEX ===")
for term, doc_ids in inverted_index.items():
    print(f"{term}: {doc_ids}")

# 4. Boolean Search Function
def boolean_search(query):
    query_terms = [term for term in word_tokenize(query.lower()) 
                  if term.isalpha() and term not in stop_words]
    
    if not query_terms:
        return set()
    
    result = set(inverted_index.get(query_terms[0], []))
    for term in query_terms[1:]:
        result.intersection_update(inverted_index.get(term, []))
    return result

# 5. BM25 Ranking
corpus = [' '.join(tokens) for tokens in processed_docs.values()]
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_search(query, top_n=2):
    tokenized_query = word_tokenize(query.lower())
    tokenized_query = [t for t in tokenized_query if t.isalpha() and t not in stop_words]
    
    scores = bm25.get_scores(tokenized_query)
    ranked_docs = sorted(zip(documents.keys(), scores), key=lambda x: x[1], reverse=True)
    return ranked_docs[:top_n]

# 6. Example Usage
print("\n=== SEARCH RESULTS ===")
query = "Python NLP"
print(f"\nBoolean Search for '{query}':")
boolean_results = boolean_search(query)
for doc_id in boolean_results:
    print(f"Document {doc_id}: {documents[doc_id]}")

print(f"\nBM25 Ranking for '{query}':")
bm25_results = bm25_search(query)
for doc_id, score in bm25_results:
    print(f"Score {score:.2f}: Document {doc_id} - {documents[doc_id]}")