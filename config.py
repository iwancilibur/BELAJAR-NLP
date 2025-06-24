import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    MODEL_NAME = 'indobenchmark/indobert-base-p1'
    POSITIVE_WORDS = ['bagus', 'baik', 'puas', 'senang', 'mantap', 'recommend']
    NEGATIVE_WORDS = ['buruk', 'jelek', 'tidak', 'jangan', 'gagal', 'kecewa']