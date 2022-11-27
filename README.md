Tahap menjalankan:
1. Buat venv, kemudian install dari requirements.txt
```bash
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt
```
2. Jalankan `letor.py`. Ini penting untuk membuat model LSI dan LambdaMART yang akan digunakan untuk rerank

3. Jika ingin melihat dokumen apa saja yang dihasilkan dari BM25 serta hasil reranking, jalankan `search.py`. Jika ingin melihat hasil evaluasi, jalankan `experiment.py`