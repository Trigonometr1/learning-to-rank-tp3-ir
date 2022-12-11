# TAHAP REPRODUCE MODEL UNTUK TP 3

Tahap menjalankan:
1. Buat venv, kemudian install dari requirements.txt
```bash
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt
```
2. Jalankan `letor.py`. Ini penting untuk membuat model LSI dan LambdaMART yang akan digunakan untuk rerank

3. Jika ingin melihat dokumen apa saja yang dihasilkan dari BM25 serta hasil reranking, jalankan `search.py`. Jika ingin melihat hasil evaluasi, jalankan `experiment.py`

# LINK PENTING UNTUK TP 4
1. Paper BioClinicalBERT: https://arxiv.org/pdf/1904.03323.pdf
2. Lokasi model BioClinicalBERT di HuggingFace: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
3. Lokasi tempat penyimpanan untuk kebutuhan TP 4 (akan hilang pada 18 Desember 2022 karena kebijakan dari UI): https://drive.google.com/drive/folders/1yBbCyzty9SedKDeqIi9FYjHbLfBlJmDN?usp=share_link

# PENJELASAN TP 4
File yang penting pada TP4 adalah semua file ipynb berawalan TP4_, sisanya file-file dari TP3