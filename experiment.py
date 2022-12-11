import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings
from util import text_preprocess
from letor import LETOR

######## >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
  """ menghitung search effectiveness metric score dengan 
      Discounted Cumulative Gain

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score DCG
  """
  # TODO
  score = 0
  for i in range(len(ranking)):
    score += ranking[i]/math.log2(i+2)
  return score

def ap(ranking):
  """ menghitung search effectiveness metric score dengan 
      Average Precision

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score AP
  """
  # TODO
  def prec_k(k):
    score = 0
    for i in range(k):
      score += ranking[i]/(k)
    return score
    
  R = sum(ranking) if sum(ranking) > 0 else 1

  score = 0
  for k in range(len(ranking)):
    score += (prec_k(k+1)*ranking[k])/R
  return score

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 100, reranking = True):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-100 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  dict_query_docs = dict()

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])
      dict_query_docs[qid] = dict()

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (doc, score) in BSBI_instance.retrieve_bm25(query, k = k, k1 = 2, b = 0.75):
          did = int(re.search(r'.*\\.*\\(.*)\.txt', doc).group(1))
          if reranking:
            with open(doc, encoding="utf-8") as collection_file:
              dict_query_docs[qid][did] = text_preprocess(collection_file.read())
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))
  print(f"Hasil evaluasi BM25 (k1 = 2, b = 0.75, k = {k}) terhadap 30 queries")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("AP score  =", sum(ap_scores) / len(ap_scores))

  if reranking:
    letor_instance = LETOR(docs_dir = "data-train-letor/train", 
                  qrels_dir = "data-train-letor/qrels",
                  query_dir = "data-train-letor/query",
                  num_latent_topics = 200,
                  model_dir = "data-train-letor\\model")
    
    letor_instance.load_lsi()
    letor_instance.load_lambdarank()

    rbp_scores = []
    dcg_scores = []
    ap_scores = []

    for qid, documents in dict_query_docs.items():
      unseen_data = []
      ranking = []
      for did, doc in documents.items():
        unseen_data.append(letor_instance.features(query.split(), doc, 
                                              letor_instance.lsi_model,
                                              letor_instance.dictionary))
        
    
      scores = letor_instance.lambdarank_model.predict(unseen_data)
      did_scores = [x for x in zip([did for (did, _) in documents.items()], scores)]
      sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

      ranking.extend([qrels[qid][did] for (did, _) in sorted_did_scores])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

    print("Hasil evaluasi reranking dengan LAMBDARank")
    print("RBP score =", sum(rbp_scores) / len(rbp_scores))
    print("DCG score =", sum(dcg_scores) / len(dcg_scores))
    print("AP score  =", sum(ap_scores) / len(ap_scores))
  

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  for k in [5,10,50,100,500,1000]:
    eval(qrels, k = k)
    print("##############################")