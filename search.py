from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LETOR
from util import text_preprocess

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]


letor_instance = LETOR(docs_dir = "data-train-letor/train", 
                  qrels_dir = "data-train-letor/qrels",
                  query_dir = "data-train-letor/query",
                  num_latent_topics = 200,
                  model_dir = "data-train-letor\\model")

dict_query_docs = dict()

print("#############################################")
print("BEFORE RERANKING")
print("#############################################")
for query in queries:
    query = " ".join(text_preprocess(query))
    dict_query_docs[query] = dict()

    print("Query  : ", query)
    print("Results:")
    # for (doc, score) in BSBI_instance.retrieve_tfidf(query, k = 10):
    for (doc, score) in BSBI_instance.retrieve_bm25(query, k = 100):
        print(f"{doc:30} {score:>.3f}")
        with open(doc, encoding="utf-8") as collection_file:
            dict_query_docs[query][doc] = text_preprocess(collection_file.read())
    print()

letor_instance.load_lsi()
letor_instance.load_lambdarank()

print("#############################################")
print("AFTER RERANKING")
print("#############################################")
for query, documents in dict_query_docs.items():
    unseen_data = []
    for doc in documents.values():
        unseen_data.append(letor_instance.features(query.split(), doc, 
                                            letor_instance.lsi_model, letor_instance.dictionary))
    
    scores = letor_instance.lambdarank_model.predict(unseen_data)
    did_scores = [x for x in zip([did for (did, _) in documents.items()], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    print("query        :", query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
        print(f"{did:30} {score:>.2f}")
