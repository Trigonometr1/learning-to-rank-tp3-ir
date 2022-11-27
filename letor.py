import random
import os
import lightgbm as lgb
import numpy as np
import pickle

from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from tqdm import tqdm

class LETOR:
    """
    Attributes
    ----------
    docs_dir(str): Menyimpan direktori untuk kumpulan dokumen yang akan di train
    qrels_dir(str): Menyimpan direktori untuk kumpulan qrel yang akan di train
    query_dir(str): Menyimpan direktori untuk kumpulan query yang akan di train
    NUM_LATENT_TOPICS(int): Jumlah fitur yang akan digunakan di model.
                            Sebaiknya diset menjadi <= jumlah fitur awal
    model_dir(str): Menyimpan direktori untuk model hasil train
    lambdarank_model(LGBMRanker): Model LambdaMART
    dictionary(gensim.corpora.Dictionary): Menyimpan pengaturan doc2bow.
    lsi_model(LsiModel): Untuk mengambil representasi vektor dari dokumen
    """
    def __init__(self, docs_dir, qrels_dir, query_dir, num_latent_topics, model_dir):
        self.docs_dir = docs_dir
        self.qrels_dir = qrels_dir
        self.query_dir = query_dir
        self.NUM_LATENT_TOPICS = num_latent_topics
        self.model_dir = model_dir
        self.lambdarank_model = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)
    
    def prepare_data(self):
        """
        Membuat dataset dengan format yang dapat diterima LightGBM LambdaRank
            Returns
            -------
            documents : Dict
                Python dictionary berisikan key (doc_id) dan value (konten di dokumen)
            group_qid_count : List[int]
                Dibutuhkan untuk parameter LambdaMART. Isinya adalah berapa banyak
                pasangan query-rel untuk satu query_id.
            dataset : List[Tuple[str, str, int]]
                format dataset: [(query_text, document_text, relevance), ...]
        """
        NUM_NEGATIVES = 1

        docs_file_path = os.path.join(self.docs_dir, 'train.docs')
        qrels_file_path = os.path.join(self.qrels_dir, 'train.3-2-1.qrel')
        query_file_path = os.path.join(self.query_dir, 'train.vid-desc.queries')

        # Buat dahulu dictionary berisi key (nama) dan value (konten)
        # untuk file berisikan data dokumen dan query.
        documents = {}
        with open(docs_file_path, encoding='utf-8') as file:
            for line in file:
                doc_id, content = line.split("\t")
                documents[doc_id] = content.split()

        queries = {}
        with open(query_file_path, encoding='utf-8') as file:
            for line in file:
                q_id, content = line.split("\t")
                queries[q_id] = content.split()

        q_docs_rel = {} # grouping by q_id terlebih dahulu
        with open(qrels_file_path, encoding='utf-8') as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in queries) and (doc_id in documents):
                    if q_id not in q_docs_rel:
                        q_docs_rel[q_id] = []
                    q_docs_rel[q_id].append((doc_id, int(rel)))

        # group_qid_count untuk model LGBMRanker
        group_qid_count = []
        # format dataset: [(query_text, document_text, relevance), ...]
        dataset = []
        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                dataset.append((queries[q_id], documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            dataset.append((queries[q_id], random.choice(list(documents.values())), 0))
        
        return documents, group_qid_count, dataset

    def features(self, query, doc, model, dictionary):
        """
        Mengubah pasangan query-doc menjadi vektor numerik dense.
        Informasi tambahan berupa cosine similarity dan jaccard (opsional).
        Fungsi ini digunakan juga untuk mengubah data yang akan diprediksi (unseen data)
            Parameters
            ----------
            query : List[str]
                Python list berisikan potongan token di query
            doc : List[str]
                Python list berisikan potongan token di dokumen
            model : LsiModel
                model LSI yang sudah di build dengan fungsi build_lsi
            dictionary : gensim.corpora.Dictionary
                Dictionary doc2bow yang sudah di build dengan fungsi build_lsi

            Returns
            -------
            Python List berisikan representasi vektor pasangan query-doc
        """
        def vector_rep(text, model=model, dictionary=dictionary):
            rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
            return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS
        
        v_q = vector_rep(query)
        v_d = vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def build_lsi(self, documents):
        """
        Membuat model LSI yang menyimpan informasi seputar representasi vektor dari query-doc
            Parameters
            ----------
            documents : Dict
                Python dictionary berisikan key (doc_id) dan value (konten di dokumen)

            Returns
            -------
            model : LsiModel
                model LSI yang sudah di build
            dictionary : gensim.corpora.Dictionary
                Dictionary doc2bow yang sudah di build
        """
        self.dictionary = Dictionary()
        print("Changing to Bag-of-Words format...")
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in tqdm(documents.values())]
        self.lsi_model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS)

        return self.dictionary, self.lsi_model
    
    def create_feature(self, dataset, model, dictionary):
        """
        Mengubah dataset ke format yang dapat diterima LamdbdaMART
            Parameters
            ----------
            dataset : List
                list berisikan tuple berformat (query, doc, relevance)
            model : LsiModel
                model LSI yang sudah di build dengan fungsi build_lsi
            dictionary : gensim.corpora.Dictionary
                Dictionary doc2bow yang sudah di build dengan fungsi build_lsi

            Returns
            -------
            X, Y : numpy.array
                X berisi vektor representasi query-doc , Y berisi level relevance
        """
        X = []
        Y = []
        print("Creating features for LambdaRank...")
        for (query, doc, rel) in tqdm(dataset):
            X.append(self.features(query, doc, model, dictionary))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def train_lambdarank(self, X, Y, group_qid_count):
        """
        Melakukan train model LamdbdaMART
            Parameters
            ----------
            X, Y : numpy.array
                X berisi vektor representasi query-doc , Y berisi level relevance
            group_qid_count : List[int]
                dihasilkan dari metode prepare_data

            Returns
            -------
            model LambdaMART yang sudah dilatih
        """
        self.lambdarank_model.fit(X, Y,
                group = group_qid_count,
                verbose = 10)

        return self.lambdarank_model

    def save_lambdarank(self):
        with open(os.path.join(self.model_dir, 'lambdarank-model.pkl'), 'wb') as f:
            pickle.dump(self.lambdarank_model, f)
    
    def load_lambdarank(self):
        with open(os.path.join(self.model_dir, 'lambdarank-model.pkl'), 'rb') as f:
            self.lambdarank_model = pickle.load(f)

    def save_lsi(self):
        with open(os.path.join(self.model_dir, 'lsi-model.pkl'), 'wb') as f:
            pickle.dump(self.lsi_model, f)
        with open(os.path.join(self.model_dir, 'lsi-dictionary.dict'), 'wb') as f:
            pickle.dump(self.dictionary, f)
    
    def load_lsi(self):
        with open(os.path.join(self.model_dir, 'lsi-model.pkl'), 'rb') as f:
            self.lsi_model = pickle.load(f)
        with open(os.path.join(self.model_dir, 'lsi-dictionary.dict'), 'rb') as f:
            self.dictionary = pickle.load(f)

if __name__ == "__main__":
    letor = LETOR(docs_dir = "data-train-letor/train", 
                  qrels_dir = "data-train-letor/qrels",
                  query_dir = "data-train-letor/query",
                  num_latent_topics = 200,
                  model_dir = "data-train-letor\\model")

    documents, group_qid_count, dataset = letor.prepare_data()
    print("number of Q-D pairs:", len(dataset))
    print("group_qid_count:", group_qid_count)
    assert sum(group_qid_count) == len(dataset), "ada yang salah"
    print(dataset[:2])

    dictionary, lsi_model = letor.build_lsi(documents)

    X, Y = letor.create_feature(dataset, lsi_model, dictionary)
    print(X.shape)
    print(Y.shape)

    lambdarank_model = letor.train_lambdarank(X, Y, group_qid_count)

    # cek apakah model sudah dapat memprediksi
    query = "how much cancer risk can be avoided through lifestyle change ?"

    docs =[("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"), 
        ("D2", "study hard as your blood boils"), 
        ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
        ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
        ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]

    X_unseen = []
    for doc_id, doc in docs:
        X_unseen.append(letor.features(query.split(), doc.split(), lsi_model, dictionary))

    X_unseen = np.array(X_unseen)
    scores = letor.lambdarank_model.predict(X_unseen)
    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    print("query        :", query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
        print(did, score)

    # jika sudah, simpan modelnya untuk memprediksi
    letor.save_lsi()
    letor.save_lambdarank()