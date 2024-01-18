from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset
import time

class TextSimilarityModel:
    def __init__(self, corpus_name, rel_name, model_name='all-MiniLM-L6-v2', top_k=10):
        """
        Initialize the model with datasets and pre-trained sentence transformer.
        """
        self.model = SentenceTransformer(model_name)
        self.corpus_name = corpus_name
        self.rel_name = rel_name
        self.top_k = top_k
        self.load_data()


    def load_data(self):
        """
        Load and filter datasets based on test queries and documents.
        """
        # Load query and document datasets
        dataset_queries = load_dataset(self.corpus_name, "queries")
        dataset_docs = load_dataset(self.corpus_name, "corpus")

        # Extract queries and documents
        self.queries = dataset_queries["queries"]["text"]
        self.query_ids = dataset_queries["queries"]["_id"]
        self.documents = dataset_docs["corpus"]["text"]
        self.document_ids = dataset_docs["corpus"]["_id"]

        # Filter queries and documents based on test set
        qrels = load_dataset(self.rel_name)["test"]
        self.filtered_query_ids = set(qrels["query-id"])
        self.filtered_doc_ids = set(qrels["corpus-id"])

        self.queries = [q for qid, q in zip(self.query_ids, self.queries) if qid in self.filtered_query_ids]
        self.query_ids = [qid for qid in self.query_ids if qid in self.filtered_query_ids]
        self.documents = [doc for did, doc in zip(self.document_ids, self.documents) if did in self.filtered_doc_ids]
        self.document_ids = [did for did in self.document_ids if did in self.filtered_doc_ids]

        self.query_id_to_relevant_doc_ids = {qid: [] for qid in self.filtered_query_ids}
        for qid, doc_id in zip(qrels["query-id"], qrels["corpus-id"]):
            if qid in self.query_id_to_relevant_doc_ids:
                self.query_id_to_relevant_doc_ids[qid].append(doc_id)

    def encode_with_glove(self, glove_file_path, sentences):
        """
        Encodes sentences by averaging GloVe 50d vectors of words in each sentence.
        Return a sequence of embeddings of the sentences.
        Download the glove vectors from here. 
        https://nlp.stanford.edu/data/glove.6B.zip
        """
        #TODO Put your code here. 
        ###########################################################################
       
        ###########################################################################

    def rank_documents(self, encoding_method='sentence_transformer'):
        """
        (1) Compute cosine similarity between each document and the query
        (2) Rank documents for each query and save the results in a dictionary "query_id_to_ranked_doc_ids" 
            This will be used in "mean_average_precision"
            Example format {2: [125, 673], 35: [900, 822]}
        """
        if encoding_method == 'glove':
            query_embeddings = self.encode_with_glove("glove.6B.50d.txt", self.queries)
            document_embeddings = self.encode_with_glove("glove.6B.50d.txt", self.documents)
        elif encoding_method == 'sentence_transformer':
            query_embeddings = self.model.encode(self.queries)
            document_embeddings = self.model.encode(self.documents)
        else:
            raise ValueError("Invalid encoding method. Choose 'glove' or 'sentence_transformer'.")
        
        #TODO Put your code here.
        ###########################################################################
      
        ###########################################################################

    @staticmethod
    def average_precision(relevant_docs, candidate_docs):
        """
        Compute average precision for a single query.
        """
        y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
        precisions = [np.mean(y_true[:k+1]) for k in range(len(y_true)) if y_true[k]]
        return np.mean(precisions) if precisions else 0

    def mean_average_precision(self):
        """
        Compute mean average precision for all queries using the "average_precision" function.
        A reference: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
        """
         #TODO Put your code here. 
        ###########################################################################
        
        ###########################################################################

    def show_ranking_documents(self, example_query):
        """
        (1) rank documents with given query with cosine similaritiy scores
        (2) prints the top 10 results along with its similarity score.
        
        """
        #TODO Put your code here. 
        query_embedding = self.model.encode(example_query)
        document_embeddings = self.model.encode(self.documents)
        ###########################################################################
      
        ###########################################################################


# Initialize and use the model
model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels")

print("Ranking with sentence_transformer...")
model.rank_documents(encoding_method='sentence_transformer')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


print("Ranking with glove...")
model.rank_documents(encoding_method='glove')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


model.show_ranking_documents("Breast Cancer Cells Feed on Cholesterol")