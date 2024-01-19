from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset
import time
import re
import numpy.linalg as la

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
        
        self.embeddings_dict = {}
        with open("glove.6B.50d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector


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
                
    def get_glove_embedding(self, word, embedding_dimension=50):
        """
        Retrieve glove embedding of a specific dimension
        """

        # Implement your code here
        if word in self.embeddings_dict: 
            return self.embeddings_dict[word]
        else:
            return np.zeros(len(self.get_glove_embedding("word")))
    
    def get_averaged_glove_embeddings(self, sentence):

        words = sentence.split(" ")
        glove_embedding = np.zeros(len(self.get_glove_embedding("word")))

        count_words = 0
        for word in words:
            glove_embedding = np.add(glove_embedding, self.get_glove_embedding(word))

        #print(glove_embedding)
        #print(glove_embedding/len(words))

        return glove_embedding/len(words)
    
    def test(self):
        self.encode_with_glove(self.queries)
        self.encode_with_glove(self.documents)
        
    def clean_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z0-9]', ' ', sentence)
        sentence = re.sub('\s+', ' ', sentence)
        return sentence
        
    def encode_with_glove(self, sentences):
        """
        Encodes sentences by averaging GloVe 50d vectors of words in each sentence.
        Return a sequence of embeddings of the sentences.
        Download the glove vectors from here. 
        https://nlp.stanford.edu/data/glove.6B.zip
        """
        #TODO Put your code here. 
        ###########################################################################
#         print(len(sentences))
#         sentence=self.clean_sentence(sentences[0])
#         print(sentence)

#         sentence=self.clean_sentence(sentences[0])
#         vector=self.get_averaged_glove_embeddings(sentence)
#         print(vector)
        vectors=[]
        for sentence in sentences:
            sentence=self.clean_sentence(sentence)
            vectors.append(self.get_averaged_glove_embeddings(sentence))
#         print(len(vectors))
        return vectors
       
        ###########################################################################
        
    def cosine_similarity(self, x, y):

        return np.dot(x,y)/max(la.norm(x)*la.norm(y),1e-3)

    def rank_documents(self, encoding_method='sentence_transformer'):
        """
        (1) Compute cosine similarity between each document and the query
        (2) Rank documents for each query and save the results in a dictionary "query_id_to_ranked_doc_ids" 
            This will be used in "mean_average_precision"
            Example format {2: [125, 673], 35: [900, 822]}
        """
        if encoding_method == 'glove':
            query_embeddings = self.encode_with_glove(self.queries)
            document_embeddings = self.encode_with_glove(self.documents)
        elif encoding_method == 'sentence_transformer':
            query_embeddings = self.model.encode(self.queries)
            document_embeddings = self.model.encode(self.documents)
        else:
            raise ValueError("Invalid encoding method. Choose 'glove' or 'sentence_transformer'.")
        
        #TODO Put your code here.
        ###########################################################################
        self.query_id_to_ranked_doc_ids={}
        for i, query_id in enumerate(self.query_ids):
            query_embedding=query_embeddings[i]
            cosine_similarities=[]
            for j, document_id in enumerate(self.document_ids):
                document_embedding=document_embeddings[j]
                cosine_similarities.append((self.cosine_similarity(query_embedding, document_embedding), document_id))
            cosine_similarities.sort(reverse=True)
            top_10_document_ids=[x[1] for x in cosine_similarities[:10]]
            self.query_id_to_ranked_doc_ids[query_id]=top_10_document_ids
                
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
        sum_precisions=0
        for query_id in self.query_ids:
            sum_precisions+=self.average_precision(self.query_id_to_relevant_doc_ids[query_id], self.query_id_to_ranked_doc_ids[query_id])
            
        return sum_precisions/len(self.query_ids)
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
        cosine_similarities=[]
        for j, document_id in enumerate(self.document_ids):
            document_embedding=document_embeddings[j]
            cosine_similarities.append((self.cosine_similarity(query_embedding, document_embedding), document_id))
        cosine_similarities.sort(reverse=True)
        top_10_document_ids=[x[1] for x in cosine_similarities[:10]]
        top_10_documents=[doc for did, doc in zip(self.document_ids, self.documents) if did in top_10_document_ids]
        for document in top_10_documents:
            print(document)
          
        ###########################################################################


# Initialize and use the model
model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels")
# model.test()

print("Ranking with sentence_transformer...")
model.rank_documents(encoding_method='sentence_transformer')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


print("Ranking with glove...")
model.rank_documents(encoding_method='glove')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


model.show_ranking_documents("Breast Cancer Cells Feed on Cholesterol")