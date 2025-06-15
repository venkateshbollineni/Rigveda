import os
import re
import nltk
import pickle
import numpy as np
nltk.download('stopwords')
from unidecode import unidecode
from nltk.corpus import stopwords
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import pandas as pd
import random
from sentence_transformers import SentenceTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels = pd.read_csv("/project/egultep_31/RIGVEDA/suktaspreprocessing/suktalabels.tsv", header=None)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedder = SentenceTransformer("all-mpnet-base-v2")
embedder.eval()

corpus_np = np.loadtxt("/project/egultep_31/RIGVEDA/experiments/sbert_queryembeddings.tsv", delimiter='\t')
corpus_embeddings = torch.tensor(corpus_np)

import cohere
co = cohere.ClientV2(api_key="API KEY")

sukta_stop_words = set(stopwords.words("english"))

grif_text = "/project/egultep_31/RIGVEDA/experiments/Rigveda - Suktas - Griffith's.txt"

with open(grif_text, 'r') as grf_text:
    griffith_text = grf_text.readlines()

fname = '/project/egultep_31/RIGVEDA/suktaspreprocessing/consuktasrigveda.txt'

lsa_sukta_data = ""

with open(fname, 'r', encoding='utf-8') as f:
        rigsuktatext = f.read()

rigsuktatext = rigsuktatext.lower()
rigsuktatext = re.sub('[0-9]+', '', rigsuktatext)
rigsuktatext = re.sub('—',' ', rigsuktatext)
rigsuktatext = re.sub('–',' ', rigsuktatext)
pattern = r'[^\w\s]'
cleaned_string = re.sub(pattern, '', rigsuktatext)
filt2_paragraphs = cleaned_string.split('\n')

suktafiltered_words = [word for word in filt2_paragraphs if word not in sukta_stop_words and len(word) > 2]

def sukta_tokenizer(suktext):
    return suktext.split()

vectorizer = TfidfVectorizer(tokenizer=sukta_tokenizer,max_df=0.75, min_df=5, token_pattern=None, lowercase=False, strip_accents=None)
X = vectorizer.fit_transform(suktafiltered_words)


def user_query_function(queries: str, text, k, embedding_model, text_embeddings, sukta_labels):
    top_k = min(k, len(text))
    collected_text = ""
    text_dict = {}
    text_embeddings = text_embeddings.to(device)

    for query in queries:
        query_lower = query.lower()
        # TF-IDF transformation
        query_trans = vectorizer.transform([query_lower])
        # Extract non-zero tf-idf scores and their feature indices
        non_zero_items = list(zip(query_trans.indices, query_trans.data))  # (feature_idx, tfidf_score)
        if non_zero_items:
            print(" Using TF-IDF refined query...")
            sorted_items = sorted(non_zero_items, key=lambda x: x[1], reverse=True)
            suktafeat_names = vectorizer.get_feature_names_out()
            top_terms = [suktafeat_names[idx] for idx, _ in sorted_items[:5]]  
            new_query_string = " ".join(top_terms)
            print(f"Top TF-IDF terms from query: {top_terms}")
        else:
            print("No TF-IDF vocabulary match found — using original query directly for embeddings.")
            new_query_string = query_lower
        # Generate embedding for the final query string
        query_embedding = embedding_model.encode(new_query_string, convert_to_tensor=True).to(device)
        query_embedding = query_embedding.to(dtype=torch.float64)
        #  Compute similarity and fetch top-k results
        similarity_scores = embedding_model.similarity(query_embedding, text_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=top_k)
        print("\n Query:", query)
        print(f"🔎 Text summary generated from top {top_k} relevant suktas in the Rigveda:\n")
        for score, idx in zip(scores, indices):
            index = idx.tolist()
            label_num = sukta_labels[0][index]
            text_dict.update({label_num: text[idx]})
            collected_text += "".join(text[idx])
    return collected_text, text_dict



from IPython.display import clear_output

next_choice = False
user_input = input("Enter the query:\n").lower()
while not next_choice:
    clear_output(wait=True)
    query_list = [user_input]  
    results, data_dict = user_query_function(query_list, griffith_text, 10 , embedder, corpus_embeddings, labels)
    import cohere 
    co = cohere.ClientV2(api_key="API KEY")
    message = f"{user_input}.Generate a concise summary of the given Rigveda hymns as bullet points, don't start the sentence with the Rigveda hymns instead return the summary only. Always stay on the initial question and choose only contextually needed information.\n{results}"
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": message}],
        temperature=0.0
    )
    print(response.message.content[0].text)
    print('\n\n')
    print(f"The above text has been generated from the following Rigveda Suktas:\n")
    for key, value in data_dict.items():
        print(f"Sukta {key}: \n{value}")
    next_question = input("Would you like to enquire next question? Enter 'yes' or 'no'").lower()
    if next_question == "no":
        next_choice = True
        clear_output(wait=True)
    else:
        user_input = input("Enter the query:\n").lower()
