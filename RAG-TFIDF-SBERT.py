import re
import os
import nltk
import torch
import cohere
import random
import pickle
import numpy as np
import pandas as pd
nltk.download('stopwords')
from unidecode import unidecode
from sbertmodel import embedder
from nltk.corpus import stopwords
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
from IPython.display import clear_output
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# choosing the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Importng the SBERT embeddings
corpus_np = np.loadtxt("/project/egultep_31/RIGVEDA/experiments/sbert_queryembeddings.tsv", delimiter='\t')
# Importing the sukta labels
labels = pd.read_csv("/project/egultep_31/RIGVEDA/suktaspreprocessing/suktalabels.tsv", header=None)
# Importing the griffith's version Rigveda text 
grif_text = "/project/egultep_31/RIGVEDA/experiments/Rigveda - Suktas - Griffith's.txt"
# Importing the Jaimeson version processed text
fname = '/project/egultep_31/RIGVEDA/suktaspreprocessing/consuktasrigveda.txt'
# Generate an API Key from the cohere web
API_KEY = "api_key"

# loading the sbert embeddings
def load_sbert_embeddings(sbert_embeddings):
    sb_embeds = torch.tensor(sbert_embeddings)
    return sb_embeds

# read the griffith text
def read_griffith_text(old_text):
    with open(old_text, 'r') as gr_text:
        griffith_read_text = gr_text.readlines()
    return griffith_read_text

# reading the jaimeson text
def read_jami_text(jami_text):
    with open(jami_text, 'r', encoding = "utf-8") as j_text:
        jamison_text = j_text.read()
    return jamison_text

# Preprocessing the text
def preprocessing(raw_text):
    raw_text = raw_text.lower()
    raw_text = re.sub('[0-9]+', '', raw_text)
    raw_text = re.sub('—',' ', raw_text)
    raw_text = re.sub('–',' ', raw_text)
    pattern = r'[^\w\s]'
    clean_raw_text = re.sub(pattern, '', raw_text)
    clean_split_text = clean_raw_text.split('\n')
    sukta_stop_words = set(stopwords.words("english"))
    processed_text = [word for word in clean_split_text if word not in sukta_stop_words and len(word) > 2]
    return processed_text

# Process the sbert embeddings, griffith text, jamison text
corpus_embeddings = load_sbert_embeddings(corpus_np)
griffith_text = read_griffith_text(grif_text)
rigsuktatext = read_jami_text(fname)

# Define sukta tokenizer
def sukta_tokenizer(suktext):
    return suktext.split()

# Apply TFIDF Vectorizer
vectorizer = TfidfVectorizer(tokenizer=sukta_tokenizer,max_df=0.75, min_df=5, token_pattern=None, lowercase=False, strip_accents=None)
processed_text = preprocessing(rigsuktatext)
vectorizer.fit_transform(processed_text)

# Define an user query function to process the query
def user_query_function(queries: str, text, k, embedding_model, text_embeddings, sukta_labels):
    top_k = min(k, len(text))
    top_terms = []
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
            sorted_items = sorted(non_zero_items, key=lambda x: x[1], reverse=True)
            suktafeat_names = vectorizer.get_feature_names_out()
            top_terms = [suktafeat_names[idx] for idx, _ in sorted_items[:5]]
        if len(top_terms) >= 2:
            new_query_string = " ".join(top_terms)
        else:
            new_query_string = query_lower
        # Generate embedding for the final query string
        query_embedding = embedding_model.encode(new_query_string, convert_to_tensor=True).to(device)
        query_embedding = query_embedding.to(dtype=torch.float64)
        #  Compute similarity and fetch top-k results
        similarity_scores = embedding_model.similarity(query_embedding, text_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=top_k)
        for score, idx in zip(scores, indices):
            index = idx.tolist()
            label_num = sukta_labels.iloc[index, 0]
            text_dict.update({label_num: text[idx]})
            collected_text += "".join(text[idx])
    return collected_text, text_dict

# Function to generate the suktas summary 
def generate_suktas_summary(cohere_key):
    # importing the LLM model
    co = cohere.ClientV2(api_key=cohere_key)
    next_choice = False

    while not next_choice:
        user_input = input("Enter your query related to the Rigveda:\n").strip().lower()

        if not user_input:
            print("❗ Please enter a non-empty query.\n")
            continue

        query_list = [user_input]
        
        # User query function
        results, data_dict = user_query_function(
            queries=query_list,
            text=griffith_text,
            k=num_of_suktas,
            embedding_model=embedder,
            text_embeddings=corpus_embeddings,
            sukta_labels=labels
        )

        # Instructions to the LLM for generating the answer from the selected suktas and the query 
        message = f"""{user_input}.
        Instructions:
        
        Generate a concise and focused summary based only on the given Rigveda hymns.
        Do not use bullet points. The summary must be written as a continuous paragraph, using natural language.
        Do not start the summary with the phrase "The Rigveda hymns"; instead, return the summary content directly.
        
        Stay strictly on the topic of the user's query, and include only information that is contextually present in the hymns provided.
        
        You are only allowed to use knowledge derived from the Rigveda context.
        If the user's query is unrelated or no relevant information is found in the hymns, respond politely:
        
        "The entered query '<user_query>' is not relevant to the Rigveda context. Please enter a query related to the Rigveda."
        
        Do not generate any content that is not grounded in the given hymns.
        
        Example 1 — Query: "What is Creation"
        
        The origins of the universe are described as beginning in a state of neither existence nor non-existence, shrouded in darkness. 
        From this void emerged Desire, the primal force of creation. The hymns reference Hiranyagarbha, the golden womb, as the cosmic source of all, 
        along with deities like Savitar and Visvakarman who shaped the cosmos. The cycle of creation, sacrifice, and divine order is emphasized, 
        along with Indra’s role in defeating chaos and releasing the life-giving waters.
        
        Example 2 — Query: "What is computer science"
        
        The entered query "What is computer science" is not relevant to the Rigveda context. Please enter a query related to the Rigveda.
        
        \n{results}"""

        try:
            response = co.chat(
                model="command-a-03-2025",
                messages=[{"role": "user", "content": message}],
                temperature=0.0
            )
            llm_text = response.message.content[0].text.strip()
            print("\nSummary:\n")
            print(llm_text)

            if "not relevant to the rigveda context" not in llm_text.lower():
                print("\nMatched Suktas:\n")
                for key, value in data_dict.items():
                    print(f"Sukta {key}:\n{value}\n")

        except Exception as e:
            print(" An error occurred while contacting Cohere:", e)
            
        next_question = input("\n Would you like to ask another question? (yes/no): ").strip().lower()
        if next_question != "yes":
            next_choice = True

# No. of suktas to consider for generating the answer
num_of_suktas =10

# Enter the query to generate the answer
generate_suktas_summary(cohere_key=API_KEY)