print("Semantic search script started...")
grif_text = "/project/egultep_31/RIGVEDA/experiments/Rigveda - Suktas - Griffith's.txt"

with open(grif_text, 'r') as grf_text:
    griffith_text = grf_text.readlines()

import torch
import pandas as pd
import numpy as np
import random
import os
from sentence_transformers import SentenceTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading SentenceTransformer...")
embedder = SentenceTransformer("all-mpnet-base-v2")
print("Model loaded.")

embedder.eval()

corpus_np = np.loadtxt("/project/egultep_31/RIGVEDA/experiments/sbert_queryembeddings.tsv", delimiter='\t')
corpus_embeddings = torch.tensor(corpus_np)

import cohere
co = cohere.ClientV2(api_key="ADD API KEY FROM COHERE MODEL")


def user_query_function(queries:str, text, k, embedding_model, text_embeddings, sukta_labels):
    top_k = min(k, len(text))
    collected_text = ""
    text_dict = {}
    text_embeddings = text_embeddings.to(device)
    for query in queries:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)
        query_embedding = query_embedding.to(dtype=torch.float64)
        similarity_scores = embedding_model.similarity(query_embedding, text_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=top_k)
        print("\nQuery:", query)
        print(f"Text summary generated from the relevant suktas in the Rigveda:")
        for score, idx in zip(scores, indices):
            index = idx.tolist()
            label_num = labels[0][index]
            text_dict.update({label_num: text[idx]})
            collected_text += "".join(text[idx])
    return collected_text, text_dict



labels = pd.read_csv("/project/egultep_31/RIGVEDA/suktaspreprocessing/suktalabels.tsv", header=None)

from IPython.display import clear_output

next_choice = False
user_input = input("Enter the query:\n").lower()
while not next_choice:
    clear_output(wait=True)
    query_list = [user_input]
    results, data_dict = user_query_function(query_list, griffith_text, 3, embedder, corpus_embeddings, labels)
    import cohere
    co = cohere.ClientV2(api_key="ADD API KEY FROM COHERE MODEL")
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