import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import chromadb
import re
# import pickle
# import pandas as pd
# import sklearn
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# from sklearn.metrics.pairwise import cosine_similarity

# def filter_similar_records(query, df, tfidf_vectorizer, tfidf_matrix, threshold=None, top_n=None):
#     # Transform the query using the same TF-IDF vectorizer
#     query_tfidf = tfidf_vectorizer.transform([query])

#     # Calculate cosine similarity between the query and all documents
#     cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

#     # Add cosine similarity scores as a new column in the DataFrame
#     df['cosine_similarity'] = cosine_similarities

#     # Sort the DataFrame based on cosine similarity scores
#     df = df.sort_values(by='cosine_similarity', ascending=False).reset_index(drop=True)

#     # Filter based on threshold similarity score or select top N similar records
#     if threshold is not None:
#         filtered_df = df[df['cosine_similarity'] >= threshold]
#     elif top_n is not None:
#         filtered_df = df.head(top_n)
#     else:
#         filtered_df = df

#     return filtered_df
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# tfidf = pickle.load(r'.\tfidf_vectorizer.pkl')

# def preprocess(raw_text):
#     lowered = clean(raw_text)

#     tokens = lowered.split()
    
#     # Lemmatization
#     clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     return [" ".join(clean_tokens)]

def embeddeing(text):
    encoding = tokenizer.batch_encode_plus(
        [text],                    # List of input texts
        padding=True,              # Pad to the maximum sequence length
        truncation=True,           # Truncate to the maximum sequence length if necessary
        return_tensors='pt',      # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state
    encoded_text = tokenizer.encode(text, return_tensors='pt')
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    tokenized_text = tokenizer.tokenize(decoded_text)
    sentence_embedding = word_embeddings.mean(dim=1)

    return sentence_embedding.tolist()[0]

def clean(text):
    text = text.lower()
    return re.sub("[^a-z A-Z]", "",text)
    
chroma_client = chromadb.PersistentClient(path="./DB")
collection = chroma_client.get_collection(name="Movies")

st.header('Enhancing Search Engine Relevance for Video Subtitles')

option = st.selectbox('Word Embedding for Search Engine',('TF-IDF', 'BERT'))

if option == 'BERT':
    text = clean(st.text_area('Search with BERT',placeholder='Enter text'))
    number = st.number_input('number of result',min_value=5,max_value=20)
    search = st.button('Search')

    if search:
        query_emd = embeddeing(text)
        result = collection.query(
            query_embeddings=[query_emd],
            n_results=number,
        )
        for i in result['metadatas'][-1]:
            st.write(i['Movie ID'],i['name'])
# elif option == 'TF-IDF':
    # text = clean(st.text_area('Search with TF-IDF',placeholder='Enter text'))
    # number = st.number_input('number of result',min_value=5,max_value=20)
    # search = st.button('Search')

    # if search:
    #     query_emd = preprocess(text)
    #     result = collection.query(
    #         query_embeddings=[query_emd],
    #         n_results=number,
    #     )
    #     for i in result['metadatas'][-1]:
    #         st.write(i['Movie ID'],i['name'])