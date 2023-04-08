import pandas as pd
# import numpy as np
# import re
import gensim
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from gensim import corpora, models, similarities
# import pyLDAvis
# from pprint import pprint
import pickle 
# import os
import matplotlib.pyplot as plt
# from gensim.models import CoherenceModel
# from gensim.test.utils import datapath
# from gensim.models.ldamodel import LdaModel
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

#load data
with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)
corpus = gensim.corpora.MmCorpus('corpus.mm')
data = pd.read_csv('dataset\DatasetLegal.csv')
corpus_lda = lda_model[corpus]
with open('id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)
index = similarities.MatrixSimilarity(corpus_lda, num_features=len(id2word))


topic_dict = {
    0 : "Sentence",
    1 : "Family",
    2 : "Criminal",
    3 : "Litigation",
    4 : "Succession",
    5 : "Contract",
    6 : "Labor"
}

# preprocessing new document data
def preprocess(text):
    stopwords = list(thai_stopwords())
    read_stopwords = pd.read_csv('dataset/add_stopwords.csv')
    add_stopwords = read_stopwords['stopword'].values.tolist()
    result = []
    str_text = str(text).replace(' ','')
    word_token = word_tokenize(str_text, engine='newmm')
    for word in word_token:
        if(word not in stopwords + add_stopwords):
            result.append(word)
        #result = map(lambda x: re.sub('[,/.?# ]', '', x), result)
    return result

# convert text from new document to bag of word
def bow(text):
  vector = id2word.doc2bow(text)
  return vector

#plot distribution of the new document
def plot_topic_distribution(topics):
    plt.figure(figsize=(8,6))
    plt.bar([t[0] for t in topics], [t[1] for t in topics])
    plt.xlabel('Topic ID')
    plt.ylabel('Topic Proportion')
    plt.title(f'Topic Distribution for new document')
    st.pyplot()

def find_similar_docs(lda_model, corpus, index, new_doc_topics, data):
    # sims = index[corpus_lda]
    # sims_sorted = sorted(enumerate(list(filter(lambda item: type(item)!=bool, sims))))
    
    # st.write(f"Topic distribution for new document: {new_doc_topics}")
    # st.write(f"{new_doc}\n")
    # for doc_id, similarity in sims_sorted[:5]:
    #     st.write(f"Document ID: {doc_id}, Similarity score: {similarity}")
    #     st.write(data.answer[doc_id])
    #     st.write("Topic distribution for similar document:")
    #     for num, dis in lda_model[corpus[doc_id]]:
    #         st.write(f"\t({topic_dict.get(num)}, {'%.5f' %dis})")

    sims = index[new_doc_topics]
    sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
    st.write(f"Topic distribution for new document : {new_doc_topics}\n{new_doc}\n")
    for doc_id, similarity in sims_sorted[:5]:
        st.write(f"Document ID: {doc_id}, Similarity score: {similarity}")
        st.write(data.answer[doc_id])
        st.write("Topic distribution for similar document : ")
        for num, dis in corpus_lda[doc_id]:
            st.write(f"\t({topic_dict.get(num)}, {'%.5f' %dis})")

#input from user
input_doc = st.text_input("Enter your document : ")
if(input_doc):
    new_doc = preprocess(input_doc)
    new_doc = bow(new_doc)
    new_doc_topics = lda_model.get_document_topics(new_doc)

    # plot_topic_distribution(new_doc_topics)
    find_similar_docs(lda_model, corpus, index, new_doc_topics, data)


    # sims = index[corpus_lda]
    # st.write(f"Topic distribution for new document: {new_doc_topics}")
    # st.write(f"{new_doc}\n")
    # sims_sorted = sorted(enumerate(list(filter(lambda item: type(item)!=bool, sims))))
    # for doc_id, similarity in sims_sorted[:5]:
    #     st.write(f"Document ID: {doc_id}, Similarity score: {similarity}")
    #     st.write(data.answer[doc_id])
    #     st.write("Topic distribution for similar document:")
    #     for num, dis in lda_model[corpus[doc_id]]:
    #         st.write(f"\t({topic_dict.get(num)}, {'%.5f' %dis})")
