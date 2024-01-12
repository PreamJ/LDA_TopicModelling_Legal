import pandas as pd
import numpy as np
import gensim
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from gensim import similarities
import pickle
import streamlit as st

def preprocess(text):
    with open('model/id2word.pkl', 'rb') as f:
        id2word = pickle.load(f)
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
    vector = id2word.doc2bow(result)
    return vector

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        self.min_val = min(data)
        self.max_val = max(data)

    def transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        scaled_data = []
        for value in data:
            scaled_value = (value - self.min_val) / (self.max_val - self.min_val) * (
                    self.feature_range[1] - self.feature_range[0]
            ) + self.feature_range[0]
            scaled_data.append(scaled_value)

        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def find_similar_docs(index_lda, index_bow, new_doc_topics, data):
    '''
    def topic_keyword_based(new_doc):
        new_doc_topics = id2word.doc2bow(preprocess(new_doc))
        lda_topics = lda_model.get_document_topics(new_doc_topics)
        sims_lda = index_lda[lda_topics]
        sims_bow = index_bow[new_doc_topics]
        max_lda = max(sims_lda)
        max_bow = max(sims_bow)
        weight = max_lda/max_bow
        sum = sims_lda+(sims_bow*weight)
        sims_sorted = sorted(enumerate(sum), key=lambda item: -item[1])
        return sims_sorted
'''
    sims_lda = index_lda[new_doc_topics]
    sims_bow = index_bow[new_doc_topics]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_lda = scaler.fit_transform(sims_lda)
    scaled_bow = scaler.fit_transform(sims_bow)
    # sum_result = [x + y for x, y in zip(scaled_data, scaled_data)]
    sum = [x+y for x,y in zip(scaled_lda, scaled_bow)]
    # max_lda = max(sims_lda)
    # max_bow = max(sims_bow)
    # sum = sims_lda+(sims_bow*3)
    sims_sorted = sorted(enumerate(sum), key=lambda item: -item[1])
    # sims_sorted = sorted(enumerate(sims_lda), key=lambda item: -item[1])
    # sims_sorted = sorted(enumerate(sims_bow), key=lambda item: -item[1])
    # st.write(f"Topic distribution for new document : {new_doc_topics}\n{new_doc}\n")
    i = 0
    for doc_id, similarity in sims_sorted[1:10]:
        # st.write(f"Document ID: {doc_id}, Similarity score: {similarity*100} %")
        st.write(f"Document ID: {doc_id}")
        
        i += 1
        # question, answer = st.columns(2)
        with st.expander(f"Document {i}:"):
            # with question:
            st.write("\n**Question**")
            st.write(data.question[doc_id])
            st.divider()
            # with answer:
            st.write("\n**Answer**")
            st.write(data.answer[doc_id])
            # with question:
        #     with st.expander(f"question {i}:"):
                
        # with answer: 
        #     with st.expander(f"answer {i}:"):
        st.divider()

def tagging(new_doc_topics):
    with open('lda/topic_dict_5.pkl', 'rb') as f:
        topic_dict = pickle.load(f)
    option = []
    for i, score in new_doc_topics:
        if(score>=0.2):
            option.append(topic_dict[i])
    st.multiselect("Recommended tag!",
                    option,
                    option)

    # label_tags = st_tag(

    # )
