import pandas as pd
import numpy as np
import gensim
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from gensim import similarities
import pickle
import streamlit as st
import docx2txt

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def read_doc_file(file_path):
    text = docx2txt.process(file_path)
    return text

# def read_input_file()

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

def find_similar_docs(index, new_doc_topics, data):
    sims = index[new_doc_topics]
    sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
    # st.write(f"Topic distribution for new document : {new_doc_topics}\n{new_doc}\n")
    i = 0
    for doc_id, similarity in sims_sorted[:50]:
        st.write(f"Document ID: {doc_id}, Similarity score: {similarity*100} %")
        i += 1
        # question, answer = st.columns(2)
        with st.expander(f"Document {i}:"):
            # with question:
            st.write("**Question**")
            st.write(data.question[doc_id])
        # with answer:
            st.write("**Answer**")
            st.write(data.answer[doc_id])
        # with question:
        #     with st.expander(f"question {i}:"):
                
        # with answer: 
        #     with st.expander(f"answer {i}:"):
        st.divider()

def tagging(new_doc_topics):
    with open('model/topic_dict.pkl', 'rb') as f:
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