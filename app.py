import pandas as pd
import numpy as np
import gensim
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from gensim import similarities
import pickle 
import streamlit as st
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

#set up session state via st.session_state so that app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

#load data
with open('model/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)
corpus = gensim.corpora.MmCorpus('model/corpus.mm')
data = pd.read_csv('dataset/DatasetLegal.csv')
corpus_lda = lda_model[corpus]
with open('model/id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)
index = similarities.MatrixSimilarity(corpus_lda, num_features=len(id2word))


topic_dict = {
    0 : "Contract",
    1 : "Family",
    2 : "Labor",
    3 : "Children",
    4 : "Sentence",
    5 : "lawyer",
    6 : "Succession"
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

def find_similar_docs(index, new_doc_topics, data):
    sims = index[new_doc_topics]
    sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
    # st.write(f"Topic distribution for new document : {new_doc_topics}\n{new_doc}\n")
    i = 0
    for doc_id, similarity in sims_sorted[:5]:
        st.write(f"Document ID: {doc_id}, Similarity score: {similarity*100} %")
        answer, question = st.columns(2)
        i += 1
        with question:
            with st.expander(f"question {i}:"):
                st.write(data.question[doc_id])
        with answer:
            with st.expander(f"answer {i}:"):
                st.write(data.answer[doc_id])
        st.divider()
        # st.write("Topic distribution for similar document : ")
        # for num, dis in corpus_lda[doc_id]:
        #     st.write(f"\t({topic_dict.get(num)}, {'%.5f' %dis})")

def tagging(new_doc_topics):
    option = []
    for i, score in new_doc_topics:
        if(score>=0.2):
            option.append(topic_dict[i])
    st.multiselect("Recomment tag!",
                    option,
                    option)


############ start with streamlit ############

#set up session state via st.session_state so that app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ title and header ############
        
st.set_page_config(
    layout="centered", page_title="Tag Recommendation and Similarity Search"
)
st.caption("")
st.title("Tag Recommendation and Similarity Search 🤗")

############ sidebar ############
st.sidebar.write("")

############ TABBED NAVIGATION ############
TagTab, SearchTab = st.tabs(["Tagging", "Searching"])

with TagTab:
    
    with st.form(key="tag_form"):
        st.write("")
        st.markdown(
        """
        Tagging Recommendation\n
        Enter the document here:
        """
        )
        #input from user
        input_doc = st.text_area("", help="Paste the document here")
        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            new_doc = preprocess(input_doc)
            new_doc = bow(new_doc)
            new_doc_topics = lda_model.get_document_topics(new_doc)
            tagging(new_doc_topics)

with SearchTab:
    with st.form(key="search_form"):
        st.write("")
        st.markdown(
        """
        Similarity Search\n
        Enter the document here:
        """
        )
        #input from user
        input_doc = st.text_area("", help="Paste the document here")
        submit_button = st.form_submit_button(label="Search")

        if submit_button:
            new_doc = preprocess(input_doc)
            new_doc = bow(new_doc)
            new_doc_topics = lda_model.get_document_topics(new_doc)
            find_similar_docs(index, new_doc_topics, data)