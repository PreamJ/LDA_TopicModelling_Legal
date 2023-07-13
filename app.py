import pandas as pd
import numpy as np
import gensim
from gensim import similarities
import pickle
import streamlit as st
import time
import mymodule
from io import StringIO
st.set_option('deprecation.showPyplotGlobalUse', False)

#set up session state via st.session_state so that app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ load data ############
with open('lda/lda_model_5.pkl', 'rb') as f:
    lda_model = pickle.load(f)
corpus_question = gensim.corpora.MmCorpus('model/corpus_question.mm')
data = pd.read_csv('dataset/DatasetLegal.csv')
corpus_lda = lda_model[corpus_question]
with open('model/id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)

index = similarities.MatrixSimilarity(corpus_lda, num_features=len(id2word))

with open('lda/topic_dict_5.pkl', 'rb') as f:
    topic_dict = pickle.load(f)


e = RuntimeError('#')


# preprocessing new document data
# def preprocess(text):
#     stopwords = list(thai_stopwords())
#     read_stopwords = pd.read_csv('dataset/add_stopwords.csv')
#     add_stopwords = read_stopwords['stopword'].values.tolist()
#     result = []
#     str_text = str(text).replace(' ','')
#     word_token = word_tokenize(str_text, engine='newmm')
#     for word in word_token:
#         if(word not in stopwords + add_stopwords):
#             result.append(word)
#         #result = map(lambda x: re.sub('[,/.?# ]', '', x), result)
#     return result

# # convert text from new document to bag of word
# def bow(text):
#   vector = id2word.doc2bow(text)
#   return vector

# def find_similar_docs(index, new_doc_topics, data):
#     sims = index[new_doc_topics]
#     sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
#     # st.write(f"Topic distribution for new document : {new_doc_topics}\n{new_doc}\n")
#     i = 0
#     for doc_id, similarity in sims_sorted[:10]:
#         st.write(f"Document ID: {doc_id}, Similarity score: {similarity*100} %")
#         question, answer = st.columns(2)
#         i += 1
#         with question:
#             with st.expander(f"question {i}:"):
#                 st.write(data.question[doc_id])
#         with answer: 
#             with st.expander(f"answer {i}:"):
#                 st.write(data.answer[doc_id])
#         st.divider()
#         # st.write("Topic distribution for similar document : ")
#         # for num, dis in corpus_lda[doc_id]:
#         #     st.write(f"\t({topic_dict.get(num)}, {'%.5f' %dis})")

# def tagging(new_doc_topics):
#     option = []
#     # latest_iteration = st.empty()
#     # bar = st.progress(0)
#     for i, score in new_doc_topics:
#         # bar.progress(i)
#         time.sleep(0.1)
#         if(score>=0.2):
#             option.append(topic_dict[i])
#     st.multiselect("Recomment tag!",
#                     option,
#                     option)


############ start with streamlit ############

#set up session state via st.session_state so that app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ title and header ############
        
st.set_page_config(
    layout="centered", page_title="Tag Recommendation and Similarity Search"
)
st.caption("")
# st.title("Tag & Search :Legal")
# st.subheader("Tag Recommendation and Similarity Search 😎")
st.title("Tag Recommendation and Similarity Search 😎")
#about app
st.sidebar.write("")
st.sidebar.write("**More infomation about this app**\n")
st.sidebar.write("This application is designed to exclusively operate with data pertaining to laws only in Thai language.")
# st.sidebar.divider()
#report bugs
# st.sidebar.write("**report bug**")
# with st.sidebar.container():
#     if(e):
#         st.info('Bug Reporting')
    # st.sidebar.write("#")
st.sidebar.divider()
st.sidebar.caption("App created by [Pream J](https://github.com/PreamJ)")


########### TABBED NAVIGATION ############
TagTab, SearchTab = st.tabs(["Tagging", "Searching"])

with st.form(key="tag_form"):
    st.write("")
    st.markdown(
    """
    Tagging Recommendation\n
    Enter the document here:
    """
    )
    #input from user
    input_doc = st.text_area("", help="Paste you text doucument or upload you document file in the section below")
    # file = open("dataset\input_text.txt", "a", encoding="utf-8")
    # file.write(input_doc)
    # file.write("\n")
    # file.close()
    # uploaded_file = st.file_uploader("Choose a file")
    submit_button = st.form_submit_button(label="Submit")
    
    if submit_button:
        new_doc = mymodule.preprocess(input_doc)
        new_doc_topics = lda_model.get_document_topics(new_doc)
        # st.success('This is a success message!', icon="✅")
        mymodule.tagging(new_doc_topics)

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
        input_doc = st.text_area("", help="Paste you text doucument or upload you document file in the section below")
        # file = open("dataset\input_text.txt", "a", encoding="utf-8")
        # file.write(input_doc)
        # file.write("\n")
        # file.close()
        # uploaded_file = st.file_uploader("Choose a file")
        submit_button = st.form_submit_button(label="Search")

        if submit_button:
            new_doc = mymodule.preprocess(input_doc)

            new_doc_topics = lda_model.get_document_topics(new_doc)
            # st.success('This is a success message!', icon="✅")
            mymodule.find_similar_docs(index, new_doc_topics, data)

# with st.form(key="tag_form"):
#     st.write("")
#     st.markdown(
#     """
#     Tagging Recommendation & similarity search\n
#     Enter the document here:
#     """
#     )
    # #input from user
    # input_doc = st.text_area("", help="Paste you text doucument or upload you document file in the section below")
    # # file = open("dataset\input_text.txt", "a", encoding="utf-8")
    # # file.write(input_doc)
    # # file.write("\n")
    # # file.close()
    # # uploaded_file = st.file_uploader("Choose a file")
    # submit_button = st.form_submit_button(label="Submit")
    
    # if submit_button:
    #     new_doc = mymodule.preprocess(input_doc)
    #     new_doc_topics = lda_model.get_document_topics(new_doc)
    #     # st.success('This is a success message!', icon="✅")
    #     mymodule.tagging(new_doc_topics)
    #     mymodule.find_similar_docs(index, new_doc_topics, data)
