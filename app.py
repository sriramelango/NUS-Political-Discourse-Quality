import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff



## Logistic Model Train and Build
def simple_logistic_classify(X_tr, y_tr, c = 1.0):
  lg = LogisticRegression(C= c, max_iter = 10000)
  lg.fit(X_tr, y_tr)
  return lg



## Data Preprocessing
data = pd.read_csv('data.csv', encoding = 'cp1252');

constructiveness = data[['message', 'Constructiveness']]
constructiveness = constructiveness.dropna()

justification = data[['message', 'Justification']]
justification = justification.dropna()

relevance = data[['message', 'Relevance']]
relevance = relevance.dropna()

reciprocity = data[['message', 'Reciprocity']]
reciprocity = reciprocity.dropna()

empathy = data[['message', 'Empathy_Respect']]
empathy = empathy.dropna()

uncivil = data[['message', 'Uncivil_abuse']]
uncivil = uncivil.dropna()



## Constructiveness
count_vect_CONSTRUCT = CountVectorizer()
tfidf_transform_CONSTRUCT = TfidfTransformer(norm=None)

# X Train
X_train_counts_CONSTRUCT = count_vect_CONSTRUCT.fit_transform(constructiveness['message'])
X_train_tfidf_CONSTRUCT = tfidf_transform_CONSTRUCT.fit_transform(X_train_counts_CONSTRUCT)

# Y Train
y_train_CONSTRUCT = constructiveness['Constructiveness'].astype(float)

# Model Building
constructiveness_model = simple_logistic_classify(X_train_tfidf_CONSTRUCT, y_train_CONSTRUCT)


##  Justification
count_vect_JUST = CountVectorizer()
tfidf_transform_JUST = TfidfTransformer(norm=None)

# X Train
X_train_counts_JUST = count_vect_JUST.fit_transform(justification['message'])
X_train_tfidf_JUST = tfidf_transform_JUST.fit_transform(X_train_counts_JUST)

# Y Train
y_train_JUST = justification['Justification'].astype(float)

# Model Building
justification_model = simple_logistic_classify(X_train_tfidf_JUST, y_train_JUST)


## Relevance
count_vect_RELAV = CountVectorizer()
tfidf_transform_RELAV = TfidfTransformer(norm=None)

# X Train
X_train_counts_RELAV = count_vect_RELAV.fit_transform(relevance['message'])
X_train_tfidf_RELAV = tfidf_transform_RELAV.fit_transform(X_train_counts_RELAV)

# Y Train
y_train_RELAV = relevance['Relevance'].astype(float)

# Model Building
relevance_model = simple_logistic_classify(X_train_tfidf_RELAV, y_train_RELAV)


## Reciprocity
count_vect_REC = CountVectorizer()
tfidf_transform_REC = TfidfTransformer(norm=None)

# X Train
X_train_counts_REC = count_vect_REC.fit_transform(reciprocity['message'])
X_train_tfidf_REC = tfidf_transform_REC.fit_transform(X_train_counts_REC)

# Y Train
y_train_REC = reciprocity['Reciprocity'].astype(float)

# Model Building
reciprocity_model = simple_logistic_classify(X_train_tfidf_REC, y_train_REC)


## Empathy
count_vect_EMP = CountVectorizer()
tfidf_transform_EMP = TfidfTransformer(norm=None)

# X Train
X_train_counts_EMP = count_vect_EMP.fit_transform(empathy['message'])
X_train_tfidf_EMP = tfidf_transform_EMP.fit_transform(X_train_counts_EMP)

# Y Train
y_train_EMP = empathy['Empathy_Respect'].astype(float)

# Model Building
empathy_model = simple_logistic_classify(X_train_tfidf_EMP, y_train_EMP)


## Uncivil
count_vect_CIV = CountVectorizer()
tfidf_transform_CIV = TfidfTransformer(norm=None)

# X Train
X_train_counts_CIV = count_vect_CIV.fit_transform(uncivil['message'])
X_train_tfidf_CIV = tfidf_transform_CIV.fit_transform(X_train_counts_CIV)

# Y Train
y_train_CIV = uncivil['Uncivil_abuse'].astype(float)

# Model Building
uncivil_model = simple_logistic_classify(X_train_tfidf_CIV, y_train_CIV)



## Main Application
st.set_page_config(layout="wide")
st.header("Dashboard: The quality of online political talk: evaluating machine learning approaches for measuring discourse quality")
st.subheader("Abstract")
st.markdown("""
Social media data creates an influx of data, where traditional methods for examining public opinion and discourse quality can no longer reasonably scale for theoretical and thematic insights from millions of participants. This study examines whether text classifiers to measure the quality of political talk can generalize to new datasets. First, six classifiers were developed following an open-vocabulary approach based on an annotated mixed social media dataset. Next, through performance evaluations against four other hand-annotated datasets from previous work, the models show modest generalizability at measuring the quality of political talk in other social media platforms. Finally, the study concludes by summarizing the strengths and weaknesses of applying machine learning methods to social media posts and theoretical insights about the quality and structure of online political discourse.
""")
st.subheader('Links')
st.markdown("""
Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3870554

Github Repository: https://github.com/kj2013/twitter-deliberative-politics
""")

page = st.sidebar.selectbox("Select a page", ["Home", "Training Data"])

if page == "Home":

    with st.form(key='form'):

        text = st.text_area("Type here")

        submit = st.form_submit_button(label='Obtain Data')

        if text:

            constructScore = float(constructiveness_model.predict(count_vect_CONSTRUCT.transform([text]))[0])
            justificationScore = float(justification_model.predict(count_vect_JUST.transform([text]))[0])
            relevanceScore = float(relevance_model.predict(count_vect_RELAV.transform([text]))[0])
            reciprocityScore = float(reciprocity_model.predict(count_vect_REC.transform([text]))[0])
            empathyScore = float(empathy_model.predict(count_vect_EMP.transform([text]))[0])
            uncivilScore = float(uncivil_model.predict(count_vect_CIV.transform([text]))[0])

            st.success("Constructiveness: " + str(constructScore) + ", Justification: " + str(justificationScore) + 
            ", Relevance: " + str(relevanceScore) + ", Reciprocity: " + str(reciprocityScore) + 
            ", Empathy Respect: " + str(empathyScore) + ", Uncivil Abuse: " + str(uncivilScore))

            scores = [constructScore, justificationScore, relevanceScore, reciprocityScore, empathyScore, uncivilScore]

            fig = go.Figure(data=[go.Bar(
                x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                y= scores
                )])

            fig.update_layout(title_text='Score Predictions')
            st.plotly_chart(fig, use_container_width=True)

elif page == "Training Data":

    st.dataframe(data, height = 800)