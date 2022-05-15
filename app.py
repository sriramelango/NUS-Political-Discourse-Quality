import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import plotly.graph_objects as go
import pickle
import re
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from nltk import tokenize


## Functions
def obtainDecisionTree(text):
    constructScore_DT = float(constructiveness_model_DT.predict(count_vect_CONSTRUCT.transform([text]))[0])
    justificationScore_DT = float(justification_model_DT.predict(count_vect_JUST.transform([text]))[0])
    relevanceScore_DT = float(relevance_model_DT.predict(count_vect_RELAV.transform([text]))[0])
    reciprocityScore_DT = float(reciprocity_model_DT.predict(count_vect_REC.transform([text]))[0])
    empathyScore_DT = float(empathy_model_DT.predict(count_vect_EMP.transform([text]))[0])
    uncivilScore_DT = float(uncivil_model_DT.predict(count_vect_CIV.transform([text]))[0])
    st.success("Constructiveness: " + str(constructScore_DT) + ", Justification: " + str(justificationScore_DT) + 
            ", Relevance: " + str(relevanceScore_DT) + ", Reciprocity: " + str(reciprocityScore_DT) + 
            ", Empathy Respect: " + str(empathyScore_DT) + ", Uncivil Abuse: " + str(uncivilScore_DT))
    return [constructScore_DT, justificationScore_DT, relevanceScore_DT, reciprocityScore_DT, empathyScore_DT, uncivilScore_DT]


def obtainGradientBoostingClassifier(text):
    constructScore_GDC = float(constructiveness_model_GDC.predict(count_vect_CONSTRUCT.transform([text]))[0])
    justificationScore_GDC = float(justification_model_GDC.predict(count_vect_JUST.transform([text]))[0])
    relevanceScore_GDC = float(relevance_model_GDC.predict(count_vect_RELAV.transform([text]))[0])
    reciprocityScore_GDC = float(reciprocity_model_GDC.predict(count_vect_REC.transform([text]))[0])
    empathyScore_GDC = float(empathy_model_GDC.predict(count_vect_EMP.transform([text]))[0])
    uncivilScore_GDC = float(uncivil_model_GDC.predict(count_vect_CIV.transform([text]))[0])
    st.success("Constructiveness: " + str(constructScore_GDC) + ", Justification: " + str(justificationScore_GDC) + 
            ", Relevance: " + str(relevanceScore_GDC) + ", Reciprocity: " + str(reciprocityScore_GDC) + 
            ", Empathy Respect: " + str(empathyScore_GDC) + ", Uncivil Abuse: " + str(uncivilScore_GDC)) 
    return [constructScore_GDC, justificationScore_GDC, relevanceScore_GDC, reciprocityScore_GDC, empathyScore_GDC, uncivilScore_GDC]


def obtainKNN(text):
    constructScore_KNN = float(constructiveness_model_KNN.predict(count_vect_CONSTRUCT.transform([text]))[0])
    justificationScore_KNN = float(justification_model_KNN.predict(count_vect_JUST.transform([text]))[0])
    relevanceScore_KNN = float(relevance_model_KNN.predict(count_vect_RELAV.transform([text]))[0])
    reciprocityScore_KNN = float(reciprocity_model_KNN.predict(count_vect_REC.transform([text]))[0])
    empathyScore_KNN = float(empathy_model_KNN.predict(count_vect_EMP.transform([text]))[0])
    uncivilScore_KNN = float(uncivil_model_KNN.predict(count_vect_CIV.transform([text]))[0])
    st.success("Constructiveness: " + str(constructScore_KNN) + ", Justification: " + str(justificationScore_KNN) + 
            ", Relevance: " + str(relevanceScore_KNN) + ", Reciprocity: " + str(reciprocityScore_KNN) + 
            ", Empathy Respect: " + str(empathyScore_KNN) + ", Uncivil Abuse: " + str(uncivilScore_KNN))
    return [constructScore_KNN, justificationScore_KNN, relevanceScore_KNN, reciprocityScore_KNN, empathyScore_KNN, uncivilScore_KNN]


def obtainLinearRegression(text):
    constructScore_LR = float(constructiveness_model_LR.predict(count_vect_CONSTRUCT.transform([text]))[0])
    justificationScore_LR = float(justification_model_LR.predict(count_vect_JUST.transform([text]))[0])
    relevanceScore_LR = float(relevance_model_LR.predict(count_vect_RELAV.transform([text]))[0])
    reciprocityScore_LR = float(reciprocity_model_LR.predict(count_vect_REC.transform([text]))[0])
    empathyScore_LR = float(empathy_model_LR.predict(count_vect_EMP.transform([text]))[0])
    uncivilScore_LR = float(uncivil_model_LR.predict(count_vect_CIV.transform([text]))[0])
    st.success("Constructiveness: " + str(constructScore_LR) + ", Justification: " + str(justificationScore_LR) + 
            ", Relevance: " + str(relevanceScore_LR) + ", Reciprocity: " + str(reciprocityScore_LR) + 
            ", Empathy Respect: " + str(empathyScore_LR) + ", Uncivil Abuse: " + str(uncivilScore_LR))
    return [constructScore_LR, justificationScore_LR, relevanceScore_LR, reciprocityScore_LR, empathyScore_LR, uncivilScore_LR]


def obtainRandomForest(text):
    constructScore_RF = float(constructiveness_model_RF.predict(count_vect_CONSTRUCT.transform([text]))[0])
    justificationScore_RF = float(justification_model_RF.predict(count_vect_JUST.transform([text]))[0])
    relevanceScore_RF = float(relevance_model_RF.predict(count_vect_RELAV.transform([text]))[0])
    reciprocityScore_RF = float(reciprocity_model_RF.predict(count_vect_REC.transform([text]))[0])
    empathyScore_RF = float(empathy_model_RF.predict(count_vect_EMP.transform([text]))[0])
    uncivilScore_RF = float(uncivil_model_RF.predict(count_vect_CIV.transform([text]))[0])
    st.success("Constructiveness: " + str(constructScore_RF) + ", Justification: " + str(justificationScore_RF) + 
            ", Relevance: " + str(relevanceScore_RF) + ", Reciprocity: " + str(reciprocityScore_RF) + 
            ", Empathy Respect: " + str(empathyScore_RF) + ", Uncivil Abuse: " + str(uncivilScore_RF))
    return [constructScore_RF, justificationScore_RF, relevanceScore_RF, reciprocityScore_RF, empathyScore_RF, uncivilScore_RF]


def obtainSDG(text):
    constructScore_SDG = float(constructiveness_model_SDG.predict(count_vect_CONSTRUCT.transform([text]))[0])
    justificationScore_SDG = float(justification_model_SDG.predict(count_vect_JUST.transform([text]))[0])
    relevanceScore_SDG = float(relevance_model_SDG.predict(count_vect_RELAV.transform([text]))[0])
    reciprocityScore_SDG = float(reciprocity_model_SDG.predict(count_vect_REC.transform([text]))[0])
    empathyScore_SDG = float(empathy_model_SDG.predict(count_vect_EMP.transform([text]))[0])
    uncivilScore_SDG = float(uncivil_model_SDG.predict(count_vect_CIV.transform([text]))[0])
    st.success("Constructiveness: " + str(constructScore_SDG) + ", Justification: " + str(justificationScore_SDG) + 
            ", Relevance: " + str(relevanceScore_SDG) + ", Reciprocity: " + str(reciprocityScore_SDG) + 
            ", Empathy Respect: " + str(empathyScore_SDG) + ", Uncivil Abuse: " + str(uncivilScore_SDG))
    return [constructScore_SDG, justificationScore_SDG, relevanceScore_SDG, reciprocityScore_SDG, empathyScore_SDG, uncivilScore_SDG]


def obtainLinearSVC(text):
    constructScore_SVC = float(constructiveness_model_SVC.predict(count_vect_CONSTRUCT.transform([text]))[0])
    justificationScore_SVC = float(justification_model_SVC.predict(count_vect_JUST.transform([text]))[0])
    relevanceScore_SVC = float(relevance_model_SVC.predict(count_vect_RELAV.transform([text]))[0])
    reciprocityScore_SVC = float(reciprocity_model_SVC.predict(count_vect_REC.transform([text]))[0])
    empathyScore_SVC = float(empathy_model_SVC.predict(count_vect_EMP.transform([text]))[0])
    uncivilScore_SVC = float(uncivil_model_SVC.predict(count_vect_CIV.transform([text]))[0])
    st.success("Constructiveness: " + str(constructScore_SVC) + ", Justification: " + str(justificationScore_SVC) + 
            ", Relevance: " + str(relevanceScore_SVC) + ", Reciprocity: " + str(reciprocityScore_SVC) + 
            ", Empathy Respect: " + str(empathyScore_SVC) + ", Uncivil Abuse: " + str(uncivilScore_SVC))
    return [constructScore_SVC, justificationScore_SVC, relevanceScore_SVC, reciprocityScore_SVC, empathyScore_SVC, uncivilScore_SVC]

# Functions for Analysis
def genWordCloud(text):
    stopwords = set(STOPWORDS)
    text = text.lower()
    wordCloud = WordCloud(background_color = "black", stopwords=stopwords, prefer_horizontal=1).generate(text) 
    plt.imshow(wordCloud, interpolation="bilinear") 
    plt.axis('off') 
    st.pyplot() 
        

def dataBarGraphProcess(data,xlabel,ylabel, unique):
    dataFiltered = []
    dataUnique = unique
    for i in range(len(dataUnique)):
        occurrences = data.count(dataUnique[i])
        dataFiltered.append([dataUnique[i], occurrences])
    dataFiltered = pd.DataFrame(dataFiltered, columns=[xlabel, ylabel])
    dataFiltered = dataFiltered.dropna()
    return dataFiltered
    

def plotSubjectivity(text):
    sentences = tokenize.sent_tokenize(text)
    data = []
    for i in range(len(sentences)):
        data.append(subClassification(sentences[i]))
    data = dataBarGraphProcess(data, "Perception", "Frequency", ["Objective", "Subjective"])
    fig = px.bar(data, x = "Perception", y= "Frequency", color = "Perception")
    st.plotly_chart(fig, use_container_width = True)


def subClassification(sentence):
    scoreSubjectivity = TextBlob(sentence).sentiment.subjectivity
    if scoreSubjectivity < 0.5:
        return "Objective"
    else:
        return "Subjective"


def plotSentiment(text):
    sentences = tokenize.sent_tokenize(text)
    data = []
    for i in range(len(sentences)):
        data.append(sentClassification(sentences[i]))
    data = dataBarGraphProcess(data, "Sentiment", "Frequency", ["Positive", "Negative", "Neutral"])
    fig = px.bar(data, x = "Sentiment", y= "Frequency", color = "Sentiment")
    st.plotly_chart(fig, use_container_width = True)


def sentClassification(sentence):
    sentimentScore = TextBlob(sentence).sentiment.polarity
    if sentimentScore < 0:
        return "Negative"
    elif sentimentScore == 0:
        return "Neutral"
    else:
        return "Positive"



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
count_vect_CONSTRUCT.fit_transform(constructiveness['message'])

# Model Building
constructiveness_model_DT = pickle.load(open('ML Models/Decision Tree/construct_model.pkl', 'rb'))
constructiveness_model_GDC = pickle.load(open('ML Models/Gradient Boosting Classifier/construct_model.pkl', 'rb'))
constructiveness_model_KNN = pickle.load(open('ML Models/KNN/construct_model.pkl', 'rb'))
constructiveness_model_LR = pickle.load(open('ML Models/Linear Regression/construct_model.pkl', 'rb'))
constructiveness_model_RF = pickle.load(open('ML Models/Random Forest/construct_model.pkl', 'rb'))
constructiveness_model_SDG = pickle.load(open('ML Models/SDG Classification/construct_model.pkl', 'rb'))
constructiveness_model_SVC = pickle.load(open('ML Models/SVC/construct_model.pkl', 'rb'))


##  Justification
count_vect_JUST = CountVectorizer()
count_vect_JUST.fit_transform(justification['message'])

# Model Building
justification_model_DT = pickle.load(open('ML Models/Decision Tree/justification_model.pkl', 'rb'))
justification_model_GDC = pickle.load(open('ML Models/Gradient Boosting Classifier/justification_model.pkl', 'rb'))
justification_model_KNN = pickle.load(open('ML Models/KNN/justification_model.pkl', 'rb'))
justification_model_LR = pickle.load(open('ML Models/Linear Regression/justification_model.pkl', 'rb'))
justification_model_RF = pickle.load(open('ML Models/Random Forest/justification_model.pkl', 'rb'))
justification_model_SDG = pickle.load(open('ML Models/SDG Classification/justification_model.pkl', 'rb'))
justification_model_SVC = pickle.load(open('ML Models/SVC/justification_model.pkl', 'rb'))


## Relevance
count_vect_RELAV = CountVectorizer()
count_vect_RELAV.fit_transform(relevance['message'])

# Model Building
relevance_model_DT = pickle.load(open('ML Models/Decision Tree/relevance_model.pkl', 'rb'))
relevance_model_GDC = pickle.load(open('ML Models/Gradient Boosting Classifier/relevance_model.pkl', 'rb'))
relevance_model_KNN = pickle.load(open('ML Models/KNN/relevance_model.pkl', 'rb'))
relevance_model_LR = pickle.load(open('ML Models/Linear Regression/relevance_model.pkl', 'rb'))
relevance_model_RF = pickle.load(open('ML Models/Random Forest/relevance_model.pkl', 'rb'))
relevance_model_SDG = pickle.load(open('ML Models/SDG Classification/relevance_model.pkl', 'rb'))
relevance_model_SVC = pickle.load(open('ML Models/SVC/relevance_model.pkl', 'rb'))


## Reciprocity
count_vect_REC = CountVectorizer()
count_vect_REC.fit_transform(reciprocity['message'])

# Model Building
reciprocity_model_DT = pickle.load(open('ML Models/Decision Tree/reciprocity_model.pkl', 'rb'))
reciprocity_model_GDC = pickle.load(open('ML Models/Gradient Boosting Classifier/reciprocity_model.pkl', 'rb'))
reciprocity_model_KNN = pickle.load(open('ML Models/KNN/reciprocity_model.pkl', 'rb'))
reciprocity_model_LR = pickle.load(open('ML Models/Linear Regression/reciprocity_model.pkl', 'rb'))
reciprocity_model_RF = pickle.load(open('ML Models/Random Forest/reciprocity_model.pkl', 'rb'))
reciprocity_model_SDG = pickle.load(open('ML Models/SDG Classification/reciprocity_model.pkl', 'rb'))
reciprocity_model_SVC = pickle.load(open('ML Models/SVC/reciprocity_model.pkl', 'rb'))


## Empathy
count_vect_EMP = CountVectorizer()
count_vect_EMP.fit_transform(empathy['message'])

# Model Building
empathy_model_DT = pickle.load(open('ML Models/Decision Tree/empathy_model.pkl', 'rb'))
empathy_model_GDC = pickle.load(open('ML Models/Gradient Boosting Classifier/empathy_model.pkl', 'rb'))
empathy_model_KNN = pickle.load(open('ML Models/KNN/empathy_model.pkl', 'rb'))
empathy_model_LR = pickle.load(open('ML Models/Linear Regression/empathy_model.pkl', 'rb'))
empathy_model_RF = pickle.load(open('ML Models/Random Forest/empathy_model.pkl', 'rb'))
empathy_model_SDG = pickle.load(open('ML Models/SDG Classification/empathy_model.pkl', 'rb'))
empathy_model_SVC = pickle.load(open('ML Models/SVC/empathy_model.pkl', 'rb'))


## Uncivil
count_vect_CIV = CountVectorizer()
count_vect_CIV.fit_transform(uncivil['message'])

# Model Building
uncivil_model_DT = pickle.load(open('ML Models/Decision Tree/uncivil_model.pkl', 'rb'))
uncivil_model_GDC = pickle.load(open('ML Models/Gradient Boosting Classifier/uncivil_model.pkl', 'rb'))
uncivil_model_KNN = pickle.load(open('ML Models/KNN/uncivil_model.pkl', 'rb'))
uncivil_model_LR = pickle.load(open('ML Models/Linear Regression/uncivil_model.pkl', 'rb'))
uncivil_model_RF = pickle.load(open('ML Models/Random Forest/uncivil_model.pkl', 'rb'))
uncivil_model_SDG = pickle.load(open('ML Models/SDG Classification/uncivil_model.pkl', 'rb'))
uncivil_model_SVC = pickle.load(open('ML Models/SVC/uncivil_model.pkl', 'rb'))



## Main Application
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
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

page = st.sidebar.selectbox("Select a page", ["Home", "Sentiment Analysis", "Training Data"])

if page == "Home":

    with st.form(key='form'):

        text = st.text_area("Type here")

        optionModel = st.selectbox("What model woud you like to utilize?",("Linear Regression","Random Forest","Decision Tree","KNN", "Gradient Boosting Classifier", "SDG Classifier", "Linear SVC"))

        submit = st.form_submit_button(label='Obtain Data')

        if text:

            if optionModel == "Linear Regression":

                scoresLR = obtainLinearRegression(text)

                fig = go.Figure(data=[go.Bar(
                    x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                    y= scoresLR
                    )])

                fig.update_layout(title_text='Score Predictions')
                st.plotly_chart(fig, use_container_width=True)

            elif optionModel == "Gradient Boosting Classifier":

                scoresGDC = obtainGradientBoostingClassifier(text)

                fig = go.Figure(data=[go.Bar(
                    x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                    y= scoresGDC
                    )])

                fig.update_layout(title_text='Score Predictions')
                st.plotly_chart(fig, use_container_width=True)

            elif optionModel == "Random Forest":

                scoresRF = obtainRandomForest(text)

                fig = go.Figure(data=[go.Bar(
                    x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                    y= scoresRF
                    )])

                fig.update_layout(title_text='Score Predictions')
                st.plotly_chart(fig, use_container_width=True)

            elif optionModel == "Decision Tree":

                scoresDT = obtainDecisionTree(text)

                fig = go.Figure(data=[go.Bar(
                    x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                    y= scoresDT
                    )])

                fig.update_layout(title_text='Score Predictions')
                st.plotly_chart(fig, use_container_width=True)

            elif optionModel == "KNN":

                scoresKNN = obtainKNN(text)

                fig = go.Figure(data=[go.Bar(
                    x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                    y= scoresKNN
                    )])

                fig.update_layout(title_text='Score Predictions')
                st.plotly_chart(fig, use_container_width=True)

            elif optionModel == "SDG Classifier":
                    
                    scoresSDG = obtainSDG(text)
        
                    fig = go.Figure(data=[go.Bar(
                        x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                        y= scoresSDG
                        )])
        
                    fig.update_layout(title_text='Score Predictions')
                    st.plotly_chart(fig, use_container_width=True)

            elif optionModel == "Linear SVC":

                scoresSVC = obtainLinearSVC(text)

                fig = go.Figure(data=[go.Bar(
                    x=['Constructivness', 'Justification', 'Relevance','Reciprocity', 'Empathy Respect', 'Uncivil Abuse'],
                    y= scoresSVC
                    )])

                fig.update_layout(title_text='Score Predictions')
                st.plotly_chart(fig, use_container_width=True)

elif page == "Sentiment Analysis":

    with st.form(key='form'):

        text = st.text_area("Type here")

        submit = st.form_submit_button(label='Obtain Data')

        if text:

            # Entity Viewer
            st.header("Word Cloud")
            genWordCloud(text)

            # Perception Viewer
            st.header("Perception Data")
            plotSubjectivity(text)

            # Sentiment Viewer
            st.header("Sentiment Analysis")
            plotSentiment(text)

elif page == "Training Data":

    st.dataframe(data, height = 800)