import streamlit as st
import joblib
import re
from rake_nltk import Rake
from streamlit_option_menu import option_menu
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from textblob import TextBlob
from email import parser
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import joblib,os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")


# load Vectorizer For Gender Prediction
news_vectorizer = open("models/final_news_cv_vectorizer.pkl","rb")
news_cv = joblib.load(news_vectorizer)


# Fxn
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

nlp = spacy.load('en_core_web_sm')

# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key





# Sidebar options
option = st.sidebar.selectbox('Navigation', 
["Home",
 "Summarize",
 "Analyse sentiment",
 "News Classifier"
])

st.set_option('deprecation.showfileUploaderEncoding', False)

#Home

if option == 'Home':
	st.write(
			"""
            # What AUDIT can do for you?
            ###This Software helps users gain insights from both structured and unstructured text data using NLP(Natural Language Processing)
			"""
		)

#Summarize

elif option == 'Summarize':
    st.header("Summarize")

    st.subheader("Enter a corpus that you want to Summarize")
    text_input = st.text_area("Enter a paragraph", height=150)
    stopwords = list(STOP_WORDS)
    doc = nlp(text_input)
    tokens = [token.text for token in doc]
    st.write(punctuation = punctuation + '\n')

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1


    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]

    sentence_tokens = [sent for sent in doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
               if sent not in sentence_scores.keys():
                   sentence_scores[sent] = word_frequencies[word.text.lower()]
               else:
                   sentence_scores[sent] += word_frequencies[word.text.lower()]

    from heapq import nlargest



    select_length = int(len(sentence_tokens)*0.2)

    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    final_summary = [word.text for word in summary]

    summary = ' '.join(final_summary)
    if st.button("Summarize"):
        st.write(summary)

#Sentiment analysis

elif option == "Analyse sentiment":
		with st.form(key='nlpForm'):
			raw_text = st.text_area("Enter Text Here")
			submit_button = st.form_submit_button(label='Analyze')

		# layout
		col1,col2 = st.columns(2)
		if submit_button:

			with col1:
				st.info("Results")
				sentiment = TextBlob(raw_text).sentiment
				st.write(sentiment)

				# Emoji
				if sentiment.polarity > 0:
					st.markdown("Sentiment:: Positive :smiley: ")
				elif sentiment.polarity < 0:
					st.markdown("Sentiment:: Negative :angry: ")
				else:
					st.markdown("Sentiment:: Neutral 😐 ")

				# Dataframe
				result_df = convert_to_df(sentiment)
				st.dataframe(result_df)

				# Visualization
				c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
				st.altair_chart(c,use_container_width=True)



			with col2:
				st.info("Token Sentiment")

				token_sentiments = analyze_token_sentiment(raw_text)
				st.write(token_sentiments)

#Classify
elif option == "News Classifier":

	"""News Classifier"""
	st.title("News Classifier")
	# st.subheader("ML App with Streamlit")
	html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Streamlit ML App </h1>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	activity = ['Prediction','NLP']
	choice = st.sidebar.selectbox("Select Activity",activity)


	if choice == 'Prediction':
		st.info("Prediction with ML")

		news_text = st.text_area("Enter News Here","Type Here")
		all_ml_models = ["LR","RFOREST","NB","DECISION_TREE"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'business': 0,'tech': 1,'sport': 2,'health': 3,'politics': 4,'entertainment': 5}
		if st.button("Classify"):
			st.text("Original Text::\n{}".format(news_text))
			vect_text = news_cv.transform([news_text]).toarray()
			if model_choice == 'LR':
				predictor = load_prediction_models("models/newsclassifier_Logit_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'RFOREST':
				predictor = load_prediction_models("models/newsclassifier_RFOREST_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'NB':
				predictor = load_prediction_models("models/newsclassifier_NB_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'DECISION_TREE':
				predictor = load_prediction_models("models/newsclassifier_CART_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)

			final_result = get_key(prediction,prediction_labels)
			st.success("News Categorized as:: {}".format(final_result))

	if choice == 'NLP':
		st.info("Natural Language Processing of Text")
		raw_text = st.text_area("Enter News Here","Type Here")
		nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text::\n{}".format(raw_text))

			docx = nlp(raw_text)
			if task_choice == 'Tokenization':
				result = [token.text for token in docx ]
			elif task_choice == 'Lemmatization':
				result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
			elif task_choice == 'NER':
				result = [(entity.text,entity.label_)for entity in docx.ents]
			elif task_choice == 'POS Tags':
				result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

			st.json(result)

		if st.button("Tabulize"):
			docx = nlp(raw_text)
			c_tokens = [token.text for token in docx ]
			c_lemma = [token.lemma_ for token in docx ]
			c_pos = [token.pos_ for token in docx ]

			new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(new_df)

	




