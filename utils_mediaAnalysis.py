import argparse
from collections import Counter
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import time
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from utils_textMining import *
from wordcloud import WordCloud
font = 'font_ch.ttf'


# Variables
parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='健身')
parser.add_argument('--media', type=str, default='Youtube')
parser.add_argument('--dataset_type', type=str, default='videoDetail')
args = parser.parse_args()
category = args.category
media = args.media
dataset_type = args.dataset_type

# media_list = ['PTT','Mobile01','Dcard', 'Youtube', 'Instagram']
inputFile = f'Preprocessed-{media}-query={category}-{dataset_type}.csv'
inputFilepath = os.path.join(media, inputFile)
inputFilepathLower = inputFilepath.lower()
keywordLevelOutputFilepath = os.path.join(media, f'KeywordLevelStats-{inputFile}')

# Initializations
# TFIDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords_zhCN)#, max_df=1.0, min_df=0.2)
# RoBERTa-base chinese fine-tuned model
if dataset_type == 'productReviews':
    print('Loading RoBERTa pretrained model...')
    roberta_tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-jd-full-chinese')
    roberta_model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-full-chinese')
    sentiment_classifier = pipeline('sentiment-analysis', model=roberta_model, tokenizer=roberta_tokenizer)


# Function


# Main
if __name__ == '__main__':
    #
    avg_sentiment_df = yt_d[['title', 'description', 'comment_sentiment']].groupby(['title', 'description']) \
        .agg({'comment_sentiment': 'mean'}).reset_index()
    yt = pd.merge(yt, avg_sentiment_df, on='title', how='left')
    for col in ['title', 'description', 'comment_sentiment']:
        yt[col] = yt[col].fillna(' ')

