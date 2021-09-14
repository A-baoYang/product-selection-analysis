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
parser.add_argument('--ec', type=str, default='Momo')
parser.add_argument('--dataset_type', type=str, default='productDetail')
args = parser.parse_args()
category = args.category
ec = args.ec
dataset_type = args.dataset_type

# ec_list = ['Momo','PChome','Shopee']
inputFile = f'Preprocessed-{ec}-query={category}-{dataset_type}.csv'
inputFilepath = os.path.join(ec, inputFile)
inputFilepathLower = inputFilepath.lower()
keywordLevelOutputFilepath = os.path.join(ec, f'KeywordLevelStats-{inputFile}')

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
def review_scoring(row):
    # 有抓到星級數可直接計算
    if type(row['review_score']) != float:
        score = row['review_score'].count('icon-rating-solid--active')
    # 沒抓到的部分用 fine-tuned model 預測
    else:
        score = int(sentiment_classifier(row['review_content'])[0]['label'][-1])
    return score


def review_labeling(x):
    if x <= 2:
        label = 'negative'
    elif x == 3:
        label = 'normal'
    else:
        label = 'positive'
    return label


def load_datasets(inputFilepath):
    """
    對匯入的資料做不同處理。
    :param inputFilepath: 要匯入的資料集位址
    :return: 處理後的 dataframe, metric columns list, brand column
    """
    df = pd.read_csv(inputFilepath).drop_duplicates()
    metric_col_list = ['review_score', 'review_num', 'origin_price', 'sales_price', 'discount', 'tfidf_score']
    brand_col = 'brand'
    if re.search('productDetail', inputFilepath):
        if re.search('momo|pchome', inputFilepath.lower()):
            df['review_score'] = 0
            df['review_num'] = 0
            df['discount'] = (df['origin_price'] - df['sales_price']) / df['origin_price']
        elif re.search('shopee', inputFilepath.lower()):
            df['origin_price'] = (df['origin_price_floor'] + df['origin_price_ceil']) / 2
            df['sales_price'] = (df['sales_price_floor'] + df['sales_price_ceil']) / 2
        elif re.search('jd', inputFilepath.lower()):
            df['discount'] = df['origin_price'] - df['sales_price']
            brand_col = 'seller'
        elif re.search('taobao', inputFilepath.lower()):
            df['origin_price'] = (df['price_floor'] + df['price_ceil']) / 2
            df['sales_price'] = df['origin_price']
            df['discount'] = df['origin_price'] - df['sales_price']
        else:
            print('This is a new EC, need human check.')
    elif re.search('productReviews', inputFilepath):
        if re.search('shopee', inputFilepath.lower()):
            df = df[df['review_content'].notnull()].reset_index(drop=True)
            df['review_score'] = df.apply(lambda row: review_scoring(row), axis=1)
            df['sentiment_label'] = df['review_score'].apply(lambda x: review_labeling(x))
        elif re.search('jd', inputFilepath.lower()):
            df = df[df['comment_content'].notnull()].reset_index(drop=True)
        else:
            print('This is a new EC, need to check.')
    elif re.search('searchResult', inputFilepath):
        if re.search('youtube', inputFilepath.lower()):
            pass
    else:
        print('This is a new dataset_type, need to check.')
    return df, metric_col_list, brand_col


def get_keyword_cols(ec):
    if ec.lower() == 'momo':
        kw_col_list = ['title', 'description', 'color', 'breadcrumb']
    elif ec.lower() == 'pchome':
        kw_col_list = ['title', 'description', 'specification', 'breadcrumb']
    elif ec.lower() == 'jd':
        kw_col_list = ['title', 'desc', 'style', 'category']
    elif ec.lower() == 'taobao':
        kw_col_list = ['title', 'specifications', 'subcategory', 'tags', 'suggest_keywords']
    elif ec.lower() == 'youtube':
        kw_col_list = ['title', 'description']
    else:
        kw_col_list = ['title', 'description']
    return kw_col_list


def ner_brand_extraction(x):
    """
        在字串萃取實體，如品牌、組織名、人名、數量單位等
        ---------------
        x: 要被萃取的字串
    """
    ckip_ws = ws_driver([x], use_delim=True)
    ckip_ner = ner_driver(ckip_ws[0], use_delim=True)
    ner_term = [term[0][0] for term in ckip_ner if len(term) > 0]
    return ner_term


def specific_level_stats(df, specific_col, metric_col_list):
    specific_dict = dict()
    for col in tqdm(metric_col_list):
        if re.search('num', col.lower()):
            specific_dict.update({col: df.groupby([specific_col])[col].aggregate(['sum', 'mean', 'std', 'min', 'max']).reset_index()})
        else:
            specific_dict.update({col: df.groupby([specific_col])[col].aggregate(['count', 'mean', 'std', 'min', 'max']).reset_index()})
    if specific_col == 'keywords':
        if 'brand' in df.columns:
            keyword_products_covered = df.groupby([specific_col])['product_id', 'brand'].aggregate(
                ['nunique']).reset_index()
            keyword_products_covered.columns = [specific_col, 'products_covered', 'brands_covered']
        else:
            keyword_products_covered = df.groupby([specific_col])['product_id'].aggregate(
                ['nunique']).reset_index()
            keyword_products_covered.columns = [specific_col, 'products_covered']
        specific_dict.update({'unique_covered': keyword_products_covered})
    return specific_dict


def get_tfidf_col(df, index_col, kw_col):
    """
    計算 tfidf score 後以 dataframe 格式輸出
    :param df:
    :param kw_col:
    :return:
    """
    tfidfVector = tfidf_vectorizer.fit_transform(df[kw_col].values.tolist())
    # print(word_pool_tfidfVector.shape)
    # print(vectorizer.get_feature_names())
    keyword_tfidf = pd.DataFrame(tfidfVector.toarray(), index=df.index.values.tolist(),
                                 columns=tfidf_vectorizer.get_feature_names()) \
                      .reset_index().rename(columns={'index': index_col})
    return keyword_tfidf


def get_tfidf_score(row, df, kw_col='keywords', prod_col='product_id'):
    try:
        return df[row[kw_col]][row[prod_col]]
    except:
        return 0.0


def ec_aspect_extractor(sentence):
    sent_split = [sent.strip() for sent in re.split('，|。|；|！|？|,|;|!|\?|\n', sentence) if sent.strip() != '']
    aspect_count = list()

    for sent in sent_split:
        ckip_ws = ws_driver([sent], use_delim=True)
        ckip_pos = pos_driver(ckip_ws, use_delim=False)
        score = int(sentiment_classifier(sent)[0]['label'][-1])

        for ws, pos in zip(ckip_ws[0], ckip_pos[0]):
            if pos == 'Na':
                aspect_count += [ws] * score
            # elif (pos == 'VA') or (pos == 'VH') or (pos == 'VG'):
            #     emo_list.append(ws)
            else:
                pass
    return dict(Counter(aspect_count))


def gen_avg_sentiment(df, aspect_col):
    aspect_sentiments = list()
    aspect_frequencies = list()
    for _dict in tqdm(df[aspect_col].values.tolist()):
        for k in _dict.keys():
            aspect_sentiments += [k] * _dict[k]
            aspect_frequencies += [k]
    aspect_sentiment_dict = dict(Counter(aspect_sentiments))
    aspect_frequencies_dict = dict(Counter(aspect_frequencies))
    aspect_avg_sentiment_dict = {k: aspect_sentiment_dict[k] / aspect_frequencies_dict[k] for k in
                                 aspect_frequencies_dict.keys() & aspect_sentiment_dict}
    aspect_avg_sentiment_df = pd.DataFrame([aspect_avg_sentiment_dict]).T.reset_index().rename(columns={'index': 'aspect',
                                                                                                        0: 'avg_sentiment'})
    aspect_frequencies_df = pd.DataFrame([aspect_frequencies_dict]).T.reset_index().rename(columns={'index': 'aspect',
                                                                                                    0: 'frequency'})
    aspect_avg_sentiment_df = pd.merge(aspect_avg_sentiment_df, aspect_frequencies_df, on='aspect', how='inner')
    return aspect_avg_sentiment_df


# Main
if __name__ == '__main__':
    print('Importing dataset...')
    df, metric_col_list, brand_col = load_datasets(inputFilepath=inputFilepath)
    print(df.shape)
    print(df.info())

    if dataset_type == 'productDetail':
        # 子品類關鍵詞維度
        # 判斷做為關鍵詞來源的欄位
        word_cut_method = 'ckipWs'
        kw_col_list = get_keyword_cols(ec=ec)

        # 生成斷詞結果欄位
        print('Do word cutting...')
        df = add_wordCut_col(df=df, method=word_cut_method, col_list=kw_col_list)
        # 集中關鍵詞到一個欄位(keywords)
        wordCut_col_list = [f'{col}_{word_cut_method}' for col in kw_col_list]
        if len([col for col in ['brand', 'seller'] if col in df.columns]) > 0:
            df['keywords'] = df[wordCut_col_list + [brand_col]].agg(' '.join, axis=1)
        else:
            df['keywords'] = df[wordCut_col_list].agg(' '.join, axis=1)
        df = df.drop(kw_col_list + wordCut_col_list, axis=1)
        # 計算商品關鍵詞 tfidf 分數
        print('Computing keyword TFIDF Scores...')
        df_tfidf = get_tfidf_col(df=df, index_col='product_id', kw_col='keywords')
        pick_col = [col for col in df_tfidf.columns if not re.search('[0-9]+', col)]
        df_tfidf = df_tfidf[pick_col]
        # stack by keywrods
        df = df.reset_index().rename(columns={'index': 'product_id'})
        df = pd.merge(df, df_tfidf, on='product_id', how='left').drop_duplicates()
        df = df.drop('keywords', axis=1).join(df['keywords']
                                              .str
                                              .split(' ', expand=True)
                                              .stack()
                                              .reset_index(drop=True, level=1)
                                              .rename('keywords'))
        df = df[df['keywords'].notnull()].reset_index(drop=True)
        df.to_csv(keywordLevelOutputFilepath, index=False)

        # 關鍵詞量化指標
        print('aggregating keyword metrics...')
        df['tfidf_score'] = df.apply(lambda row: get_tfidf_score(row, df=df, kw_col='keywords', prod_col='product_id'), axis=1)
        pick_col = ['product_id'] + [col for col in df if col not in df_tfidf.columns]
        keyword_metrics = df[pick_col]
        keyword_stats_dict = specific_level_stats(df=keyword_metrics, specific_col='keywords', metric_col_list=metric_col_list)
        for key, frame in keyword_stats_dict.items():
            frame.to_csv(f'{ec}/{ec}-keyword_{key}.csv', index=False)

        # 品牌維度
        if len([col for col in ['brand', 'seller'] if col in df.columns]) > 0:
            print('aggregating brand metrics...')
            brand_stats_dict = specific_level_stats(df=df, specific_col=brand_col, metric_col_list=metric_col_list)
            for key, frame in brand_stats_dict.items():
                frame.to_csv(f'{ec}/{ec}-brand_{key}.csv', index=False)
            # 以品牌為參照，aggregate keywords 欄位
            brand_keywords = df.groupby([brand_col]).agg({'keywords': lambda x: ' '.join(x)})
            # 計算品牌 tfidf 向量
            print('Computing brand TFIDF vector...')
            df_tfidf = get_tfidf_col(df=brand_keywords, index_col='brand_id', kw_col='keywords')

    # 商品評論部分
    elif dataset_type == 'productReviews':
        df['aspect_n_emo'] = df['review_content'].apply(lambda x: ec_aspect_extractor(x))
        df = df.reset_index(drop=True)
        aspect_avg_sentiment_df = gen_avg_sentiment(df=df, aspect_col='aspect_n_emo')
        aspect_avg_sentiment_df.to_csv(f'{ec}/{ec}-{dataset_type}-aspect_avg_sentiment.csv', index=False)

    elif dataset_type == 'searchResult':
        pass
    else:
        print('Undefined dataset_type, need check')
    print('All job done.')
