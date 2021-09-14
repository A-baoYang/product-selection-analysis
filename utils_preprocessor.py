import argparse
import datetime as dt
import numpy as np
import pandas as pd
import re
import os
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from utils_textMining import *


# Variable
parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='健身')
parser.add_argument('--platform', type=str, default='Momo')
parser.add_argument('--dataset_type', type=str, default='productDetail')
args = parser.parse_args()
category = args.category
platform = args.platform
dataset_type = args.dataset_type

inputFile = f'{platform}-query={category}-{dataset_type}.csv'
inputFilepath = os.path.join(platform, inputFile)
preprocessedOutputFilepath = os.path.join(platform, f'Preprocessed-{inputFile}')

# Initializations
# RoBERTa-base chinese fine-tuned model
if dataset_type == 'videoDetail':
    print('Loading RoBERTa pretrained model...')
    roberta_tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-jd-full-chinese')
    roberta_model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-full-chinese')
    sentiment_classifier = pipeline('sentiment-analysis', model=roberta_model, tokenizer=roberta_tokenizer)


# Function
def product_desc_cleansing(x):
    desc = list()
    for spec in x.split(';'):
        if (len(spec.split('：')) == 2) and (
                spec.split('：')[0] not in ['品牌', '商品名称', '商品编号', '商品毛重', '商品产地', '货号', '店铺', '国产/进口', '重量']):
            for item in re.split('/|／|（|）|，', spec.split('：')[1]):
                desc.append(item)
    return ' '.join(desc)


def timestamp_converter(x):
    x_formatted = x.replace('Streamed ', '')
    time_amount = int(x_formatted.split(' ')[0])
    time_unit = x_formatted.split(' ')[1]
    if 'year' in time_unit:
        hrs = 24 * 365
    elif 'month' in time_unit:
        hrs = 24 * 30
    elif 'week' in time_unit:
        hrs = 24 * 7
    elif 'day' in time_unit:
        hrs = 24
    else:
        hrs = 1
    return dt.datetime.strftime(dt.datetime.now() - dt.timedelta(hours=time_amount * hrs), '%Y-%m-%d')


def views_converter(x):
    view_amount = x.replace('views', '')
    if 'M' in view_amount:
        num_part = float(view_amount.replace('M', ''))
        view_amount = num_part * 1000000
    elif 'K' in view_amount:
        num_part = float(view_amount.replace('K', ''))
        view_amount = num_part * 1000
    elif 'No' in view_amount:
        view_amount = 0
    else:
        view_amount = float(view_amount)
    return int(view_amount)


def amount_converter(x):
    try:
        # Subscribers
        if 'subscriber' in x:
            x = x.replace('subscribers', '').replace('subscriber', '').strip()
            if 'M' in x:
                power = 1000000
            elif 'K' in x:
                power = 1000
            else:
                power = 1
            output = float(x.replace('M', '').replace('K', '')) * power
        # Description
        elif 'views' in x:
            output = x.strip().split('•')[2].strip()
        # Comments
        elif 'Comment' in x:
            output = float(x.replace('Comments', '').replace('Comment', '').replace(',', '').strip())
        # Comment_date
        elif 'ago' in x:
            output = timestamp_converter(x)
        else:
            if 'K' in x:
                output = float(x.strip().replace('K', '')) * 1000
            else:
                output = x.strip()
    except:
        output = x
    return output


def comment_sentiment(x):
    try:
        res = int(sentiment_classifier(chinese_converter(x, target_lang='zh-CN'))[0]['label'][-1])
    # 遇到長度超過  510 的句子會 RuntimeError
    except:
        res = 'Error'
    return res


def amazon_rules(df):
    df.columns = [col.lower() for col in df.columns]
    amazon_prod = df[df['title'] != 'undefined']
    amazon_prod['inventory'] = amazon_prod['inventory'].fillna('In Stock.')
    amazon_prod = amazon_prod.astype(str)
    amazon_prod = amazon_prod.assign(
        brand=amazon_prod.brand.apply(
            lambda x: x.replace('Visit the', '').replace('Store', '').replace('Brand:', '').strip()),
        subcategory=amazon_prod.best_sellers_rank.apply(
            lambda x: np.unique([char.strip() for char in re.split('\n|\(|\)| in |\xa0', x.strip()) if
                                 (char != '') and (not any(c in char for c in ['#', ';', 'Top ']))]).tolist()),
        bullet_points=amazon_prod.bullet_points.apply(lambda x: ';'.join([string for string
                                                                          in re.split(
                '\xa0|\n|This fits your| Make sure this fits|by entering your model number.|About this item', x)
                                                                          if not any(s == string for s in ['', '.'])])),
        star_rating=amazon_prod.star_rating.apply(lambda x: x.replace(' out of 5 stars', '')),
        price_floor=amazon_prod.price.apply(
            lambda x: x.split('-')[0].strip().replace('$', '') if len(x.split('-')) == 2 else x.replace('$', '')),
        price_ceil=amazon_prod.price.apply(
            lambda x: x.split('-')[1].strip().replace('$', '') if len(x.split('-')) == 2 else x.replace('$', '')),
        number_of_reviews=amazon_prod.number_of_reviews.apply(
            lambda x: x.replace(' global rating', '').replace('s', '').replace(',', '')),
        number_of_answered_questions=amazon_prod.number_of_answered_questions.apply(
            lambda x: x.replace(' answered questions', '')),
        inventory=amazon_prod.inventory.apply(lambda x: re.split('\n| - ', x)[0]),
        seller=amazon_prod.seller.apply(lambda x: x[:int(len(x) / 2)] if x != 'nan' else x)
    )
    for col in ['price_floor', 'price_ceil', 'star_rating', 'number_of_reviews', 'number_of_answered_questions']:
        amazon_prod[col] = pd.to_numeric(amazon_prod[col], errors='coerce')
        if col in ['star_rating', 'number_of_reviews', 'number_of_answered_questions']:
            amazon_prod[col] = amazon_prod[col].fillna(0.0)

    amazon_prod['inventory'] = amazon_prod['inventory'].apply(
        lambda x: 'In Stock' if ('out' not in x) and ('soon' not in x) and ('tock' in x) else 'Currently Unavailable')
    amazon_prod = amazon_prod[
        ['title', 'brand', 'subcategory', 'bullet_points', 'price_floor', 'price_ceil', 'star_rating',
         'number_of_reviews', 'number_of_answered_questions', 'inventory', 'seller']]
    amazon_prod.columns = ['title', 'brand', 'subcategory', 'description', 'price_floor', 'price_ceil', 'review_score',
                           'review_num', 'answered_question_num', 'inventory_status', 'seller']
    return amazon_prod


def jd_rules(df):

    df.columns = ['title', 'platform', 'url', 'seller', 'sales_price', 'spec', 'desc', 'category', 'style', 'ps',
                       'current_time',
                       'origin_price', 'review_num', 'review_score']
    jd_prod = df[df['title'].notnull()]
    for col in ['style', 'ps']:
        jd_prod[col] = jd_prod[col].fillna(' ')

    jd_prod['origin_price'] = jd_prod['origin_price'].fillna(jd_prod['sales_price'])
    jd_prod = jd_prod.assign(
        brand=jd_prod.desc.apply(lambda x: x.split(';')[0].replace('品牌： ', '')),
        desc=jd_prod.desc.apply(lambda x: product_desc_cleansing(x)),
        category=jd_prod.category.apply(lambda x: ' '.join(x.split('>')[:-2])),
        style=jd_prod['style'].apply(
            lambda x: ' '.join([opt for opt in x.split(';') if (opt != '') and (opt != 'nan')])),
        review_score=jd_prod.review_score.apply(lambda x: x.replace('%', ''))
    )
    for col in ['style', 'ps']:
        jd_prod[col] = jd_prod[col].fillna(' ')

    for col in ['origin_price', 'sales_price', 'review_num', 'review_score']:
        jd_prod[col] = pd.to_numeric(jd_prod[col], errors='coerce')

    jd_prod['origin_price'] = jd_prod['origin_price'].fillna(jd_prod['sales_price'])
    jd_prod['style'] = jd_prod['style'].apply(lambda x: '' if len(x) == 0 else x)
    jd_prod = jd_prod[
        ['title', 'category', 'desc', 'origin_price', 'sales_price', 'seller', 'review_num', 'review_score', 'style']]
    return jd_prod


def taobao_rules(df):

    for col in ['tags', 'specifications', 'suggest_keywords']:
        df[col] = df[col].fillna('None')
    taobao_prod = df.assign(
        review_num=df.review_num.apply(lambda x: x.replace('評價', '').replace('+', '').replace('萬', '0000')),
        sales_num=df.sales_num.apply(
            lambda x: x.replace('已售', '').replace('件', '').replace('+', '').replace('萬', '0000')),
        price_floor=df.price.apply(
            lambda x: x.split('-')[0].strip().replace('¥', '') if len(x.split('-')) == 2 else x.replace('¥', '')),
        price_ceil=df.price.apply(
            lambda x: x.split('-')[1].strip().replace('¥', '') if len(x.split('-')) == 2 else x.replace('¥', '')),
        subcategory=df.breadcrumb.apply(
            lambda x: ' '.join([char for char in re.split('\n| > ', x) if char.strip() != ''][1:])),
        tags=df.tags.apply(
            lambda x: ' '.join([char.strip()[1:] for char in re.split('\n|\(|\)|[0-9]+', x) if char.strip() != ''])),
        specifications=df.specifications.apply(
            lambda x: {item.split(': ')[0]: item.split(': ')[1] if (len(item.split(': ')) == 2) else 'None'
                       for item in [char.strip() for char in x.split('\n') if char.strip() != '']}),
        suggest_keywords=df.suggest_keywords.apply(
            lambda x: ' '.join([char.strip() for char in x.split('\n') if char.strip() != '']))
    )
    for col in ['review_num', 'sales_num', 'price_floor', 'price_ceil']:
        taobao_prod[col] = pd.to_numeric(taobao_prod[col], errors='coerce')

    taobao_prod['brand'] = taobao_prod['specifications'].apply(lambda x: x['品牌'] if '品牌' in x.keys() else 'None')
    for col in ['subcategory', 'suggest_keywords']:
        taobao_prod[col] = taobao_prod[col].apply(lambda x: 'None' if len(x) == 0 else x)

    taobao_prod = taobao_prod[['title', 'subcategory', 'brand', 'review_num', 'sales_num', 'price_floor', 'price_ceil',
                               'specifications', 'tags', 'suggest_keywords']]
    return taobao_prod


def momo_rules(df):

    for col in ['color', 'breadcrumb']:
        df[col] = df[col].fillna('None')

    momo_prod = df.assign(
        description=df.description.apply(lambda x: ' '.join(
            [string.strip() for string in x.split('\n') if (string.strip() != '') and ('品號' not in string)])),
        origin_price=df.origin_price.apply(
            lambda x: x.replace('\n', '').replace('建議售價', '').replace('促銷價', '').replace('元', '').replace(',',
                                                                                                          '').replace(
                '賣貴通報', '').strip()),
        sales_price=df.sales_price.apply(
            lambda x: x.replace(' ', '').replace('\n', '').replace('折扣後價格', '').replace('建議售價', '').replace('促銷價',
                                                                                                            '').replace(
                '元', '').replace('賣貴通報', '').replace('下單再折', '').replace(',', '').strip()),
        brand=df.brand.apply(lambda x: x.split('\n:\n')[1] if '品牌' in x else 'None'),
        color=df.color.apply(lambda x: x[3:-5] if x != 'None' else x),
        breadcrumb=df.breadcrumb.apply(lambda x: ' '.join(
            [char.strip() for char in x.split('\n') if char.strip() != ''][1:]) if x != 'None' else x)
    )
    for col in ['description']:
        momo_prod[col] = momo_prod[col].apply(lambda x: 'None' if len(x) == 0 else x)
    for col in ['origin_price', 'sales_price']:
        momo_prod[col] = pd.to_numeric(momo_prod[col], errors='coerce')
    return momo_prod


def pchome_rules(df):

    pchome_prod = df[df['title'].notnull()]
    pchome_prod['origin_price'] = pchome_prod['origin_price'].fillna(pchome_prod['sales_price'])
    for col in ['description', 'specification']:
        pchome_prod[col] = pchome_prod[col].fillna('None')

    pchome_prod = pchome_prod.assign(
        title=pchome_prod.title.apply(lambda x: remove_emoji(x)),
        description=pchome_prod.description.apply(lambda x: remove_emoji(x).replace('\n', ' ')),
        specification=pchome_prod.specification.apply(lambda x: remove_emoji(x).replace('\n', ' ')),
        origin_price=pchome_prod.origin_price.apply(
            lambda x: x.replace('\n', '').replace('網路價', '').replace('建議售價', '').replace('$', '').strip()),
        sales_price=pchome_prod.sales_price.apply(
            lambda x: x.replace('\n', '').replace('網路價', '').replace('$', '').strip()),
        breadcrumb=pchome_prod.breadcrumb.apply(lambda x: ' '.join(
            [char.strip() for char in remove_emoji(x).split('\n') if char.strip() != ''][3:]))
    )
    for col in ['origin_price', 'sales_price']:
        pchome_prod[col] = pd.to_numeric(pchome_prod[col])
    return pchome_prod


def shopee_rules(df):
    df[['origin_price', 'sales_price', 'discount']] = df.price.str.split('\n', expand=True)
    df['sales_price'] = df['sales_price'].fillna(df['origin_price'])
    for col in ['origin_price', 'sales_price']:
        df[col] = df[col].apply(lambda x: x.replace('$', '').replace(',', ''))

    for col in ['spec', 'avg_rating', 'review_num', 'discount']:
        df[col] = df[col].fillna('None')

    df = df.rename(columns={'avg_rating': 'review_score'})
    shopee_prod = df.assign(
        title=df.title.apply(lambda x: remove_emoji(x)),
        description=df.product_desc.apply(lambda x: remove_emoji(x).replace('\n', ' ').strip()),
        spec=df.spec.apply(lambda x: ' '.join(x.split('\n')) if x != 'None' else x),
        origin_price_floor=df.origin_price.apply(
            lambda x: x.split('-')[0].strip() if (len(x.split('-')) == 2) else x.strip()),
        origin_price_ceil=df.origin_price.apply(
            lambda x: x.split('-')[1].strip() if (len(x.split('-')) == 2) else x.strip()),
        sales_price_floor=df.sales_price.apply(
            lambda x: x.split('-')[0].strip() if (len(x.split('-')) == 2) else x.strip()),
        sales_price_ceil=df.sales_price.apply(
            lambda x: x.split('-')[1].strip() if (len(x.split('-')) == 2) else x.strip()),
        discount=df.discount.apply(lambda x: float(x[:-1].strip()) / 10 if x != 'None' else 0.0),
        review_num=df.review_num.apply(lambda x: x.replace(',', '')),
        sold_num=df.sold_num.apply(lambda x: x.replace(',', '')),
        stock_num=df.in_stock.apply(lambda x: x.replace('還剩', '').replace('件', ''))
    )
    for col in ['spec']:
        shopee_prod[col] = shopee_prod[col].apply(lambda x: 'None' if len(x) == 0 else x)

    for col in ['origin_price_floor', 'origin_price_ceil', 'sales_price_floor', 'sales_price_ceil',
                'review_score', 'review_num', 'sold_num', 'stock_num']:
        shopee_prod[col] = pd.to_numeric(shopee_prod[col], errors='coerce')

    shopee_prod[['review_score', 'review_num', 'sold_num']] = shopee_prod[
        ['review_score', 'review_num', 'sold_num']].fillna(0)
    shopee_prod = shopee_prod[['title', 'description', 'spec', 'origin_price_floor', 'origin_price_ceil',
                               'sales_price_floor', 'sales_price_ceil', 'discount',
                               'review_score', 'review_num', 'sold_num', 'stock_num']]
    return shopee_prod


def youtube_rules(df, dataset_type):
    if dataset_type == 'searchResult':
        df = df.drop(['Video_Link', 'Channel_Link', 'Description'], axis=1)
        df.columns = ['title', 'channel', 'views', 'publish_date']
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if type(x) != float else x)
        df['title'] = df['title'].apply(lambda x: remove_nonChinese(x))
        df['publish_date'] = df['publish_date'].apply(lambda x: timestamp_converter(x))
        df['views'] = df['views'].apply(lambda x: views_converter(x))

    elif dataset_type == 'videoDetail':
        df = df.drop(['View', 'Date', 'thumbs_up', 'thumbs_down'], axis=1)
        df.columns = ['title', 'channel', 'subscribers', 'description', 'comment_num', 'commenter_name',
                        'comment_date', 'comment', 'comment_thumb_ups']
        for col in ['subscribers', 'comment_num', 'comment_thumb_ups']:
            df[col] = df[col].fillna(0.0)
        for col in ['commenter_name', 'comment_date', 'comment']:
            df[col] = df[col].fillna('')
        for col in df.columns:
            df[col] = df[col].apply(lambda x: amount_converter(x))
        df['comment_sentiment'] = df['comment'].apply(lambda x: comment_sentiment(x))
        df = df[df['comment_sentiment'] != 'Error'].reset_index(drop=True)
        for col in ['subscribers', 'comment_num', 'comment_thumb_ups', 'comment_sentiment']:
            try:
                df[col] = df[col].astype(float)
            except:
                print(col)
        for col in ['title', 'description', 'comment']:
            df[col] = df[col].apply(lambda x: remove_nonChinese(x))
    else:
        print('Undefined dataset_type, need to check')
    return df


# Main
# Amazon
cleansed = 1
df = pd.read_csv(inputFilepath)
if 'amazon' in inputFilepath.lower():
    df = amazon_rules(df=df)
elif 'jd' in inputFilepath.lower():
    df = jd_rules(df=df)
elif 'taobao' in inputFilepath.lower():
    df = taobao_rules(df=df)
elif 'momo' in inputFilepath.lower():
    df = momo_rules(df=df)
elif 'pchome' in inputFilepath.lower():
    df = pchome_rules(df=df)
elif 'shopee' in inputFilepath.lower():
    df = shopee_rules(df=df)
elif 'youtube' in inputFilepath.lower():
    dataset_type = 'videoDetail'
    inputFile = f'{platform}-query={category}-{dataset_type}.csv'
    inputFilepath = os.path.join(platform, inputFile)
    df_videoDetail = pd.read_csv(inputFilepath)
    df_videoDetail = youtube_rules(df=df_videoDetail, dataset_type=dataset_type)

    dataset_type = 'searchResult'
    inputFile = f'{platform}-query={category}-{dataset_type}.csv'
    inputFilepath = os.path.join(platform, inputFile)
    df_searchResult = pd.read_csv(inputFilepath)
    df_searchResult = youtube_rules(df=df_searchResult, dataset_type=dataset_type)

    df_videoDetail = df_videoDetail.drop(['channel', 'commenter_name', 'comment_date', 'comment_thumb_ups'], axis=1)
    avg_sentiment = df_videoDetail.groupby(['title', 'description']).agg({'subscribers': 'mean', 'comment_num': 'mean',
                                                                'comment': ' '.join,
                                                                'comment_sentiment': 'mean'}).reset_index()
    df = pd.merge(df_searchResult, avg_sentiment, on='title', how='left')
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna(' ')
        else:
            df[col] = df[col].fillna(.0)
    preprocessedOutputFilepath = os.path.join(platform, f'Preprocessed-{inputFile}')  # use videoDetail
else:
    print(f'Currently not yet built cleansing rule function for {inputFile}')
    cleansed = 0

if cleansed == 1:
    df.to_csv(preprocessedOutputFilepath, index=False)
    print(df.head(1))
    print(df.info())

