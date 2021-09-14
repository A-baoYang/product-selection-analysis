# library
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import jieba
import re
from opencc import OpenCC
from tqdm import tqdm


# variables
stopword_file = 'stopwords_zhCN.txt'
official_dict = 'official_dict.txt'
custom_dict = 'custom_dict.txt'

# Initializations
# CKIP transformers models
print('Loading CKIP transformers model...')
ws_driver = CkipWordSegmenter(level=3)
pos_driver = CkipPosTagger(level=3)
ner_driver = CkipNerChunker(level=3)

# function
def chinese_converter(x, target_lang):
    """
        繁簡體轉換
        ----------------
        x: 要被轉換的字串
        target_lang: 要轉換成什麼語體

    """
    # Opencc converters
    converter_s2tw = OpenCC('s2tw')
    converter_tw2s = OpenCC('tw2s')
    # 簡體轉繁體
    if target_lang == 'zh-TW':
        res = converter_s2tw.convert(x) if type(x) != float else x
    # 繁體轉簡體
    elif target_lang == 'zh-CN':
        res = converter_tw2s.convert(x) if type(x) != float else x
    else:
        print('please specify your target language as `target_lang`. We now provide "zh-TW", "zh-CN". ')
    return res


def add_wordCut_col(df, method, col_list):
    """
    對特定欄位內容做斷詞後將結果存在另一欄位
    :param df:
    :param method: {'ckip_ws', 'jieba'}
    :param col_list:
    :return:
    """
    for col in tqdm(col_list):
        if method == 'ckipWs':
            df[f'{col}_{method}'] = df[col].apply(
                lambda x: ' '.join(
                    [char.strip() for char in ws_driver([chinese_converter(x, target_lang='zh-TW')], use_delim=True)[0] if \
                     (len(char.strip()) > 0) and
                     (char.strip() not in stopwords_zhCN) and
                     (char.strip() not in stopwords_zhTW) and
                     (not re.search('[0-9]+', char.strip()))]))
        elif method == 'jiebaCut':
            jieba.load_userdict('official_dict.txt')
            df[f'{col}_{method}'] = df[col].apply(
                lambda x: ' '.join([chinese_converter(char.strip(), target_lang='zh-TW') for \
                                    char in jieba.cut(chinese_converter(str(x), target_lang='zh-CN'), cut_all=False) if \
                                    (len(char.strip()) > 0) and
                                    (char.strip() not in stopwords_zhCN) and
                                    (char.strip() not in stopwords_zhTW) and
                                    (not re.search('[0-9]+', char.strip()))]))
    return df


def remove_nonChinese(x):
    #     x = x.encode('utf-8') # convert context from str to unicode
    _filter = re.compile(u'[^\u4E00-\u9FA5]')  # non-Chinese unicode range
    x = _filter.sub(r' ', x)  # remove all non-Chinese characters
    #     x = x.encode('utf-8') # convert unicode back to str
    return x


def remove_emoji(x):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F000-\U0001FFFF"
    #         u"\U0001F600-\U0001F64F"  # emoticons
    #         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #         u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #         u"\U0001F900-\U0001F990"
    #         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    x = regrex_pattern.sub(r' ', x)
    return x


def remove_link(x):
    rule1 = r'https://[a-zA-z./\d]*'
    rule2 = r'http://[a-zA-z./\d]*'
    x = re.sub(rule1, ' ', x, flags=re.MULTILINE)
    x = re.sub(rule2, ' ', x, flags=re.MULTILINE)
    return x


def remove_tags(x):
    rule = '#.{0,30}#'
    #     tags = re.findall(rule, x)
    x = re.sub(rule, ' ', x)
    return x


# invalid
def remove_at_someone(x):
    rule = '@([^@]{0,30})）'
    x = re.sub(rule, ' ', x)
    return x


def remove_punctuation(x):
    rule1 = '[\s+\.\!\/_,$%^*(+\"\']+|[+——──！，。？、~@#￥%……&*（）｜]+'
    rule2 = '[【】╮╯▽╰╭★→「」]+'
    rule3 = '！，❤。～《》：（）【】「」？”“；：、'
    x = re.sub(rule1, ' ', x)
    x = re.sub(rule2, ' ', x)
    x = re.sub(rule3, ' ', x)
    return x


def remove_other_char(x):
    # spaces
    rule1 = '\s'
    # digits
    rule2 = '\d'
    # ellipsis
    rule3 = '\.*'
    for rule in [rule1, rule2, rule3]:
        x = re.sub(rule, ' ', x)
    return x


# Main
# Simplified Chinese stopwords
print('Loading Simplified Chinese stopwords...')
stopwords_zhCN = list()
with open(stopword_file, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        if not len(l.strip()):
            continue
        stopwords_zhCN.append(l.strip())


# Traditional Chinese stopwords
print('Loading Traditional Chinese stopwords...')
stopwords_zhTW = list()
with open(stopword_file, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        if not len(l.strip()):
            continue
        stopwords_zhTW.append(chinese_converter(l.strip(), target_lang='zh-TW'))


# # dictionary
# print('Loading custom dictionary...')
# with open(custom_dict, 'r', encoding='utf-8') as custom_f:
#     with open(official_dict, 'a', encoding='utf-8') as official_f:
#         for line in custom_f:
#             official_f.write(line)
