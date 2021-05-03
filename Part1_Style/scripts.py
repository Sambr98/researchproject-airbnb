import pandas as pd
import statistics as stats
import nltk
import textstat
import re
import collections
import math
from spellchecker import SpellChecker
import string

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def preprocess_text(text, remove_punctuation=False):
    doc = text.lower()
    for form in contractions:
        doc = doc.replace(form,contractions[form])
    doc = re.sub(r'/|-', ' ', doc)
    if (remove_punctuation):
        doc = re.sub(r'(?u)[^\w\s]', '', doc)
    else:
        doc = re.sub(r'(?u)[^\w\s!,.:;?()]', '', doc)
    return doc

def get_basic_metrics(input_df):
    df = input_df.copy()
    
    values_char_len = []
    values_sent_count = []
    values_word_count = []
    size = df.shape[0]
    for i in range(size):
        print("{}/{}".format(i, size), end='\r')
        current_char_len = []
        current_sent_count = []
        current_word_count = []
        descriptions_ = df['full_description'][i]
        for text in descriptions_:
            doc = preprocess_text(text, remove_punctuation=True)
            tokens = nltk.word_tokenize(doc)
            current_char_len.append(len(text))
            current_sent_count.append(textstat.sentence_count(text))
            current_word_count.append(len(tokens))
        values_char_len.append(stats.median(current_char_len))
        values_sent_count.append(stats.median(current_sent_count))
        values_word_count.append(stats.median(current_word_count))
    df['char_len'] = values_char_len
    df['sent_count'] = values_sent_count
    df['word_count'] = values_word_count

    return df

def get_all_readability_metrics(input_df):
    df = input_df.copy()

    medians_CLI = []
    medians_ARI = []
    medians_GFI = []
    medians_SMOG = []
    medians_DCRI = []
    medians_FKRI = []
    size = df.shape[0]
    for i in range(size):
        print("{}/{}".format(i, size), end='\r')
        values_CLI = []
        values_ARI = []
        values_GFI = []
        values_SMOG = []
        values_DCRI = []
        values_FKRI = []
        descriptions_ = df['full_description'][i]
        for text in descriptions_:
            values_CLI.append(textstat.coleman_liau_index(text))
            values_ARI.append(textstat.automated_readability_index(text))
            values_GFI.append(textstat.gunning_fog(text))
            values_SMOG.append(textstat.smog_index(text))
            values_DCRI.append(textstat.dale_chall_readability_score(text))
            values_FKRI.append(textstat.flesch_reading_ease(text))
        medians_CLI.append(stats.median(values_CLI))
        medians_ARI.append(stats.median(values_ARI))
        medians_GFI.append(stats.median(values_GFI))
        medians_SMOG.append(stats.median(values_SMOG))
        medians_DCRI.append(stats.median(values_DCRI))
        medians_FKRI.append(stats.median(values_FKRI))
    df['CLI'] = medians_CLI
    df['ARI'] = medians_ARI
    df['GFI'] = medians_GFI
    df['SMOG'] = medians_SMOG
    df['DCRI'] = medians_DCRI
    df['FKRI'] = medians_FKRI

    return df

def get_pos_metrics(input_df):
    df = input_df.copy()
    
    values_CC = []
    values_DT = []
    values_IN = []
    values_JJ = []
    values_VB = []
    values_NN = []
    values_RB = []
    values_EX = []
    values_PO = []
    values_CD = []
    size = df.shape[0]
    for i in range(size):
        print("{}/{}".format(i, size), end='\r')
        descriptions_ = df['full_description'][i]
        current_CC = []
        current_DT = []
        current_IN = []
        current_JJ = []
        current_VB = []
        current_NN = []
        current_RB = []
        current_EX = []
        current_PO = []
        current_CD = []
        for desc in descriptions_:
            doc = preprocess_text(desc)
            text = nltk.pos_tag(nltk.word_tokenize(doc))
            count_CC = 0
            count_DT = 0
            count_IN = 0
            count_JJ = 0
            count_VB = 0
            count_NN = 0
            count_RB = 0
            count_EX = 0
            count_PO = 0
            count_CD = 0
            count_ignore = 0
            for j in text:
                if ("CC" in j[1]):
                    count_CC += 1
                elif ("DT" in j[1]):
                    count_DT += 1
                elif ("IN" in j[1]):
                    count_IN += 1
                elif ("JJ" in j[1]):
                    count_JJ += 1
                elif ("VB" in j[1]):
                    count_VB += 1
                elif ("NN" in j[1]):
                    count_NN += 1
                elif ("RB" in j[1]):
                    count_RB += 1
                elif ("EX" in j[1]):
                    count_EX += 1
                elif ("POS" in j[1] or "PRP$" in j[1] or "WP$" in j[1]):
                    count_PO += 1
                elif ("CD" in j[1]):
                    count_CD += 1
                elif (j[1] == '.' or j[1] == ',' or j[1] == '(' or j[1] == ')' or j[1] == ':'):
                    count_ignore += 1
            current_CC.append(count_CC/(len(text)-count_ignore))
            current_DT.append(count_DT/(len(text)-count_ignore))
            current_IN.append(count_IN/(len(text)-count_ignore))
            current_JJ.append(count_JJ/(len(text)-count_ignore))
            current_VB.append(count_VB/(len(text)-count_ignore))
            current_NN.append(count_NN/(len(text)-count_ignore))
            current_RB.append(count_RB/(len(text)-count_ignore))
            current_EX.append(count_EX/(len(text)-count_ignore))
            current_PO.append(count_PO/(len(text)-count_ignore))
            current_CD.append(count_CD/(len(text)-count_ignore))
        values_CC.append(stats.median(current_CC))
        values_DT.append(stats.median(current_DT))
        values_IN.append(stats.median(current_IN))
        values_JJ.append(stats.median(current_JJ))
        values_VB.append(stats.median(current_VB))
        values_NN.append(stats.median(current_NN))
        values_RB.append(stats.median(current_RB))
        values_EX.append(stats.median(current_EX))
        values_PO.append(stats.median(current_PO))
        values_CD.append(stats.median(current_CD))
    df['CC_freq'] = values_CC
    df['DT_freq'] = values_DT
    df['IN_freq'] = values_IN
    df['JJ_freq'] = values_JJ
    df['VB_freq'] = values_VB
    df['NN_freq'] = values_NN
    df['RB_freq'] = values_RB
    df['EX_freq'] = values_EX
    df['PO_freq'] = values_PO
    df['CD_freq'] = values_CD

    return df

def get_yules_k(token_counter):
    s1 = sum(token_counter.values())
    s2 = sum([freq ** 2 for freq in token_counter.values()])
    k = 10000 * (s2 - s1) / (s1 ** 2)
    return k

def get_brunet_w(token_counter):
    n = sum(token_counter.values())
    v = len(token_counter)
    w = n ** (v ** (-0.165))
    return w

def get_honore_r(token_counter, hapaxes):
    n = sum(token_counter.values())
    v = len(token_counter)
    r = (100 * math.log(n)) / (1 - (hapaxes/v))
    return r

def get_simpson(token_counter):
    n = sum(token_counter.values())
    up = sum([(freq * (1 - freq)) for freq in token_counter.values()])
    do = n * (1 - n)
    s = up / do
    return abs(s)

def get_wl_metrics(input_df):
    df = input_df.copy()
    
    hapax_values = []
    hapax_dis_values = []
    yules_k_values = []
    brunet_w_values = []
    honore_r_values = []
    simpson_values = []
    size = df.shape[0]
    for i in range(size):
        print("{}/{}".format(i, size), end='\r')
        doc = ""
        descriptions_ = df['full_description'][i]
        for desc in descriptions_:
            doc += desc
            doc += " "
        doc = doc[:-1]

        doc = preprocess_text(doc, remove_punctuation=True)

        tokens = nltk.word_tokenize(doc)
        token_counter = collections.Counter(tok for tok in tokens)

        hapaxes = sum([1 for freq in token_counter.values() if freq == 1])
        hapaxes_dis = sum([2 for freq in token_counter.values() if freq == 2])
        yules_k = get_yules_k(token_counter)
        brunet_w = get_brunet_w(token_counter)
        honore_r = get_honore_r(token_counter, hapaxes)
        simpson = get_simpson(token_counter)
        
        hapax_values.append(hapaxes/len(tokens))
        hapax_dis_values.append(hapaxes_dis/len(tokens))
        yules_k_values.append(yules_k)
        brunet_w_values.append(brunet_w)
        honore_r_values.append(honore_r)
        simpson_values.append(simpson)
    df['hapax_freq'] = hapax_values
    df['hapax_dis_freq'] = hapax_dis_values
    df['yules_k'] = yules_k_values
    df['brunet_w'] = brunet_w_values
    df['honore_r'] = honore_r_values
    df['simpson'] = simpson_values
    
    return df

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_spelling_mistakes_metrics(input_df, path):
    df = input_df.copy()

    spell = SpellChecker()
    spell.word_frequency.load_text_file(path + 'en_full.txt')
    spell.word_frequency.load_words(['httpaddr','airbnb','covid','coronavirus'])

    spe_values = []

    size = df.shape[0]
    for i in range(size):
        print("{}/{}".format(i, size), end='\r')
        doc = ""
        descriptions_ = df['full_description'][i]
        for desc in descriptions_:
            doc += desc
            doc += " "
        doc = doc[:-1]

        doc = preprocess_text(doc, remove_punctuation=True)
        
        tokens = nltk.word_tokenize(doc)
        misspelled = spell.unknown(tokens)
        
        counter = 0
        for word in misspelled:
            if (hasNumbers(word)):
                continue
            if (not isEnglish(word)):
                continue
            counter += 1
        spe_values.append(counter/len(tokens))
    df['spelling_mistakes_freq'] = spe_values

    return df
