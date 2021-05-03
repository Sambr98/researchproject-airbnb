import pandas as pd
import nltk
import re
import collections
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

def init_liwc_dataframe(input_df, category_names):
    df = input_df.copy()
    
    for i in category_names:
        df[i] = 0.0

    return df

def get_liwc(input_df, parse, category_names):
    df = input_df.copy()

    size = df.shape[0]
    for i in range(size):
        print("{}/{}".format(i, size), end='\r')

        properties_count = len(df['full_description'][i])
        doc = ""
        descriptions_ = df['full_description'][i]
        for desc in descriptions_:
            doc += desc
            doc += " "
        doc = doc[:-1]

        doc = preprocess_text(doc)

        tokens = nltk.word_tokenize(doc)

        doc_counts = collections.Counter(category for token in tokens for category in parse(token))

        for key in doc_counts:
            df[key][i] = float(doc_counts[key]) / float(properties_count)

    for cat in category_names:
        count_per_doc = df[cat].value_counts().to_dict()
        zero_count = count_per_doc.get(0.0)
        if (zero_count == None):
            zero_count = 0.0
        N = df.shape[0]
        doc_freq = N - zero_count
        idf = 1.0 / doc_freq if doc_freq != 0.0 else 0.0
        df[cat] = df[cat].mul(idf)

    return df
