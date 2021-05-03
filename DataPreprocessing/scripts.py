import re
import string

def clean_text(text):
    text = re.sub(r"http\S+", "httpaddr", text)

    i = 0
    while (i < len(text)-1):
        if (text[i] == '.' and text[i+1].isdigit()):
            text = text[:i] + '' + text[i+1:]
            continue
        elif (text[i].isdigit() and not text[i+1].isdigit() and text[i+1] not in string.punctuation and text[i+1] != ' '):
            text = text[:i+1] + ' ' + text[i+1:]
            continue
        i += 1

    text = re.sub(r'\.(\.|\!|\?|\,|\;| )* *','. ', text)
    text = re.sub(r'\!(\.|\!|\?|\,|\;| )* *','! ', text)
    text = re.sub(r'\?(\.|\!|\?|\,|\;| )* *','? ', text)
    text = re.sub(r'\,(\.|\!|\?|\,|\;| )* *',', ', text)
    text = re.sub(r'\;(\.|\!|\?|\,|\;| )* *','; ', text)

    filtering = ['\r\n', '\r', '\n', '\t', '(URL HIDDEN)','(Email hidden by Airbnb)','(Hidden by Airbnb)','(Website hidden by Airbnb)','(Phone number hidden by Airbnb)']
    replacing = [' ',' ',' ',' ','','','','','']
    for f in range(len(filtering)):
        text = text.replace(filtering[f],replacing[f])

    text = text.rstrip()
    if (len(text) > 0):
        if (text[-1] not in ['.','!','?']):
            text += '. '
        else:
            text += ' '

    return text
