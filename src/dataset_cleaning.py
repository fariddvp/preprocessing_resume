import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from autocorrect import Speller

def dataset_cleaning():
    
    # Read dataset
    df = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/RawEnglishResume.csv')


    # Set the seed for reproducibility sample DataFrame
    np.random.seed(45)
    # Select 10.000 sample from raw dataset
    df_sample = df.sample(n=100, replace=True)  # Set replace=True if you want to allow duplicates
    print('Step 01 --> Select randomly 10.000 sample from: RawEnglishResume.csv')

    
    # Drop duplicate rows
    df_sample.drop_duplicates(inplace=True)
    print('step 02 --> Drop duplicate rows from sample dataset.')
    
    # Drop entire rows which the most of columns are NaN
    df_sample = df_sample.dropna(subset=['Academic Background', 'Skills', 'Industry Interest', 'Job Type Interest'], how='all')
    print('step 03 --> Drop rows that 4 major features are null, from sample dataset.')
    
    # Convert entire values in dataset to str
    df_sample = df_sample.astype(str)
    print('step 04 --> Entire values in dataset converted to string.')

    
    # Remove new lines, tabs and white spaces
    df_sample = df_sample.replace("\n", " ", regex=True) 
    df_sample = df_sample.replace("\t", " ", regex=True)
    df_sample = df_sample.applymap(lambda x: re.sub(r'\s+', ' ', x.strip()) if isinstance(x, str) else x)
    print('step 05 --> Removed new lines, tabs and white spaces.')


    # Encoded each string value in dataset to ASCII
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].apply(lambda x: x.encode("ascii", "ignore").decode())
    print('step 06 --> Encoded each string value in dataset to ASCII, and then decoded back to a Unicode string.')


    # Removal of emojis
    def remove_emoji(string):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)
    # Apply remove_emoji on entire column
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].apply(remove_emoji)
    print('step 07 --> Removed emojis from dataset values.')

    

    EMOTICONS = {
        u":‑\)":"Happy face or smiley",
        u":\)":"Happy face or smiley",
        u":-\]":"Happy face or smiley",
        u":\]":"Happy face or smiley",
        u":-3":"Happy face smiley",
        u":3":"Happy face smiley",
        u":->":"Happy face smiley",
        u":>":"Happy face smiley",
        u"8-\)":"Happy face smiley",
        u":o\)":"Happy face smiley",
        u":-\}":"Happy face smiley",
        u":\}":"Happy face smiley",
        u":-\)":"Happy face smiley",
        u":c\)":"Happy face smiley",
        u":\^\)":"Happy face smiley",
        u"=\]":"Happy face smiley",
        u"=\)":"Happy face smiley",
        u":‑D":"Laughing, big grin or laugh with glasses",
        u":D":"Laughing, big grin or laugh with glasses",
        u"8‑D":"Laughing, big grin or laugh with glasses",
        u"8D":"Laughing, big grin or laugh with glasses",
        u"X‑D":"Laughing, big grin or laugh with glasses",
        u"XD":"Laughing, big grin or laugh with glasses",
        u"=D":"Laughing, big grin or laugh with glasses",
        u"=3":"Laughing, big grin or laugh with glasses",
        u"B\^D":"Laughing, big grin or laugh with glasses",
        u":-\)\)":"Very happy",
        u":‑\(":"Frown, sad, andry or pouting",
        u":-\(":"Frown, sad, andry or pouting",
        u":\(":"Frown, sad, andry or pouting",
        u":‑c":"Frown, sad, andry or pouting",
        u":c":"Frown, sad, andry or pouting",
        u":‑<":"Frown, sad, andry or pouting",
        u":<":"Frown, sad, andry or pouting",
        u":‑\[":"Frown, sad, andry or pouting",
        u":\[":"Frown, sad, andry or pouting",
        u":-\|\|":"Frown, sad, andry or pouting",
        u">:\[":"Frown, sad, andry or pouting",
        u":\{":"Frown, sad, andry or pouting",
        u":@":"Frown, sad, andry or pouting",
        u">:\(":"Frown, sad, andry or pouting",
        u":'‑\(":"Crying",
        u":'\(":"Crying",
        u":'‑\)":"Tears of happiness",
        u":'\)":"Tears of happiness",
        u"D‑':":"Horror",
        u"D:<":"Disgust",
        u"D:":"Sadness",
        u"D8":"Great dismay",
        u"D;":"Great dismay",
        u"D=":"Great dismay",
        u"DX":"Great dismay",
        u":‑O":"Surprise",
        u":O":"Surprise",
        u":‑o":"Surprise",
        u":o":"Surprise",
        u":-0":"Shock",
        u"8‑0":"Yawn",
        u">:O":"Yawn",
        u":-\*":"Kiss",
        u":\*":"Kiss",
        u":X":"Kiss",
        u";‑\)":"Wink or smirk",
        u";\)":"Wink or smirk",
        u"\*-\)":"Wink or smirk",
        u"\*\)":"Wink or smirk",
        u";‑\]":"Wink or smirk",
        u";\]":"Wink or smirk",
        u";\^\)":"Wink or smirk",
        u":‑,":"Wink or smirk",
        u";D":"Wink or smirk",
        u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":‑\|":"Straight face",
        u":\|":"Straight face",
        u":$":"Embarrassed or blushing",
        u":‑x":"Sealed lips or wearing braces or tongue-tied",
        u":x":"Sealed lips or wearing braces or tongue-tied",
        u":‑#":"Sealed lips or wearing braces or tongue-tied",
        u":#":"Sealed lips or wearing braces or tongue-tied",
        u":‑&":"Sealed lips or wearing braces or tongue-tied",
        u":&":"Sealed lips or wearing braces or tongue-tied",
        u"O:‑\)":"Angel, saint or innocent",
        u"O:\)":"Angel, saint or innocent",
        u"0:‑3":"Angel, saint or innocent",
        u"0:3":"Angel, saint or innocent",
        u"0:‑\)":"Angel, saint or innocent",
        u"0:\)":"Angel, saint or innocent",
        u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"0;\^\)":"Angel, saint or innocent",
        u">:‑\)":"Evil or devilish",
        u">:\)":"Evil or devilish",
        u"\}:‑\)":"Evil or devilish",
        u"\}:\)":"Evil or devilish",
        u"3:‑\)":"Evil or devilish",
        u"3:\)":"Evil or devilish",
        u">;\)":"Evil or devilish",
        u"\|;‑\)":"Cool",
        u"\|‑O":"Bored",
        u":‑J":"Tongue-in-cheek",
        u"#‑\)":"Party all night",
        u"%‑\)":"Drunk or confused",
        u"%\)":"Drunk or confused",
        u":-###..":"Being sick",
        u":###..":"Being sick",
        u"<:‑\|":"Dump",
        u"\(>_<\)":"Troubled",
        u"\(>_<\)>":"Troubled",
        u"\(';'\)":"Baby",
        u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(-_-\)zzz":"Sleeping",
        u"\(\^_-\)":"Wink",
        u"\(\(\+_\+\)\)":"Confused",
        u"\(\+o\+\)":"Confused",
        u"\(o\|o\)":"Ultraman",
        u"\^_\^":"Joyful",
        u"\(\^_\^\)/":"Joyful",
        u"\(\^O\^\)／":"Joyful",
        u"\(\^o\^\)／":"Joyful",
        u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
        u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
        u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
        u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
        u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
        u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
        u"\('_'\)":"Sad or Crying",
        u"\(/_;\)":"Sad or Crying",
        u"\(T_T\) \(;_;\)":"Sad or Crying",
        u"\(;_;":"Sad of Crying",
        u"\(;_:\)":"Sad or Crying",
        u"\(;O;\)":"Sad or Crying",
        u"\(:_;\)":"Sad or Crying",
        u"\(ToT\)":"Sad or Crying",
        u";_;":"Sad or Crying",
        u";-;":"Sad or Crying",
        u";n;":"Sad or Crying",
        u";;":"Sad or Crying",
        u"Q\.Q":"Sad or Crying",
        u"T\.T":"Sad or Crying",
        u"QQ":"Sad or Crying",
        u"Q_Q":"Sad or Crying",
        u"\(-\.-\)":"Shame",
        u"\(-_-\)":"Shame",
        u"\(一一\)":"Shame",
        u"\(；一_一\)":"Shame",
        u"\(=_=\)":"Tired",
        u"\(=\^\·\^=\)":"cat",
        u"\(=\^\·\·\^=\)":"cat",
        u"=_\^=	":"cat",
        u"\(\.\.\)":"Looking down",
        u"\(\._\.\)":"Looking down",
        u"\^m\^":"Giggling with hand covering mouth",
        u"\(\・\・?":"Confusion",
        u"\(?_?\)":"Confusion",
        u">\^_\^<":"Normal Laugh",
        u"<\^!\^>":"Normal Laugh",
        u"\^/\^":"Normal Laugh",
        u"\（\*\^_\^\*）" :"Normal Laugh",
        u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
        u"\(^\^\)":"Normal Laugh",
        u"\(\^\.\^\)":"Normal Laugh",
        u"\(\^_\^\.\)":"Normal Laugh",
        u"\(\^_\^\)":"Normal Laugh",
        u"\(\^\^\)":"Normal Laugh",
        u"\(\^J\^\)":"Normal Laugh",
        u"\(\*\^\.\^\*\)":"Normal Laugh",
        u"\(\^—\^\）":"Normal Laugh",
        u"\(#\^\.\^#\)":"Normal Laugh",
        u"\（\^—\^\）":"Waving",
        u"\(;_;\)/~~~":"Waving",
        u"\(\^\.\^\)/~~~":"Waving",
        u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
        u"\(T_T\)/~~~":"Waving",
        u"\(ToT\)/~~~":"Waving",
        u"\(\*\^0\^\*\)":"Excited",
        u"\(\*_\*\)":"Amazed",
        u"\(\*_\*;":"Amazed",
        u"\(\+_\+\) \(@_@\)":"Amazed",
        u"\(\*\^\^\)v":"Laughing,Cheerful",
        u"\(\^_\^\)v":"Laughing,Cheerful",
        u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
        u'\(-"-\)':"Worried",
        u"\(ーー;\)":"Worried",
        u"\(\^0_0\^\)":"Eyeglasses",
        u"\(\＾ｖ\＾\)":"Happy",
        u"\(\＾ｕ\＾\)":"Happy",
        u"\(\^\)o\(\^\)":"Happy",
        u"\(\^O\^\)":"Happy",
        u"\(\^o\^\)":"Happy",
        u"\)\^o\^\(":"Happy",
        u":O o_O":"Surprised",
        u"o_0":"Surprised",
        u"o\.O":"Surpised",
        u"\(o\.o\)":"Surprised",
        u"oO":"Surprised",
        u"\(\*￣m￣\)":"Dissatisfied",
        u"\(‘A`\)":"Snubbed or Deflated"
    }



    # Removal of emotions
    def remove_emoticons(text):
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', text)
    # Apply remove_emotions on entire rows
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].apply(remove_emoticons)
    print('step 08 --> Removed the most common emotions from dataset values.')


    # Rmoval of URLs
    def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    # Apply remove_urls on entire rows
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].apply(remove_urls)
    print('step 09 --> Removed all of URLs from dataset values.')

    
    # Removal of HTML tags
    def remove_html(text):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)
    # Apply remove_urls on entire rows
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].apply(remove_html)
    print('step 10 --> Removed all of HTML tags from dataset values.')



    # Removal of punctuations
    def remove_punctuation(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    # Remove punctuations
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].apply(remove_punctuation)
    print('step 11 --> Removed all of punctuations from dataset values.')


    # Lower casing entire values
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].str.lower()
    print('step 12 --> Lower casted all of dataset values.')



    # Create a copy from sample DataFrame
    df_temp = df_sample.copy()


    # Removal of stopwords
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

    def remove_stopwords(token):
        return " ".join([word for word in str(token).split() if word not in STOPWORDS])

    for column in df_temp.columns:
        df_temp[column] = df_temp[column].apply(remove_stopwords)
    print('step 13 --> Removed all of stop words in dataset values.')


    # Tokenization all fo columns
    for column in df_temp.columns:
        df_temp[column] = df_temp[column].apply(lambda x: word_tokenize(str(x)))
    print('step 14 --> Tokenized all columns of dataset.')



    # Stemming
    stemmer = PorterStemmer()
    def stem_list(lst):
        return [stemmer.stem(word) for word in lst]

    def stem_words(tokens):
        return [stemmer.stem(token) for token in tokens]

    for column in df_temp.columns:
        df_temp[column] = df_temp[column].apply(stem_words)
        
    print('step 15 --> Stemmed all columns of dataset.')



    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]

    for column in df_temp.columns:
        df_temp[column] = df_temp[column].apply(lemmatize_words)
    print('step 16 --> Lemmatized all columns of dataset.')



    # Class for Remove Repeat Replacer
    class RepeatReplacer():
        def __init__(self):
            self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
            self.repl = r'\1\2\3'

        def replace(self, word):
            if wordnet.synsets(word):
                return word
            repl_word = self.repeat_regexp.sub(self.repl, word)
            if repl_word != word:
                return self.replace(repl_word)
            else:
                return repl_word
    # Apply on entire columns
    def reduce_repeated_characters(df):
        replacer = RepeatReplacer()
        for column in df.columns:
            df[column] = df[column].apply(lambda tokens: [replacer.replace(token) for token in tokens])
        return df

    df_temp = reduce_repeated_characters(df_temp)
    print('step 17 --> Removed all repeat characters from dataset.')

    
    
    
    # Initialize the autocorrect Speller
    spell = Speller()

    # Define the function to autocorrect a single token
    def autocorrect_token(token):
        if isinstance(token, str):
            return spell(token)
        elif isinstance(token, list):
            return [spell(t) for t in token]
        else:
            return token

    # Specify the columns you want to autocorrect
    columns_to_correct = df_temp.columns

    # Perform token correction for each column
    for column in columns_to_correct:
        df_temp[column] = df_temp[column].apply(lambda x: autocorrect_token(x))

    print('step 18 --> Corrected all tokens in the dataset.')

    
    
    # Export a csv file from cleaned dataset
    df_temp.to_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/data_cleaned_final.csv', index=False)

    print('Data cleaned and prepared for pre-processing phase, csv file exported too.')
    












    