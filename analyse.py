import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk import sentiment
#from nltk import word_tokenize

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Ejecutado este script obtenemos una variable llamada «sentences» donde tendremos frases de este texto, en distintas casillas de un array.
sentences = tokenizer.tokenize("love")

# sentence = tokenizer.tokenize('Big data is a new revolution')
analizador = SentimentIntensityAnalyzer()

for sentence in sentences:
    print(sentence)
    scores = analizador.polarity_scores(sentence)
    for key in scores:
         print(key, ': ', scores[key])
         print()