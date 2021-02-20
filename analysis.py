#!/usr/bin/env python
# coding: utf-8

# # Região dos Imports

# In[1]:


import json
import glob
import re
import csv
import string
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist


# # Função de cálculo de resultados

# Função que permite avaliar cada modelo, a partir das métricas Accuracy, Precision, Recall e F1.

# In[2]:


def metrics_result(true_positve, true_negative, false_positive, false_negative):
    print("Positivos verdadeiros: " , true_positive)
    print("Negativos verdadeiros: " , true_negative)
    print("Positivos falsos: " , false_positive)
    print("Negativos falsos: " , false_negative)
    print("\nPalpites:")
    print("\tPalpites correctos: ", true_positive + true_negative)
    print("\tPalpites errados: ", false_negative + false_positive)
    print("\nMétricas:")
    accuracy = round(((true_positive + true_negative)/(true_positive + true_negative+false_negative + false_positive))*100, 1)
    print("\tAccuracy: ", accuracy , "%")
    precision = round(true_positive/(true_positive+false_positive)*100,1)
    print("\tPrecision: ", precision , "%")
    recall = round(true_positive/(true_positive+false_negative)*100,1)
    print("\tRecall: ", recall , "%")
    f1 = round((2 * precision * recall)/(precision  + recall),1)
    print("\tF1: ", f1 , "%")


# # Função para tratamento dos dados

# In[3]:


def run_treatment(text):
        text = handle_hashtag(text)
        text = handle_URL(text)
        text = handle_elongated_words(text)
        text = handle_lower_case(text)
        text = handle_abbreviations(text)        
        text = handle_punctuation(text)        
        return text


# # 1.2 Preparação dos dados e criação de uma baseline

# O conjunto de dados escolhido para a realização deste trabalho foi o Tweets_EN_sentiment.json. Nesta etapa iremos realizar a baseline de forma a comparar-mos os resultados no futuro. Iremos usar 80% destes dados para o treino do modelo (as primeiras linhas do conjunto de dados) e os restantes dados (20%) será utilizado para teste. Visto que os dados já se encontram "desorganizados" não houve a necessidade de realizar um "Shuffle".
# 

# In[4]:


dataset = []
for tweet in open('../TM/data/en/Tweets_EN_sentiment.json', 'r'):
    dataset.append(json.loads(tweet))


# De maneira a termos uma noção de quantos tweets positivos e negativos foram usados para treino e quantos foram usados para teste, obtemos os primeiros 40.000 tweets (80% dos dados, referentes ao treino do modelo) e verificamos quantos elementos temos de cada categoria. Os restantes elementos de cada categoria, pertencem ao teste do modelo.

# In[5]:


pos_tweets_train, neg_tweets_train = 0, 0
train_size = int(len(dataset) * 0.8)
train_tweets = dataset[:train_size]
for tweet in train_tweets:
    if(tweet["class"] == "pos"):
        pos_tweets_train += 1
    else:
        neg_tweets_train += 1

print("Tweets positivos no modelo de treino: ", pos_tweets_train)
print("Tweets negativos no modelo de treino: ", neg_tweets_train)
print("Total de tweets no modelo de treino: ", train_size)

pos_tweets_test, neg_tweets_test = 0, 0
test_size = len(dataset) - train_size
test_tweets = dataset[train_size:]
for tweet in test_tweets:
    if(tweet["class"] == "pos"):
        pos_tweets_test += 1
    else:
        neg_tweets_test += 1
        
print("\nTweets positivos no modelo de teste: ", pos_tweets_test)
print("Tweets negativos no modelo de teste: ", neg_tweets_test)
print("Total de tweets no modelo de teste: ", test_size)


# Para fazer a análise de sentimento diretamente a um texto, decidimos escolher a ferramenta TextBlob. De maneira a estabelecer um resultado de comparação com o trabalho futuro a ser realizado, criamos uma Baseline, onde aplicamos um modelo pré-treinado, sem efectuar qualquer tipo de tratamento de dados. Posteriormente, iremos fazer uma avaliação dos resultados obtidos, através de métricas de performance, mais especificamente: Precision, Cobertura (Recall), F-measure e Accuracy.
# 
# ![image.png](attachment:image.png)
# 
# Figura 1 (Bhagyashri Wagh, Prof. J. V. Shinde, Prof. P. A. Kale. (2017). A Twitter Sentiment Analysis Using NLTK and Machine Learning Techniques, 9359(12), 37–44.)
# 
# Para aplicar a ferramenta TextBlob temos que aplicar a função TextBlob() ao texto do tweet. Depois de aplicar esta função, é nos fornecida uma propriedade, com o nome Sentiment que contém outras duas propriedades, Polarity e Subjectivity. A propriedade que nos interessa neste caso é a Polarity, pois é o que nos permite saber se a classificação foi correta ou não.

# In[6]:


true_positive, true_negative, false_positive, false_negative = 0,0,0,0
for tweet in test_tweets:
    text = tweet["text"]
    avaliation = TextBlob(text).sentiment.polarity
    if((tweet["class"] == "pos" and avaliation > 0) or (tweet["class"] == "neg" and avaliation <0)):     
        if((tweet["class"] == "pos" and avaliation > 0)):
            true_positive += 1
        else:
            true_negative += 1
    else:
        if(avaliation > 0):
            false_positive += 1
        else:
            false_negative += 1
                
correct_guesses = true_positive + true_negative
incorrect_guesses = false_negative + false_positive

print("-Baseline")
metrics_result(true_positive, true_negative, false_positive, false_negative)


# Neste caso, fazemos a avaliação de várias métricas de performance, obtendo valores de 52% em Accuracy, o que demonstra que o desempenho do modelo pré-treinado é bom, tendo em conta que os dados não foram tratados, representando o rácio de previsões correctas pelo total dos tweets testados. No caso do Precision, o resultado obtido foi 89.6% que representa o rácio de tweets positivos corretamente previstos para o total de positivos previstos. O valor obtido de Recall é de 51.9%, o que revela que o rácio de tweets positivos correctamente previstos é bom, visto ser acima de 50%. Por último, a medida F1 demonstra a média do peso da Precision e Recall, obtendo um valor de 65.7%.

# # 1.3 Aplicação de um léxico de sentimentos

# Para o desenvolvimento de um classificador de sentimentos, vamos utilizar o NCR EmoLex, em formato csv, disponibilizado pelos professores. Vamos criar um léxico apartir do csv de onde se vai considerar apenas as colunas "English", "Positive" e "Negative". Com esta informação vamos criar um dicionário em que a chave é a palavra e o valor será "-1" caso a palavra seja negativa e "1" caso seja positiva.
# 
# Foi verificado que existem casos em que uma palavra é considerada Positiva e Negativa, assim como casos em que uma palavra é considerada como não sendo Positiva nem Negativa. Nestes casos não adicionamos ao léxico e será depois considerada "Neutra".

# In[7]:


lexicon = {}
neutral = 0
with open("../TM/data/en/NCR-lexicon.csv", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";" )
    for row in reader:
        if ((row["Positive"] == "1" and row["Negative"] == "1") or (row["Positive"] == "0" and row["Negative"] == "0")):
            neutral += 1
        elif(row["Positive"] == "1"):
            lexicon[row["English"]] = 1
        elif(row["Negative"] == "1"):
            lexicon[row["English"]] = -1
            
print("Número de palavras no léxico: ", neutral+len(lexicon))
print("Número de palavras Neutras: ", neutral)
print("Número de palavras adicionadas (Positivas ou Negativas): ", len(lexicon))


# # Palavras alongadas

# A função handle_elongated_words vai ser usada para tratar das palavras "alongadas", por exemplo "happyyy". Para estes casos, consideramos que estas repetições são intensificadores do sentimento da palavra. Assim sendo, vamos retirar os caracteres repetidos e replicar a palavra uma vez, por exemplo "happy happy".

# In[8]:


def handle_elongated_words(text):
    test = text
    for el in re.finditer(r"(.)\1{2,}",text):
        test = test.replace(el.group(0),el.group(0)[0]+"_int")
    
    arrayAux = re.findall("\w+_int",test)
    
    for word_int in arrayAux:
        realWord = word_int.split("_int")[0]
        test = test.replace(word_int, realWord + " " + realWord)
    
    return test


# # Hashtags

# Criámos a função "handle_hashtag" que vai tratar de todas as hashtags presentes nos tweets, removendo apenas o símbolo '#', deixando a palavra intacta. 

# In[9]:


def handle_hashtag(text):
    return text.replace("#", "")


# # URL's

# A função "handle_URL" vai fazer o tratamento dos URL presentes nos tweets. Para isto usámos um regex, retirado do site https://www.regextester.com/96504 , e removemos o URL.

# In[10]:


def handle_URL(text):
    text = re.sub(r'''(?:(?:https?|ftp):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))?''',"",text)
    text = re.sub(' +',' ',text)
    return text


# # Minúsculas

# Função "handle_lower_case" para alterar todos os caracteres do tweet para minúsculas.

# In[11]:


def handle_lower_case(text):
    return text.lower()


# # Acrónimos/Abreviaturas

# Tabela para converter as abreviações/acrónimos nas palavras que cada caracter representa. Esta tabela apenas representa alguns acrónimos, nomeadamente aqueles que achamos que eram mais utilizados.  A tabela que usamos para fazer a conversão dos acrónimos é um ficherio CSV com o seguinte nome "acronyms.csv". Este ficheiro terá que se encontrar numa pasta com o nome "Trabalho1".
# 

# In[12]:


acronyms = {}
with open('../Trabalho1/acronyms.csv') as file:
    for row in file:
        row = row.replace("\n", "").split(",")
        key = row[0]
        value = row[1]
        acronyms[key] = value


# Função que permite tratar das abreviações/acrónimos. Se a palavra que estamos a avaliar se encontrar na tabela de acrónimos, irá fazer-se um match dessa palavra pelas palavras que representa. Se esta abreviatura não estiver representada na tabela de acrónimos, mantemos o acrónimo como está.

# In[13]:


def handle_abbreviations(text):
    test = []
    words = re.findall(r"[\w']+|[.,!?;]", text)
    for word in words:
        if word in acronyms:
            test.append(acronyms[word])
        else:
            test.append(word)
            
    return " ".join(test)


# # Pontuação da frase

# Função que trata de remover todos os sinais de pontuação de uma frase.

# In[14]:


def handle_punctuation(text):
    words = re.sub(r'[^\w\s]','', text).replace("  ", " ")
    return words


# # Aplicação do Léxico com opções 

# Função utilizada para avaliar todos as aplicações do léxico. Estas opções passam desde aplicar Lemmatization, Stemming ou simplesmente o léxico. Para além disso, podemos decidir correr uma das duas opções de tratamento de negação com qualquer uma das opções anteriormente descritas.

# In[15]:


def apply_lexicon(isLemma, isStemmer, isNegateOp1, isNegateOp2):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    true_positive, true_negative, false_positive, false_negative = 0,0,0,0
    for tweet in test_tweets:
        
        text = tweet["text"]
        
        if(isNegateOp1):
            text = negate_op1(text)
        elif(isNegateOp2):
            text = negate_op2(text)
            
        text = run_treatment(text)
        positive, negative = 0,0
        text = text.split(" ")
        occurrences = FreqDist(text)
        
        if(isLemma):
            if(isNegateOp1 or isNegateOp2):
                for occ in occurrences.most_common():
                    if(occ[0].startswith("not_")):
                        if(lexicon.get(occ[0][4:])==1 or lexicon.get(lemmatizer.lemmatize(occ[0][4:]))==1):
                            negative += occ[1]
                        elif(lexicon.get(occ[0][4:])==-1 or lexicon.get(lemmatizer.lemmatize(occ[0][4:]))==-1):
                            positive += occ[1]
                    else:
                        if(lexicon.get(occ[0])==1 or lexicon.get(lemmatizer.lemmatize(occ[0]))==1):
                            positive += occ[1]
                        elif(lexicon.get(occ[0])==-1 or lexicon.get(lemmatizer.lemmatize(occ[0]))==-1):
                            negative += occ[1]
            else:
                for occ in occurrences.most_common():
                    if(lexicon.get(occ[0])==1 or lexicon.get(lemmatizer.lemmatize(occ[0]))==1):
                        positive += occ[1]
                    elif(lexicon.get(occ[0])==-1 or lexicon.get(lemmatizer.lemmatize(occ[0]))==-1):
                        negative += occ[1]
        elif(isStemmer):
            if(isNegateOp1 or isNegateOp2):
                for occ in occurrences.most_common():
                    if(occ[0].startswith("not_")):
                        if(lexicon.get(occ[0][4:])==1 or lexicon.get(stemmer.stem(occ[0][4:]))==1):
                            negative += occ[1]
                        elif(lexicon.get(occ[0][4:])==-1 or lexicon.get(stemmer.stem(occ[0][4:]))==-1):
                            positive += occ[1]
                    else:
                        if(lexicon.get(occ[0])==1 or lexicon.get(stemmer.stem(occ[0]))==1):
                            positive += occ[1]
                        elif(lexicon.get(occ[0])==-1 or lexicon.get(stemmer.stem(occ[0]))==-1):
                            negative += occ[1]
            else:
                for occ in occurrences.most_common():
                    if(lexicon.get(occ[0])==1 or lexicon.get(stemmer.stem(occ[0]))==1):
                        positive += occ[1]
                    elif(lexicon.get(occ[0])==-1 or lexicon.get(stemmer.stem(occ[0]))==-1):
                        negative += occ[1]
        else:
            if(isNegateOp1 or isNegateOp2):
                for occ in occurrences.most_common():
                    if(occ[0].startswith("not_")):
                        if(lexicon.get(occ[0][4:])==1):
                            negative += occ[1]
                        elif(lexicon.get(occ[0][4:])==-1):
                            positive += occ[1]
                    else:
                        if(lexicon.get(occ[0])==1):
                            positive += occ[1]
                        elif(lexicon.get(occ[0])==-1):
                            negative += occ[1]
            else:
                for occ in occurrences.most_common():
                    if(lexicon.get(occ[0])==1):
                        positive += occ[1]
                    elif(lexicon.get(occ[0])==-1):
                        negative += occ[1]
                    
        if((tweet["class"] == "pos" and (positive>negative)) or (tweet["class"] == "neg" and (positive<negative))):     
            if(tweet["class"] == "pos" and positive>negative):
                true_positive += 1
            elif(tweet["class"] == "neg" and positive<negative): 
                true_negative += 1
        else:
            if(positive>negative):
                false_positive += 1
            elif(positive<negative):
                false_negative += 1

    metrics_result(true_positive, true_negative, false_positive, false_negative)


# # Resultados com aplicação de Léxico

# In[16]:


print("-Aplicação do Léxico\n")
apply_lexicon(False,False,False,False)
print("\n-Aplicação do Léxico com Lemmatization\n")
apply_lexicon(True,False,False,False)
print("\n-Aplicação do Léxico com Stemming\n")
apply_lexicon(False,True,False,False)


# # Tratamento da Negação

# Para tratamento da negação, foram consideradas as seguintes palavras: "no", "not" e qualquer palavra com a terminação "n't". 
# Decidimos tomar duas abordagens para o tratamento da negação, e posteriormente, iremos avaliar qual delas se comporta da melhor forma. 
# 
# - 1º Opção: Considerar todas as palavras como negação até chegar a um terminador de frase, nomeadamente: ".", ",", "!", "?", ";", ":","but", "however", "although", "nevertheless", "still", "yet" e "though".
# 
# - 2º Opção: Considerar todas as palavras como negação até chegar a uma palavra com polaridade positiva, ou até chegar a um dos terminadores de frase (anteriormente apresentados na 1º Opção).
# 
# Todas as palavras que são consideradas negação, serão transformadas na seguinte sequência: "not_" + palavra original.

# # 1º Opção de Tratamento de Negação

# In[17]:


def negate_op1(text):
    
    text = handle_lower_case(text)
    text = text.split()
    delimit = ",.?!:;"
    terminators = ['but', 'however', "although", "nevertheless", "still", "yet", "though"]
    final_result = []
    cont_negation = False
    for word in text:
        if(not cont_negation):
            final_result.append(word)
        else:
            if((word not in delimit) and (word not in terminators)):
                final_result.append("not_" + word)
            else:
                final_result.append(word)
            
        if (word == "not" or word == "no" or word.endswith("n't")):
            cont_negation = True
        elif(any(c in word for c in delimit) or (word in terminators)):
            cont_negation = False
    
    return ' '.join(final_result)


# # 2º Opção de Tratamento de Negação

# In[18]:


def negate_op2(text):
    
    text = handle_lower_case(text)
    text = text.split()
    delimit = ",.?!:;"
    terminators = ['but', 'however', "although", "nevertheless", "still", "yet", "though"]
    final_result = []
    cont_negation = False
    for word in text:
        if(not cont_negation):
            final_result.append(word)
        else:
            if((word not in delimit) and (word not in terminators)):
                final_result.append("not_" + word)
                if(lexicon.get(word) == 1):
                    cont_negation = False
            else:
                final_result.append(word)
            
        if (word == "not" or word == "no" or word.endswith("n't")):
            cont_negation = True
        elif(any(c in word for c in delimit) or (word in terminators)):
            cont_negation = False
    
    return ' '.join(final_result)


# # Resultados com Opções de Negação

# In[19]:


print("\n-Aplicação do Léxico com Negação Opção 1\n")
apply_lexicon(False,False,True,False)
print("\n-Aplicação do Léxico com Negação Opção 2\n")
apply_lexicon(False,False,False,True)
print("\n-Aplicação do Léxico com Lematization com Negação Opção 1\n")
apply_lexicon(True,False,True,False)
print("\n-Aplicação do Léxico com Lematization com Negação Opção 2\n")
apply_lexicon(True,False,False,True)
print("\n-Aplicação do Léxico com Stemming com Negação Opção 1\n")
apply_lexicon(False,True,True,False)
print("\n-Aplicação do Léxico com Stemming com Negação Opção 2\n")
apply_lexicon(False,True,False,True)


# # Comparação de Resultados

# <b>Baseline</b>
# Accuracy: 52.0 %
# 
# <b>Aplicação do Léxico</b>
# - Sem opções -> Accuracy: 79.0 %
# - Com Lemmatization -> Accuracy: 78.5 %
# - Com Stemming -> Accuracy: 78.6 %
# 
# <b>Aplicação do Léxico com Tratamento de Negação</b>
# - Com opção 1 de tratamento de negação -> Accuracy: <b>79.6 %</b>
# - Com opção 2 de tratamento de negação -> Accuracy: <b>79.6 %</b>
# - Com Lemmatization e com opção 1 de tratamento de negação -> Accuracy: 79.2 %
# - Com Lemmatization e com opção 2 de tratamento de negação -> Accuracy: 79.3 %
# - Com Stemming e com opção 1 de tratamento de negação -> Accuracy: 79.0 %
# - Com Stemming e com opção 2 de tratamento de negação -> Accuracy: 79.1 %
# 
# Nesta comparação de resultados, iremos concentrar nos valores de Accuracy. Posto isto, podemos verificar que, comparativamente com os valores de Baseline, estes subiram bastante, não só em Accuracy, mas nas restantes métricas de desempenho. No início, tinhamos um valor de 52.0 % de Accuracy na Baseline e no melhor sistema desenvolvido os valores foram 79.6 %, com os modelos de aplicação do léxico e com qualquer uma das das opções de tratamento de negação. Isto demonstra que o tratamento dos dados pode influenciar em muito os resultados. Os resultados obtidos com Lemmatization e tratamento de negação foram melhores que os de Stemming, não sendo uma melhoria substancial. Podemos também concluir que com o tratamento da negação, os resultados melhoraram em todos os casos.

# In[ ]:




