### This script is used to extract 108 features from the compiled dataset dict2.csv
### The details of the 108 features are mentioned in the project report
### The output of this script will be used as training and test data from our algorithm


import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from pprint import pprint
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import PorterStemmer

train = pd.read_csv("dict2.csv", header=0, encoding = 'utf-8', delimiter= "*", quoting=3)

### Creating placeholders for the 108 features to be extracted

train["uniqueWords"] = "0"
train["hapaxLegomenon"] = "0"
train["disLegomenon"] = "0"
train["stops"] = "0"

for i in range(1,27):
    train["sentenceLength%d"%i] = "0"
    train["wordLength%d"%i] = "0"
    train["nConjunctions%d"%i] = "0"
    train["nPronouns%d"%i] = "0"


### Running a loop over each Editorial Article

counter = 0     ## To keep count of number of editorial whose features have been extracted
for edit in train['editorial']:
    counter += 1
    edit_id = train.loc[train["editorial"]==edit,"id"].values[0]    ## storing id to access dataframe later

    ### Sentence Length Distribution (26 features here .'. 26 feature yet)
    ###
    sents = edit.split('.') # split into sentences
    nsents = len(sents) # count the number of sentences
    sl = [0]*26 # initialise array for count of each sentence length

    ## Count the number of sentences of each length
    for x in sents:
        senlength = len(x.split())
        if senlength < 26:
            sl[senlength - 1] += 1
        else:
            sl[25] +=1

    ## Normalise sentence length distribution by total number of sentences
    j=0
    for i in sl:
        if(nsents != 0):
            i = i/nsents
            sl[j] = i
            j += 1

    ### Word Length Distribution Feature (26 Features here .'. 52 features yet)
    ###
    j=0
    words = edit.split()  # split into words by splitting on space
    for word in words:
        words[j] = word.strip(string.punctuation) # strip punctuation
        j += 1

    nwords = j
    wl = [0]*26

    ## count number of words of each length

    for word in words:
        wordlength = len(word)
        if wordlength < 26:
            wl[wordlength - 1] += 1
        else:
            sl[25] +=1

    ## normalise word length by total number of words

    j=0
    for i in wl:
        if (nwords != 0):
            i = i/nwords
            wl[j] = i
            j += 1

    ### Conjunction and Pronoun Distribution Feature (26 + 26 Features here .'. 104 features yet)
    ###
    conjunctionDistri = [0]*26  ##initialising array to hold conjunctionDistribution
    pronounDistri = [0]*26      ##initialising array to hold pronounDistribution
    for x in sents:
        wordsinsent = x.split()
        k=0
        for word in wordsinsent:
            wordsinsent[k] = word.strip(string.punctuation)   ##removing punctuation from each word
            k += 1
        while '' in wordsinsent:
            wordsinsent.remove('')
        taggedwordsinsent = nltk.pos_tag(wordsinsent)      ## Using NLTK Library for Parts of Speech Tagging
        nconjunctionsinsent = 0
        npronounsinsent = 0
        for tag in taggedwordsinsent:
            if ((tag[1] == 'CC') | (tag[1] == 'IN')):
                nconjunctionsinsent += 1
            if (tag[1] == 'PRP'):
                npronounsinsent += 1
        if (nconjunctionsinsent < 26) & (nconjunctionsinsent != 0):
            conjunctionDistri[nconjunctionsinsent - 1] += 1
        if(nconjunctionsinsent >25):
            conjunctionDistri[25] += 1
        if (npronounsinsent < 26) & (npronounsinsent != 0):
            pronounDistri[npronounsinsent - 1] += 1
        if(npronounsinsent >25):
            pronounDistri[25] += 1

    ## Normalising Conjunction Distribution by number of sentences
    j=0
    for i in conjunctionDistri:
        if (nsents != 0):
            i = i/nsents
            conjunctionDistri[j] = i
            j += 1

    ## Normalising Pronoun Distribution by number of sentences
    j=0
    for i in pronounDistri:
        if (nsents != 0):
            i = i/nsents
            pronounDistri[j] = i
            j += 1


    ### hapaxLegomenon and disLegomenon Feature
    ### Also extracting Number of Unique Words and Number of Stop Words (4 features here and .'. 108 features in total.)
    nhapax = 0
    ndis = 0
    nuniques = 0
    nstops = 0
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    stemmedwords = [PorterStemmer().stem_word(word) for word in filtered_words]
    nfiltwords = len(filtered_words)
    wordfrequncy ={}
    for word in stemmedwords:
        if word in wordfrequncy:
            wordfrequncy[word] += 1
        else:
            wordfrequncy[word] = 1
    nuniques = len(wordfrequncy)
    for word, times in wordfrequncy.items():
        #print ("%s ------ %d times" % (word, times))
        if (times == 1):
            nhapax += 1
        if(times == 2):
            ndis += 1

    if (nuniques != 0):
        nhapax = nhapax/nuniques
        ndis = ndis/nuniques
    if (len(filtered_words) != 0):
        nuniques = nuniques/len(filtered_words)

    nstops = len(words) - len(filtered_words)
    if (len(words) != 0):
        nstops = nstops/len(words)


    ### Assigning values back to the dataframe

    train.loc[train["id"]==edit_id,"uniqueWords"]= nuniques
    train.loc[train["id"]==edit_id,"hapaxLegomenon"]= nhapax
    train.loc[train["id"]==edit_id,"disLegomenon"]= ndis
    train.loc[train["id"]==edit_id,"stops"]= nstops

    for i in range(1,27):
        train.loc[train["id"]==edit_id,"sentenceLength%d"%i]= sl[i-1]
        train.loc[train["id"]==edit_id,"wordLength%d"%i]= wl[i-1]
        train.loc[train["id"]==edit_id,"nConjunctions%d"%i]= conjunctionDistri[i-1]
        train.loc[train["id"]==edit_id,"nPronouns%d"%i]= pronounDistri[i-1]

    print(counter)

train.drop('editorial', axis=1, inplace=True)


###### Residual Data Cleaning becuase of some discrepancies in the way newspaper names appear in the dataframe
###### This happend because of conversion from a dictionary to a dataframe as some articles had multiple instances.

for newspapername in train['newspaper']:
    edit_id = train.loc[train["newspaper"]==newspapername,"id"].values[0]

    if (newspapername == "['Economic Times', 'Economic Times']" ) | (newspapername == "['Economic Times']" ):
        train.loc[train["id"]==edit_id,"newspaper"]= "ET"

    if (newspapername == "['Guardian', 'Guardian']" ) | (newspapername == "['Guardian']" ):
        train.loc[train["id"]==edit_id,"newspaper"]= "Guardian"

    if (newspapername == "['TOI', 'TOI']" ) | (newspapername == "['TOI']" ):
        train.loc[train["id"]==edit_id,"newspaper"]= "TOI"

    if (newspapername == "['Indian Express']" ) | (newspapername == "['Indian Express', 'Indian Express']" ):
        train.loc[train["id"]==edit_id,"newspaper"]= "IEx"

    if (newspapername == "['Financial Express']" ) | (newspapername == "['Financial Express', 'Financial Express'] " ):
        train.loc[train["id"]==edit_id,"newspaper"]= "FEx"

    if (newspapername == "['Deccan Chronicle']" ) | (newspapername == "['Deccan Chronicle', 'Deccan Chronicle']" ):
        train.loc[train["id"]==edit_id,"newspaper"]= "DC"

    if (newspapername == "['TOI', 'TOI', 'TOI', 'TOI', 'TOI', 'TOI']" ) | (newspapername == "['TOI', 'TOI', 'TOI', 'TOI', 'TOI']" )| (newspapername == "['TOI', 'TOI', 'TOI']" ):
        train.loc[train["id"]==edit_id,"newspaper"]= "TOI"

### Writing output to a csv file to be used for further classification

train.to_csv('features3.csv', sep='\t')
