'''
This package is an implementation of the following three papers:
    (1) Jindal, N., & Liu, B. (2006, August). Identifying comparative sentences in text documents. In Proceedings of the 29th
annual international ACM SIGIR conference on Research and development in information retrieval (pp. 244-251). ACM.
    (2) Jindal, N., & Liu, B. (2006, July). Mining comparative sentences and relations. In AAAI (Vol. 22, pp. 1331-1336).
    (3) Ganapathibhotla, M., & Liu, B. (2008, August). Mining opinions in comparative sentences. In Proceedings of the 22nd
International Conference on Computational Linguistics-Volume 1 (pp. 241-248). Association for Computational Linguistics.
TODO:
train()
predict()
'''
from __future__ import print_function
import os
import sys
import nltk
import nltk.tokenize.punkt
import pickle
import codecs
import string
import csv
import sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from nltk.stem.snowball import SnowballStemmer

from collections import defaultdict

class Sequences:
    def __init__(self):
        self.seqs = []
        self.seq_labels = []

#global variables setting
reload(sys)
sys.setdefaultencoding('utf8')
stemmer = SnowballStemmer("english")
window_size=3 #radius from the keyword to generate the sequence
#sequences=[]# sequences
#seq_labels=[]#labels of the sequences
TAU=1
min_confidence=0.6
# the keyword list is derived from the keyweord list provided by Liu Bing, but we use the stemmed ones, we also remove repeated ones and phrases
# phrases are used for exact match in another part
keyword_dict = {'advantag': 1, 'after': 1, 'ahead': 1, 'all': 1, 'altern': 1, 'altogeth': 1, 'beat': 1, 'befor': 1,
                'behind': 1, 'both': 1, 'choic': 1, 'choos': 1, 'compar': 1, 'compet': 1, 'defeat': 1, 'differ': 1,
                'domin': 1, 'doubl': 1,
                'either': 1, 'equal': 1, 'equival': 1, 'exceed': 1, 'favor': 1, 'first': 1, 'fraction': 1,
                'half': 1, 'ident': 1, 'improv': 1,
                'inferior': 1, 'last': 1, 'lead': 1, 'least': 1, 'less': 1, 'like': 1, 'match': 1,
                'most': 1, 'near': 1, 'nobodi': 1,
                'none': 1, 'nonpareil': 1, 'onli': 1, 'outclass': 1, 'outdist': 1, 'outdo': 1, 'outfox': 1,
                'outmatch': 1, 'outperform': 1,
                'outsel': 1, 'outstrip': 1, 'outwit': 1, 'peerless': 1, 'prefer': 1, 'recommend': 1, 'rival': 1,
                'same': 1, 'second': 1,
                'similar': 1, 'superior': 1, 'thrice': 1, 'togeth': 1, 'top': 1, 'twice': 1, 'unlik': 1,
                'unmatch': 1, 'unriv': 1,
                'versus': 1, 'vs': 1, 'win': 1}
comparative_phrases = ['number one', 'on par with', 'one of few', 'up against']

def SentenceTokenizationTrain(): #the training is unsupervised. More could be added to the corpus at any time.
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    text = codecs.open("SentenceTokenizationTrainCorpus", "r", "utf8").read()
    tokenizer.train(text)
    out = open("/usr/local/share/nltk_data/literature.pickle", "wb")
    pickle.dump(tokenizer, out)
    out.close()

def Paragraph_to_Sentence(paragraph):
    '''
    :param paragraph: a paragraph with multiple sentence
    :return: a list of sentences
    '''
    #text = 'The benets of model combination are also very substantial. In all cases the (uniformly) combined model performed better than the best single model. As a sighte ect, model averaging also deliberates from selecting the \optimal" model dimensionality. In terms of computational complexity, despite of the iterative nature of EM, the computing time for TEM model tting at K = 128 was roughly comparable to SVD in a standard implementation. For larger data sets one may also consider speeding up TEM by on-line learning [11]. Notice that the PLSI-Q scheme has the advantage that documents can be represented in a low{dimensional vector space (as in LSI), while PLSI-U requires the calculation of the high{ dimensional multinomials P(wjd) which oers advantages in terms of the space requirements for the indexing information that has to be stored. Finally, we have also performed an experiment to stress the importance of tempered EM over standard EM{based model tting. Figure 4 plots the performance of a 128 factor model trained on CRAN in terms of perplexity and in terms of precision as a function of . It can be seen that it is crucial to control the generalization performance of the model, since the precision is inversely correlated with the perplexity. In particular, notice that the model obtained by maximum likelihood estimation (at  = 1) actually deteriorates the retrieval performance.'
    sent_detector = nltk.data.load('/usr/local/share/nltk_data/literature.pickle')
    #print('\n-----\n'.join(sent_detector.tokenize(paragraph.strip())))
    return sent_detector.tokenize(paragraph.strip())


def getSequence (tagged_tuples, idx, tag, word, window_size,label,seq_object):
    '''
    given a tagged sentence, index of the keyword, tag of the keyword, the keyword itself, the window size, return a sequence
    :param tagged_tuples:tagged sentence
    :param idx:index of the keyword
    :param tag:tag of the keyword
    :param word: keyword itself
    :param window_size: radius from the keyword to form the sequence
    :param label: label of the tagged sentence
    :return: modified global list: sequences and seq_labels
    '''
    start = idx - window_size if idx - window_size >= 0  else 0  # start index of sequence
    end = idx + window_size + 1 if (idx + window_size + 1) <= len(tagged_tuples) else len(
        tagged_tuples)  # end index of sequence
    left_sub_tuples = [i[1] for i in tagged_tuples[start: idx]]  # tags before keyword in the sequence
    right_sub_tuples = [i[1] for i in tagged_tuples[idx + 1: end]]  # tags after keyword in the sequence
    keyword = []
    keyword.append(tag + '_' + word)  # make keyword as a list
    sub_tuples = left_sub_tuples + keyword + right_sub_tuples  # concatenate together

    #print(sub_tuples)
    seq_object.seqs.append(sub_tuples)
    #for sequence in seq_object.seqs:
        #print(sequence)
    seq_object.seq_labels.append(label)

    return seq_object
    #print(tagged_tuples)
    #print("in the other function:"+tag+","+word)
    #print(sub_tuples)
    #print(seq_labels)

def SequenceBuilder(sentences,labels, window_size, seq_object):
    '''
    given a list of sentences, generate a set of sequences
    :param sentences: a list of sentences
    :param window_size: the radius from the keyword to generate the sequence
    :param seq_object: a object of sequences and corresponding labels
    :return:
    '''

    #seq_object = Sequences()
    for id, text in enumerate(sentences):
        flag='NO' #whether this is a comparative candidate,flag=1:is candidate;flag=0: is not a candidate



        #text = "heavier  than the previous algorithm, but the previous one is number one, as efficient as it seems, our work improves rest of the work "
        #remove punctuations
        text = text.translate(None, string.punctuation)
        #POS tag the sentence
        print(text)
        tagged_tuples = nltk.pos_tag(nltk.word_tokenize(text.lower()))


        # if the sentence contains standard comparative, keep it
        for idx, item in enumerate(tagged_tuples):
            if item[1] == 'JJR' or item[1] == 'RBR' or item[1] == 'JJS' or item[1] == 'RBS':
                flag='YES'
                #print(item)
                seq_object = getSequence(tagged_tuples, idx, item[1], item[0], window_size,labels[id], seq_object)

        # if the sentence contains as {} as,keep it, here we increase window size by 1 to accomodate the as...as as context
        indices = [i for i, item in enumerate(tagged_tuples) if item[0] == 'as' and item[1] == 'RB']
        for index in indices:
            if (tagged_tuples[index + 1][1] == 'JJ' or tagged_tuples[index + 1][1] == 'RB') and \
                            tagged_tuples[index + 2][
                                0] == 'as' and tagged_tuples[index + 2][1] == 'IN':
                flag = 'YES'
                #print('as X as')
                seq_object = getSequence(tagged_tuples, index + 1, tagged_tuples[index + 1][1], tagged_tuples[index + 1][0],
                            window_size + 1,labels[id], seq_object)

        # if the sentence contains certain keyword, keep it
        for idx, item in enumerate(tagged_tuples):
            if stemmer.stem(item[0]) in keyword_dict:
                flag = 'YES'
                #print(stemmer.stem(item[0]))
                seq_object = getSequence(tagged_tuples, idx, item[1], str(stemmer.stem(item[0])), window_size,labels[id], seq_object)

        # if the sentence contains phrases, use this as a feature
        for phrase in comparative_phrases:
            if phrase in text:
                flag = 'YES'
                #print("candidate based on \'" + phrase + "\'")



    #write sequences into file for later PrefixSpan sequence pattern mining
    file = open(('sequence.csv'), 'w')
    for sequence in seq_object.seqs:
        file.write("%s\n" % sequence)

    return seq_object


def train(filename):#train/test phase: one-time pass when training corpus is available.
    # train the sentence segmenter, only one-time effort
    #SentenceTokenizationTrain()

    # read training corpus line by line
    labels=[] #list of labels
    sentences=[]#list of sentences
    with open(filename) as f:#read labels and sentences from file
        rows = [line.split(',') for line in f]  # create a list of lists
        for row in rows:#row[0]: label; row[1]:sentence
            labels.append(row[0])
            sentences.append(row[1])

    # generate keyword-POS tag sequences from sentences
    seq_object = Sequences()
    seq_object = SequenceBuilder(sentences, labels, window_size,seq_object)

    # find sequence patterns with PrefixSpan, return in <rule, frequency, num of positive labels, number of negative labels.
    CSR_Rules= PrefixSpanCSR(seq_object.seqs, seq_object.seq_labels, TAU,min_confidence)
    # save the patterns to file
    with open(filename+'_CSR_rules.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['rule','frequency','sup','confidence','label','ID'])
        for rule in CSR_Rules:
            writer.writerow(rule)

    #Done: train with bayes classifier: read the sentence and label again, build a method to test match between sentence and rule
    # read the rules into a dictionary for quick lookup
    rule_dict = {}
    for rule in CSR_Rules:
        rule_dict[tuple(rule[0])] = rule[1:6] #frequency, sup, confidence, label, id
        #print(rule_dict[tuple(rule[0])])

    feature_matrix = []
    for idx, sentence in enumerate(sentences):
        features = Sentence_Rule_test(sentence, rule_dict, labels[idx])
        feature_matrix.append(features)
    print(feature_matrix)

    #train the Naive Bayes classifier

    clf = MultinomialNB()
    data = np.array(feature_matrix)
    data_X = data[:, 0: len(rule_dict)].astype(np.float)
    data_Y = data[:, len(rule_dict)]
    y_pred = clf.fit(data_X, data_Y).predict(data_X)
    print("Number of mislabeled points out of a total %d points : %d"  % (data_X.shape[0], (data_Y != y_pred).sum()))
    #TODO: cross-validation, predict a new paper
    #print(sequences)
    #print(seq_labels)
def Sentence_Rule_test (sentence, rule_dict, label):
    '''
    given a sentence and its label, return a feature list (feature meaning the covered rules)
    :param sentence: a raw sentence
    :param rule_dict: a rule dictionary
    :param label: label of sentence (comparative or not)
    :return: a feature list (features are all keys (rules) of the dict): 1-sentence cover this rule; 0-sentence doesn't cover this rule
    the last column of the feature will be the label
    '''
    print('=======================test now')
    sentence_as_list = []
    sentence_as_list.append(sentence)

    seq_object = Sequences()
    print('====first time object')
    for sequence in seq_object.seqs:
        print(sequence)
    seq_object = SequenceBuilder(sentence_as_list, label, window_size, seq_object)
    print('====after time object')
    for sequence in seq_object.seqs:
        print(sequence)
    features = [0] * (len(rule_dict) + 1)
    for sequence in seq_object.seqs:
        if tuple(sequence) in rule_dict:
            #print(sequence)
            index = rule_dict[tuple(sequence)][4]
            features[index] = 1
    features[len(rule_dict)] = label
    return features







def PrefixSpanCSR(sequences,seq_labels,TAU,min_confidence):
    '''
    This method is a slight modification of this implementation of PrefixSpan in Python:
    https://github.com/chuanconggao/PrefixSpan-py
    The original paper is here:
    Han, J., Pei, J., Mortazavi-Asl, B., Pinto, H., Chen, Q., Dayal, U., & Hsu, M. C. (2001, April). Prefixspan: Mining sequential patterns efficiently by prefix-projected pattern growth. In proceedings of the 17th international conference on data engineering (pp. 215-224).

    :param sequences: a set of sequences derived in SequenceBuilder()
    :param seq_labels: a set of labels derived in SequenceBuilder()
    :param TAU: the hyperparameter used in (Jindal & Liu, 2006) paper, used to give different items different min_sup
    :param min_confidence:the hyperparameter used in (Jindal & Liu, 2006) paper
    :return:
    '''
    results = []
    def mine_rec(patt, mdb):
        numYES = 0
        numNO = 0
        for coordinate in mdb:
            if seq_labels[coordinate[0]] == 'YES':
                numYES += 1
            else:
                numNO += 1
        # the pattern, the frequency of the pattern, the number of YES labels, the number of NO labels
        results.append((patt, len(mdb), numYES, numNO))
        occurs = defaultdict(list)
        for (i, startpos) in mdb:
            seq = sequences[i]
            for j in xrange(startpos, len(seq)):
                l = occurs[seq[j]]
                if len(l) == 0 or l[-1][0] != i:
                    l.append((i, j + 1))

        for (c, newmdb) in occurs.iteritems():
            # the following if-statement is pruning, we stop this since we will prune in final stage using both sup and conf
            # if len(newmdb) >= minsup:
            mine_rec(patt + [c], newmdb)

    mine_rec([], [(i, 0) for i in xrange(len(sequences))])

    #filtering  the patterns by min_sup and min_confidence
    count = 0
    CSR_rules=[]
    for result in results: #[0]the rule; [1]the frequency of the rule; [2]number of positive labels of this rule [3] number of negative labels of this rule
        positive_sup = result[2]
        negative_sup = result[3]
        min_sup = result[1] * TAU
        positive_confidence = result[2] / result[1]
        negative_confidence = result[3] / result[1]
        #generate positive and negative rules respectively
        if positive_sup >= min_sup and positive_confidence >= min_confidence: #positive rules
            rule = []
            rule.append(result[0])
            rule.append(result[1])
            rule.append(positive_sup)
            rule.append(positive_confidence)
            rule.append('YES')
            rule.append(count)
            count += 1
            CSR_rules.append(rule)
        if negative_sup >= min_sup and negative_confidence >= min_confidence: #negative rules
            rule = []
            rule.append(result[0])
            rule.append(result[1])
            rule.append(negative_sup)
            rule.append(negative_confidence)
            rule.append('NO')
            rule.append(count)
            count += 1
            CSR_rules.append(rule)

    return CSR_rules

def predict():
    #read a new paper
    with open('sentences_pLSI.txt', 'r') as file:
        paragraph = file.read().replace('\n', ' ')
    #paragraph to a list of sentences
    sentences = Paragraph_to_Sentence(paragraph)


def main():
    filename = "CSRTrainCorpus.txt"
    train(filename)





main()
