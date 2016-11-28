'''
This package is an implementation of the following three papers:
    (1) Jindal, N., & Liu, B. (2006, August). Identifying comparative sentences in text documents. In Proceedings of the 29th
annual international ACM SIGIR conference on Research and development in information retrieval (pp. 244-251). ACM.
    (2) Jindal, N., & Liu, B. (2006, July). Mining comparative sentences and relations. In AAAI (Vol. 22, pp. 1331-1336).
    (3) Ganapathibhotla, M., & Liu, B. (2008, August). Mining opinions in comparative sentences. In Proceedings of the 22nd
International Conference on Computational Linguistics-Volume 1 (pp. 241-248). Association for Computational Linguistics.
'''
import os
import sys
import nltk
import nltk.tokenize.punkt
import pickle
import codecs
import string
from nltk.stem.snowball import SnowballStemmer

#global variables setting
reload(sys)
sys.setdefaultencoding('utf8')
stemmer = SnowballStemmer("english")
window_size=3 #radius from the keyword to generate the sequence
sequences=[]# sequences

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


def getSequence (tagged_tuples, idx, tag, word, window_size):
    start = idx - window_size if idx - window_size >= 0  else 0  # start index of sequence
    end = idx + window_size + 1 if (idx + window_size + 1) <= len(tagged_tuples) else len(
        tagged_tuples)  # end index of sequence
    left_sub_tuples = [i[1] for i in tagged_tuples[start: idx]]  # tags before keyword in the sequence
    right_sub_tuples = [i[1] for i in tagged_tuples[idx + 1: end]]  # tags after keyword in the sequence
    keyword = []
    keyword.append(tag + '_' + word)  # make keyword as a list
    sub_tuples = left_sub_tuples + keyword + right_sub_tuples  # concatenate together

    sequences.append(sub_tuples)
    #print(tagged_tuples)
    #print("in the other function:"+tag+","+word)
    #print(sequences)

def SequenceBuilder(sentences,window_size):
    '''
    :param sentences: a list of sentences
    :param window_size: the radius from the keyword to generate the sequence
    :return:
    '''
    for text in sentences:
        flag='NO' #whether this is a comparative candidate,flag=1:is candidate;flag=0: is not a candidate
        #text = "heavier  than the previous algorithm, but the previous one is number one, as efficient as it seems, our work improves rest of the work "
        text = text.translate(None, string.punctuation)
        tagged_tuples = nltk.pos_tag(nltk.word_tokenize(text.lower()))
        #print(tagged_tuples)

        # if the sentence contains standard comparative, keep it
        for idx, item in enumerate(tagged_tuples):
            if item[1] == 'JJR' or item[1] == 'RBR' or item[1] == 'JJS' or item[1] == 'RBS':
                flag='YES'
                #print(item)
                getSequence(tagged_tuples, idx, item[1], item[0], window_size)

        # if the sentence contains as {} as,keep it, here we increase window size by 1 to accomodate the as...as as context
        indices = [i for i, item in enumerate(tagged_tuples) if item[0] == 'as' and item[1] == 'RB']
        for index in indices:
            if (tagged_tuples[index + 1][1] == 'JJ' or tagged_tuples[index + 1][1] == 'RB') and \
                            tagged_tuples[index + 2][
                                0] == 'as' and tagged_tuples[index + 2][1] == 'IN':
                flag = 'YES'
                getSequence(tagged_tuples, index + 1, tagged_tuples[index + 1][1], tagged_tuples[index + 1][0],
                            window_size + 1)

        # if the sentence contains certain keyword, keep it
        for idx, item in enumerate(tagged_tuples):
            if stemmer.stem(item[0]) in keyword_dict:
                flag = 'YES'
                getSequence(tagged_tuples, idx, item[1], str(stemmer.stem(item[0])), window_size)

        # if the sentence contains phrases, use this as a feature
        for phrase in comparative_phrases:
            if phrase in text:
                flag = 'YES'
                #print("candidate based on \'" + phrase + "\'")

        print(flag+","+text)

    #write sequences into file for later PrefixSpan sequence pattern mining
    file = open(os.path.join(os.getcwd(), 'sequence.csv'), 'w')
    for sequence in sequences:
        file.write("%s\n" % sequence)


def main():

    #with open(os.path.join(os.getcwd(), 'sentences_pLSI.txt')) as f:
        #sentences = f.readlines()
    SentenceTokenizationTrain() #train the sentence segmenter, only one-time effort
    with open('sentences_pLSI.txt', 'r') as file:
        paragraph = file.read().replace('\n', ' ')
    sentences = Paragraph_to_Sentence(paragraph)
    SequenceBuilder(sentences,window_size)


main()
