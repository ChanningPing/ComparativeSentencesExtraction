import nltk.data
import nltk.tokenize.punkt
import pickle
import codecs


text='The benets of model combination are also very substantial. In all cases the (uniformly) combined model performed better than the best single model. As a sighte ect, model averaging also deliberates from selecting the \optimal" model dimensionality. In terms of computational complexity, despite of the iterative nature of EM, the computing time for TEM model tting at K = 128 was roughly comparable to SVD in a standard implementation. For larger data sets one may also consider speeding up TEM by on-line learning [11]. Notice that the PLSI-Q scheme has the advantage that documents can be represented in a low{dimensional vector space (as in LSI), while PLSI-U requires the calculation of the high{ dimensional multinomials P(wjd) which oers advantages in terms of the space requirements for the indexing information that has to be stored. Finally, we have also performed an experiment to stress the importance of tempered EM over standard EM{based model tting. Figure 4 plots the performance of a 128 factor model trained on CRAN in terms of perplexity and in terms of precision as a function of . It can be seen that it is crucial to control the generalization performance of the model, since the precision is inversely correlated with the perplexity. In particular, notice that the model obtained by maximum likelihood estimation (at  = 1) actually deteriorates the retrieval performance.'
sent_detector = nltk.data.load('/usr/local/share/nltk_data/literature.pickle')
print(sent_detector.tokenize(text.strip()))
#print('\n-----\n'.join(sent_detector.tokenize(text.strip())))
