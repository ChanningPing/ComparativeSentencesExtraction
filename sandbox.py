import nltk
import string
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def getSequence (tagged_tuples, idx, tag, word, window_size):
    start = idx - window_size if idx - window_size >= 0  else 0  # start index of sequence
    end = idx + window_size + 1 if (idx + window_size + 1) <= len(tagged_tuples) else len(
        tagged_tuples)  # end index of sequence
    left_sub_tuples = [i[1] for i in tagged_tuples[start: idx]]  # tags before keyword in the sequence
    right_sub_tuples = [i[1] for i in tagged_tuples[idx + 1: end]]  # tags after keyword in the sequence
    keyword = []
    keyword.append(tag + '_' + word)  # make keyword as a list
    sub_tuples = left_sub_tuples + keyword + right_sub_tuples  # concatenate together
    print(sub_tuples)
    sequences.append(sub_tuples)

text="heavier  than the previous algorithm, but the previous one is number one, as efficient as it seems, our work improves rest of the work "
text=text.translate(None, string.punctuation)
window_size=3
sequences=[]
tagged_tuples = nltk.pos_tag(nltk.word_tokenize(text.lower()))
print(tagged_tuples)

# if the sentence contains standard comparative, keep it
for idx, item in enumerate(tagged_tuples):
    if item[1] == 'JJR' or item[1] == 'RBR' or item[1] == 'JJS' or item[1] == 'RBS':
        getSequence(tagged_tuples, idx, item[1], item[0], window_size)



# if the sentence contains as {} as,keep it, here we increase window size by 1 to accomodate the as...as as context
indices = [i for i, item in enumerate(tagged_tuples) if item[0] == 'as' and item[1] == 'RB']
for index in indices:
    if (tagged_tuples[index + 1][1] == 'JJ' or tagged_tuples[index + 1][1] == 'RB') and tagged_tuples[index + 2][
        0] == 'as' and tagged_tuples[index + 2][1] == 'IN':
        getSequence(tagged_tuples, index + 1, tagged_tuples[index + 1][1], tagged_tuples[index + 1][0], window_size+1)

# if the sentence contains certain keyword, keep it
# the keyword list is derived from the keyweord list provided by Liu Bing, but we use the stemmed ones, we also remove repeated ones and phrases
# phrases are used for exact match in another part
keyword_dict = {'advantag':1,'after':1,'ahead':1,'all':1,'altern':1,'altogeth':1,'beat':1,'befor':1,
                'behind':1,'both':1,'choic':1,'choos':1,'compar':1,'compet':1,'defeat':1,'differ':1,'domin':1,'doubl':1,
                'either':1,'equal':1,'equival':1,'exceed':1,'favor':1,'first':1,'fraction':1,'half':1,'ident':1,'improv':1,
                'inferior':1,'last':1,'lead':1,'least':1,'less':1,'like':1,'match':1,'more':1,'most':1,'near':1,'nobodi':1,
                'none':1,'nonpareil':1,'onli':1,'outclass':1,'outdist':1,'outdo':1,'outfox':1,'outmatch':1,'outperform':1,
                'outsel':1,'outstrip':1,'outwit':1,'peerless':1,'prefer':1,'recommend':1,'rival':1,'same':1,'second':1,
                'similar':1,'superior':1,'thrice':1,'togeth':1,'top':1,'twice':1,'unlik':1,'unmatch':1,'unriv':1,
                'versus':1,'vs':1,'win':1}
for idx, item in enumerate(tagged_tuples):
    if stemmer.stem(item[0]) in keyword_dict:
        getSequence(tagged_tuples, idx, item[1], str(stemmer.stem(item[0])), window_size)

comparative_phrases=['number one','on par with','one of few','up against']
for phrase in comparative_phrases:
    if phrase in text:
        print("candidate based on \'"+phrase+"\'")