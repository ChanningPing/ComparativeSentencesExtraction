
from __future__ import print_function
import sys
from collections import defaultdict

sequence=[
    [1,3,6,7,8],
    [1,3,4,9],
    [2,3,4,5,6],
    [1,3,5,9]
]
sequence = [
['JJ', 'NNS', 'NN', 'RB', 'RB_well', 'IN', 'IN', 'JJ', 'NNS'],
['NN', 'VBG', 'NNS', 'RB', 'RB_well', 'IN', 'IN', 'NN'],
['IN', 'JJ', 'NN', 'VBG_match', 'NNS', 'RB', 'RB'],
['VBN', 'TO', 'VB', 'JJ_advantag'],
['NNS', 'VBP', 'NN', 'RB', 'RB_well', 'IN', 'IN', 'NN'],
['VBZ', 'VBN', 'RB', 'VBN_compar', 'IN', 'DT', 'JJ'],
['DT', 'JJ', 'NN', 'VBG_match', 'NNS', 'VBN', 'IN'],
['PRP', 'VBP', 'DT', 'JJS_best', 'NN', 'VBN', 'IN'],
['PRP', 'VBP', 'DT', 'JJS_best', 'NN', 'VBN', 'IN'],
['IN', 'JJ', 'RB', 'JJR_smaller', 'NNS', 'VBP', 'VBN'],
['RB', 'VBP', 'DT', 'NNS_advantag', 'IN', 'NN', 'IN'],
['VBN', 'VBN', 'IN', 'DT_all', 'CD', 'NNS', 'NNS'],
['NNS', 'NNS', 'CC', 'DT_both', 'NN', 'VBG', 'NNS'],
['IN', 'JJ', 'NNS', 'VBP_domin', 'DT', 'JJ', 'NN'],
['RB', 'VB', 'JJ', 'NN_advantag', 'IN', 'DT', 'NN'],
['VBP', 'IN', 'RB', 'JJR_better', 'NNS', 'MD', 'VB'],
['VBN', 'IN', 'DT', 'JJ_improv', 'NN', 'IN', 'NN'],
['VBN', 'NN', 'VBD', 'JJR_better', 'IN', 'DT', 'JJS'],
['JJR', 'IN', 'DT', 'JJS_best', 'JJ', 'NN'],
['IN', 'DT_all', 'NNS', 'DT', 'RB'],
['CD', 'VBD', 'RB', 'JJ_compar', 'TO', 'VB', 'IN'],
['IN', 'JJR_larger', 'NNS', 'NNS', 'CD'],
['NN', 'VBZ', 'DT', 'NN_advantag', 'IN', 'NNS', 'MD'],
['WDT', 'VBP', 'NNS', 'NNS_advantag', 'IN', 'NNS', 'IN']
]

labels=['YES','NO','YES','NO','YES','YES','NO','NO','NO','YES','NO','YES','YES','YES','YES','NO','NO','YES','YES','YES','YES','NO','NO','NO']
TAU=0.6 #fraction of total frequency of an item to account for the min_support for this item

min_confidence=0.6 # proportion of instances in D that covers the rule also satisfies the rule

results = []

def mine_rec(patt, mdb):
    numYES=0
    numNO=0
    for coordinate in mdb:
        if labels[coordinate[0]]=='YES':
            numYES += 1
        else:
            numNO += 1

    results.append((patt, len(mdb),numYES,numNO)) # the pattern, the frequency of the pattern, the number of YES labels, the number of NO labels

    occurs = defaultdict(list)
    for (i, startpos) in mdb:
        seq = sequence[i]
        for j in xrange(startpos, len(seq)):
            l = occurs[seq[j]]
            if len(l) == 0 or l[-1][0] != i:
                l.append((i, j + 1))

    for (c, newmdb) in occurs.iteritems():
        #if len(newmdb) >= minsup:
        mine_rec(patt + [c], newmdb)

mine_rec([], [(i, 0) for i in xrange(len(sequence))])

for result in results:
    sup = result[2]
    min_sup = result[1]*TAU
    confidence = result[2]/result[1]
    if sup >= min_sup and confidence >= min_confidence:
        print("sup="+str(sup)+",min_sup="+str(min_sup)+", confidence="+str(confidence))
        print(result)

