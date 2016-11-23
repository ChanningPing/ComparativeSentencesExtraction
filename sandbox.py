import nltk
text = "However,  wow i superior"
tagged_tuples = nltk.pos_tag(nltk.word_tokenize(text.lower()))
print tagged_tuples
#if the sentence contains standard comparative, keep it
for item in tagged_tuples:
    if item[1] == 'JJR' or item[1] == 'RBR' or item[1] == 'JJS' or item[1] == 'RBS':
        print item, 'it is comparative candidate'
        break

#if the sentence contains as {} as,keep it
indices = [i for i, item in enumerate(tagged_tuples) if item[0] == 'as' and item[1] == 'RB']
for index in indices:
    if (tagged_tuples[index+1][1]=='JJ' or tagged_tuples[index+1][1]=='RB') and tagged_tuples[index+2][0] == 'as' and tagged_tuples[index+2][1] == 'IN':
        print 'it is comparative candidate'
        break

#if the sentence contains certain keyword, keep it
keyword_dict={'beat':1,'inferior':1,'outstrip':1,'both':1,'on par with':1,'Choice':1,'choose':1,'prefer':1,'recommend':1,'outperform':1,'superior':1,'all':1,'up against':1,'less':1,'favor':1,'defeat':1,'twice':1,'thrice':1,'half':1,'same':1,'either':1,'Compete':1,'number one':1,'one of few':1,'more':1,'like':1,'behind':1,'similar':1,'identical':1,'Versus':1,'first':1,'outdistance':1,'before':1,'double':1,'outsell':1,'nobody':1,'Vs':1,'last':1,'after':1,'thrice':1,'improve':1,'equal':1,'equivalent':1,'together':1,'altogether':1,'Alternate':1,'only':1,'outmatch':1,'ahead':1,'fraction':1,'outdo':1,'match':1,'unmatched':1,'peerless':1,'differ':1,'one of few':1,'outwit':1,'rival':1,'alternate':1,'Compare':1,'top':1,'exceed':1,'lead':1,'win':1,'outstrip':1,'none':1,'near':1,'unrivaled':1,'dominate':1,'second':1,'nonpareil':1,'advantage':1,'unlike':1,'least':1,'outclass':1,'outfox':1,'outdistance':1,'most':1}
for item in tagged_tuples:
    if item[0] in keyword_dict:
        print item, 'it is comparative candidate'
        break