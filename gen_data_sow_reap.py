import os
from data.data import CnnDmDataset

try:
    DATA_DIR = '/home/yashgupta/exp/cnndm/finished_files' ##os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class MatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])
        matched_arts = [art_sents[i] for i in extracts]
        return matched_arts, abs_sents[:len(extracts)]

for split in ['train']: #['val', 'train']:
	f = open('/home/yashgupta/exp/cnndm/finished_files/extract_'+split+'.tok', 'w')
	val = MatchDataset(split)
	for i in range(val._n_data):
		art_sents, abs_sents = val.__getitem__(i)
		n_sents = len(art_sents)
		for j in range(n_sents):
			f.write(art_sents[j]+'\n')
			f.write(abs_sents[j]+'\n')
		if i%100==0:
			print("Completed%", 100*i/val._n_data)
	f.close()

# print(val._n_data)
# print(val.__getitem__(1))