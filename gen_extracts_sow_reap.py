import os
from data.data import CnnDmDataset
print("Which dataset: cnndm/newsroom?")
dataset = str(input())
assert(dataset=='cnndm' or dataset=='newsroom')

try:
    DATA_DIR = '/home/yashgupta/exp/'+dataset+'/finished_files' ##os.environ['DATA']
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
        # print(js_data)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])
        matched_arts = [art_sents[i] for i in extracts]
        return matched_arts, abs_sents[:len(extracts)]

for split in ['val', 'train']:
	# f1 = open('/home/yashgupta/exp/'+dataset+'/finished_files/hf_extracts/'+split+'.source', 'w')
	# f2 = open('/home/yashgupta/exp/'+dataset+'/finished_files/hf_extracts/'+split+'.target', 'w')
	val = MatchDataset(split)
	m = 0
	m2 = 0
	for i in range(val._n_data):
		art_sents, abs_sents = val.__getitem__(i)
		n_sents = len(art_sents)
		for j in range(n_sents):
			m = max(m, len(art_sents[j].split(" ")))
			m2 = max(m2, len(abs_sents[j].split(" ")))
			# f1.write(art_sents[j]+'\n')
			# f2.write(abs_sents[j]+'\n')
		if i%10000==0:
			print("Completed%", 100*i/val._n_data)
	print(m, m2)
	# f1.close()
	# f2.close()

# print(val._n_data)
# print(val.__getitem__(1))
