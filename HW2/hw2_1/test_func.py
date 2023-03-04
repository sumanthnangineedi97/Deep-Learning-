import sys
import torch
import pickle
import json
from compute_cuda_final_sumanth import testing_data, test, MODELS, rnn_encoder, rnn_decoder, attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU

fpath = './data/testing_data/feat'
model = torch.load('./SavedModel/model0.h5', map_location=lambda storage, loc: storage)
dataset = testing_data('{}'.format(sys.argv[1]))
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)
#model = model.cuda()
sentences = test(test_loader, model, i2w)
with open(sys.argv[2], 'w') as f:
    for id, s in sentences:
        f.write('{},{}\n'.format(id, s))


test = json.load(open('./data/testing_label.json'))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption       
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))
