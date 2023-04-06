import json
import tqdm
import faiss
import numpy as np

def generate_submission(emb, dimension=172, topK=5):
    with open('./raw/item_share_preliminary_test_info.json', 'r') as f:
        competition_A = json.load(f)

    with open('./processed/userid2num.json', 'r') as load_f:
        userid2num = json.load(load_f)

    A_inviters_index = []
    for line in competition_A:
        A_inviters_index.append(int(userid2num[line['inviter_id']]))

    with open('./processed/usernum2id.json', 'r') as load_f:
        usernum2id = json.load(load_f)
    inviters_index=np.array(A_inviters_index)
    inviters_emb = emb[inviters_index]
    indexL2 =faiss.IndexFlatL2(dimension)
    indexL2.add(emb)
    distance, topKindex = indexL2.search(inviters_emb, topK)
    submission_A = []
    for i in tqdm(range(len(inviters_index))):
        topKlist = list(topKindex[i, :])
        candidate_voter_list = [usernum2id[str(top_voter_index)] for top_voter_index in topKlist]
        submission_A.append({'triple_id': str('%06d' % i), 'candidate_voter_list': candidate_voter_list})

    with open('submission_A.json', 'w') as f:
        json.dump(submission_A, f)

