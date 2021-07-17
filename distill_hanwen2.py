# -*- coding: utf-8 -*-
# 蒸馏出来双层单向LSTM
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import BertTokenizer
from ptbert import *
from small import *
from utils import *

FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


class Teacher(object):
    def __init__(self, name, bert_model='bert-base-chinese', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        self.model = torch.load('./data/cache/model_{}'.format(name))
        self.model.eval()

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.softmax(logits, dim=1).detach().cpu().numpy()


if __name__ == '__main__':

    name = 'fruit'  # clothing, fruit, hotel, pda, shampoo
    print(name)

    teacher = Teacher(name)
    print('还不错！这个价位算是物有所值了！', teacher.predict('还不错！这个价位算是物有所值了！'))

    import pickle
    from tqdm import tqdm

    maxAcc = 0
    x_len = 50
    b_size = 64
    lr = 0.002
    epochs = 10
    # epochs = 10000

    alpha = 0.5  # portion of the original one-hot CE loss
    use_aug = False  # whether to use data augmentation
    n_iter = 5
    p_mask = 0.1
    p_ng = 0.25
    ngram_range = (3, 6)
    teach_on_dev = True
    if not use_aug:
        (x_tr, y_tr, t_tr), (x_de, y_de, t_de), (x_te, y_te, t_te), v_size, embs = load_data(name)
    else:
        # will introduce randomness, thus can't be loaded below
        (x_tr, y_tr, t_tr), (x_de, y_de, t_de), (x_te, y_te, t_te), v_size, embs = \
            load_data_aug(name, n_iter, p_mask, p_ng, ngram_range)

    model = RNN_hanwen_2layer(v_size, 256, 256, 2)
    print(model)
    print("hanwen的两层模型")
    print(len(x_tr), len(x_de), len(x_te))
    l_tr = list(map(lambda x: min(len(x), x_len), x_tr))
    l_de = list(map(lambda x: min(len(x), x_len), x_de))
    l_te = list(map(lambda x: min(len(x), x_len), x_te))

    x_tr = sequence.pad_sequences(x_tr, maxlen=x_len)
    x_de = sequence.pad_sequences(x_de, maxlen=x_len)
    x_te = sequence.pad_sequences(x_te, maxlen=x_len)
    with torch.no_grad():
        t_tr = np.vstack([teacher.predict(text) for text in tqdm(t_tr)])
        t_de = np.vstack([teacher.predict(text) for text in tqdm(t_de)])
    with open('./data/cache/t_tr', 'wb') as fout:
        pickle.dump(t_tr, fout)
    with open('./data/cache/t_de', 'wb') as fout:
        pickle.dump(t_de, fout)
    # with open('./data/cache/t_tr', 'rb') as fin:
    #     t_tr = pickle.load(fin)
    # with open('./data/cache/t_de', 'rb') as fin:
    #     t_de = pickle.load(fin)

    # model = RNN(v_size, 256, 256, 2)
    model = RNN_hanwen_2layer(v_size, 256, 256, 2)

    # model = CNN(v_size,256,128,2)

    # 注意！这里是设备的转换
    if USE_CUDA: model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.NLLLoss()
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        losses = []
        accu = []
        model.train()
        for i in range(0, len(x_tr), b_size):
            model.zero_grad()
            bx = Variable(LTensor(x_tr[i:i + b_size]))
            by = Variable(LTensor(y_tr[i:i + b_size]))
            bl = Variable(LTensor(l_tr[i:i + b_size]))
            bt = Variable(FTensor(t_tr[i:i + b_size]))
            py1, py2 = model(bx, bl)
            loss = alpha * ce_loss(py2, by) + (1 - alpha) * mse_loss(py1, bt)  # in paper, only mse is used
            # loss = mse_loss(py1, bt) # 我自己的加的，就是为了测试paper
            loss.backward()
            opt.step()
            losses.append(loss.item())
        for i in range(0, len(x_de), b_size):
            model.zero_grad()
            bx = Variable(LTensor(x_de[i:i + b_size]))
            bl = Variable(LTensor(l_de[i:i + b_size]))
            bt = Variable(FTensor(t_de[i:i + b_size]))
            py1, py2 = model(bx, bl)
            loss = mse_loss(py1, bt)
            if teach_on_dev:
                loss.backward()
                opt.step()  # train only with teacher on dev set
            losses.append(loss.item())

        model.eval()
        time0 = time.time()
        with torch.no_grad():
        #     for i in range(0, len(x_de), b_size):
        #         bx = Variable(LTensor(x_de[i:i + b_size]))
        #         by = Variable(LTensor(y_de[i:i + b_size]))
        #         bl = Variable(LTensor(l_de[i:i + b_size]))
        #         _, py = torch.max(model(bx, bl)[1], 1)
        #         accu.append((py == by).float().mean().item())
            for i in range(0, len(x_te), b_size):
                bx = Variable(LTensor(x_te[i:i + b_size]))
                by = Variable(LTensor(y_te[i:i + b_size]))
                bl = Variable(LTensor(l_te[i:i + b_size]))
                _, py = torch.max(model(bx, bl)[1], 1)
                accu.append((py == by).float().mean().item())
        predict_time = time.time() - time0
        maxAcc = max(maxAcc, np.mean(accu))
        print(epoch, np.mean(losses), np.mean(accu), "maxAcc", maxAcc, "predictTime", predict_time, "size", len(x_te),
              "average", predict_time / len(x_te))

    torch.save(model, 'data/cache/hanwen2layer_distill_model_{}'.format(name))
    print('data/cache/hanwen2layer_distill_model_{}'.format(name))
