import torch
from ptbert import *
from small import *
from utils import *

if __name__ == '__main__':
    name = 'hotel'  # clothing, fruit, hotel, pda, shampoo
    import pickle
    from tqdm import tqdm

    x_len = 50
    b_size = 64

    alpha = 0.5  # portion of the original one-hot CE loss
    (x_tr, y_tr, t_tr), (x_de, y_de, t_de), (x_te, y_te, t_te), v_size, embs = load_data(name)

    x_de = x_de[1:2]
    y_de = y_de[1:2]
    t_de = t_de[1:2]
    # print(x_de)  # 向量
    # [[715, 27, 20965, 61, 36, 16, 10, 23, 2, 449, 494, 244, 2, 227, 48, 20966, 2, 227, 498, 20, 2, 38, 121, 22, 7696, 2, 1446, 6235, 2, 340, 10, 46, 2, 38, 36, 401, 93, 696, 4, 151, 104, 96, 128, 406, 2, 19, 122, 24, 14, 128, 4]]
    print(y_de)  # [0]
    # [1]
    # print(t_de)  # 文本
    # ['几次入住戚家山宾馆感觉都很好，总台接待热情，客房服务员工作到位，客房条件不错，就是卫生间没有换气扇，通风略差，宽带很方便，就是感觉速度有点慢。以后再来这里出差，还选择住在这里。']

    l_de = list(map(lambda x: min(len(x), x_len), x_de))
    x_de = sequence.pad_sequences(x_de, maxlen=x_len)
    print(x_de)  # sequence 处理后的
    # [[27 20965    61    36    16    10    23     2   449   494   244     2
    #   227    48 20966     2   227   498    20     2    38   121    22  7696
    #   2  1446  6235     2   340    10    46     2    38    36   401    93
    #   696     4   151   104    96   128   406     2    19   122    24    14
    #   128     4]]
    # print(l_de)  # 长度

    model = torch.load('data/cache/distill_model_{}'.format(name))
    model.eval()

    accu=[]
    with torch.no_grad():
        for i in range(0, len(x_de), b_size):
            bx = Variable(LTensor(x_de[i:i + b_size]))
            by = Variable(LTensor(y_de[i:i + b_size]))
            bl = Variable(LTensor(l_de[i:i + b_size]))
            _, py = torch.max(model(bx, bl)[1], 1)
            accu.append((py == by).float().mean().item())
    print(py)
    print(by)
