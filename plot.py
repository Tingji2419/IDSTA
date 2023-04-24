import torch
import numpy as np
import matplotlib.pyplot as plt

def soft_label(x, toa, margin, tpt):
    '''
    y = {
        softmax(- margin / tpt) / (toa - margin) * x ,                                            0 < x <= toa - margin
        softmax[(x - toa) / tpt],                                                      toa - margin < x <  toa + margin
        [[softmax(margin / tpt) ] * (x - 60) - x + toa + margin] / (toa + margin - 60),  toa + margin <= x <= 1
    }
    '''
    x = torch.tensor(x).cuda()
    # toa = torch.tensor(toa).cuda()
    margin = torch.tensor(margin).cuda()
    tpt = torch.tensor(tpt).cuda()
    x1 = toa - margin
    x2 = toa + margin
    
    if x <= x1:
        y = x * torch.sigmoid(- margin / tpt) / x1
    elif x >= x2:
        y = (torch.sigmoid(margin / tpt) * (x - 60) - x + x2) / (x2 - 60)
    else:
        y = torch.sigmoid((x - toa) / tpt)
    return y


if __name__ == "__main__":
    print()
    x = np.arange(0, 61, 1)
    y1 = [soft_label(i, toa=40, margin=15, tpt=3).cpu() for i in x]
    y2 = [soft_label(i, toa=40, margin=15, tpt=6).cpu() for i in x]
    y3 = [soft_label(i, toa=40, margin=15, tpt=9).cpu() for i in x]
    plt.clf()
    plt.plot(x, y1, label='temperature=3')
    plt.plot(x, y2, label='temperature=6')
    plt.plot(x, y3, label='temperature=9')
    yt = [0]*40 + [1]*21
    plt.plot(x, yt, label='Hard Label')
    plt.legend()
    plt.show()
    plt.savefig('temp.png', dpi=1200)
