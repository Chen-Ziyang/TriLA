# coding:utf-8
import cv2
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def calculate_w(matrix):
    index = matrix
    Xs = np.zeros((index.shape[0], 6))
    H = np.zeros((6, 6))
    H[0][2], H[1][1], H[2][0] = 2.0, -1.0, 2.0
    id = 0
    for x, y in index:
        X = [pow(x, 2), x * y, pow(y, 2), x, y, 1]
        Xs[id] = X
        id += 1
    Xs = np.array(Xs.T)
    S = np.matmul(Xs, Xs.T)  # S=XX^T，λHW = SW, S^-1HW = 1/λW
    A = np.matmul(np.mat(S).I, H)  # AW=1/λW，求解A的特征向量，λ>0
    value, vector = np.linalg.eig(A)  # 计算特征值和特征向量

    value_index = (value > 0)  # 根据条件得到大于0的λ对应的特征向量，即为所求
    W = vector[:, value_index]
    return W


def ellipse_fitting(src, thresh=1.0):
    value_l = []
    dst = np.zeros((src.shape))
    index = np.argwhere(src >= thresh)
    W = calculate_w(index)
    '''
    for x, y in index:         # 计算误差
        X = np.array([pow(x, 2), x * y, pow(y, 2), x, y, 1])
        value = np.abs(np.matmul(W.T, X.T))
        value_l.append(value)
    avg = np.mean(value_l)
    std = np.std(value_l)

    for i in range(index.shape[0]):         # 除去噪声散点
        if np.abs(value_l[i]-avg) >= (2*std):
            value_l[i] = False
        else:
            value_l[i] = True

    index = np.compress(value_l, index, axis=0)
    W = calculate_w(index)      # 计算除去噪声后的椭圆方程
    '''
    # 因为我们需要的是椭圆曲线内部的区域，代入椭圆方程可能得到正值／负值，通过判断中心点的值来判断是正值还是负值
    center_x, center_y = np.mean(index[:][0]), np.mean(index[:][1])
    X = np.array([pow(center_x, 2), center_x * center_y, pow(center_y, 2), center_x, center_y, 1])
    center_value = np.matmul(W.T, X.T)
    if center_value > 0:
        insert = 1.0
    else:
        insert = 0.0

    index = np.argwhere(dst != None)
    for x, y in index:
        X = np.array([pow(x, 2), x * y, pow(y, 2), x, y, 1])
        value = np.matmul(W.T, X.T)
        if value <= 0:
            dst[x][y] = insert
        else:
            dst[x][y] = 1.0-insert
    return dst


def ifourier(img):
    # 傅立叶逆变换
    ishift = np.fft.ifftshift(img)#将频域从中间移动到左上角
    iimg = cv2.idft(ishift)#傅里叶逆变换库函数调用
    img_idf = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])#双通道结果转换为0到255的范围
    return img_idf

def fourier(img):
    # 傅立叶变换
    float32_img = np.float32(img)
    dft_img = cv2.dft(float32_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_img_ce = np.fft.fftshift(dft_img)#将低频从左上角移动到中间
    # img_df = 20 * np.log(cv2.magnitude(dft_img_ce[:, :, 0], dft_img_ce[:, :, 1]))
    return dft_img_ce

def FDA(content, style, mask):
    # FDA域适应算法，img必须是单通道。mask应该是中央低频区为1，高频区为0
    s_FDA = mask*style + (1-mask)*content
    iFDA = ifourier(s_FDA)
    iFDA /= np.max(iFDA)
    return iFDA

def MF(s_dft_img, mask):
    # mask掩膜处理频域图像，img必须是单通道
    s_FDA = mask*s_dft_img
    iFDA = ifourier(s_FDA)
    iFDA /= np.max(iFDA)
    return iFDA

def fda_generate(source, target, beta):
    H, W = source.shape[-2], source.shape[-1]
    mask = np.zeros((H, W, 2), np.uint8)
    mask[int(H // 2 - beta * H):int(H // 2 + beta * H), int(W // 2 - beta * W):int(W // 2 + beta * W)] = 1

    source_set = (source*255).cpu().numpy()
    target_set = (target*255).cpu().numpy()
    for b in range(target.shape[0]):
        s_r, s_g, s_b = source_set[b][0], source_set[b][1], source_set[b][2]
        dft_s_r = fourier(s_r)
        dft_s_g = fourier(s_g)
        dft_s_b = fourier(s_b)

        t_r, t_g, t_b = target_set[b][0], target_set[b][1], target_set[b][2]
        dft_t_r = fourier(t_r)
        dft_t_g = fourier(t_g)
        dft_t_b = fourier(t_b)

        t_b_FDA = torch.FloatTensor(FDA(dft_t_b, dft_s_b, mask)).unsqueeze(0)
        t_g_FDA = torch.FloatTensor(FDA(dft_t_g, dft_s_g, mask)).unsqueeze(0)
        t_r_FDA = torch.FloatTensor(FDA(dft_t_r, dft_s_r, mask)).unsqueeze(0)

        s_b_FDA = torch.FloatTensor(FDA(dft_s_b, dft_t_b, mask)).unsqueeze(0)
        s_g_FDA = torch.FloatTensor(FDA(dft_s_g, dft_t_g, mask)).unsqueeze(0)
        s_r_FDA = torch.FloatTensor(FDA(dft_s_r, dft_t_r, mask)).unsqueeze(0)

        data_tFDA = torch.cat((t_r_FDA, t_g_FDA, t_b_FDA), dim=0).unsqueeze(0)
        data_sFDA = torch.cat((s_r_FDA, s_g_FDA, s_b_FDA), dim=0).unsqueeze(0)
        try:
            CT_SS = torch.cat((CT_SS, data_tFDA), dim=0)
            CS_ST = torch.cat((CS_ST, data_sFDA), dim=0)
        except NameError:
            CT_SS = data_tFDA
            CS_ST = data_sFDA

    return CT_SS, CS_ST


def style_remove(dst, beta=0.005):
    # input tensor = [b,s,h,w]
    H, W = dst.shape[-2], dst.shape[-1]
    mask = np.ones((H, W, 2), np.uint8)
    mask[int(H // 2 - beta * H):int(H // 2 + beta * H), int(W // 2 - beta * W):int(W // 2 + beta * W)] = 0

    dst_g = (dst[:, 1, :, :]*255).cpu().numpy()
    for b in range(dst_g.shape[0]):
        dft_g = fourier(dst_g[b])
        dst_g[b] = MF(dft_g, mask)
    return torch.FloatTensor(dst_g).unsqueeze(1)


def clahe(data):
    data = np.array(data)
    r, g, b = cv2.split(data)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    r = clahe.apply(r)
    g = clahe.apply(g)
    data = cv2.merge((r, g, b))
    return data


def he(data):
    data = np.array(data)
    r, g, b = cv2.split(data)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)
    data = cv2.merge((r, g, b))
    return data


def gamma(data, g):
    data = np.array(data)
    data = data / 255.0
    data = np.power(data, g) * 255.0
    return data.astype(np.uint8)


def LightCorrect(data, g=6):
    data = clahe(data)
    data = gamma(data, g=g)
    return data


def get_maxarea(img_g, thresh=150, maxium=255, mode=cv2.THRESH_BINARY):
    '''
    提取最大连通区域
    '''
    area = []
    _, img_g = cv2.threshold(img_g, thresh, maxium, mode)
    try:
        img, contours, hierarch = cv2.findContours(img_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, hierarch = cv2.findContours(img_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(img_g, [contours[k]], 0)
    return img_g


def truncated_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def GDL_loss(pred, label):
    smooth = 1.
    weight = 1. - torch.sum(label, dim=(0, 2, 3)) / torch.sum(label)
    # weight = 1./(torch.sum(label, dim=(0, 2, 3))**2 + smooth); smooth = 0
    intersection = pred * label
    intersection = weight * torch.sum(intersection, dim=(0, 2, 3))
    intersection = torch.sum(intersection)
    union = pred + label
    union = weight * torch.sum(union, dim=(0, 2, 3))
    union = torch.sum(union)
    score = 1. - (2. * (intersection + smooth) / (union + smooth))
    return score


def dice_coeff(pred, label):
    smooth = 1.
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)
    intersection = (m1 * m2).sum()
    score = 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
    return score


def jaccard_loss(pred, label):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m1 = torch.abs(1 - m1)
    m2 = label.view(num, -1)  # Flatten
    m2 = torch.abs(1 - m2)
    score = 1 - ((torch.min(m1, m2).sum() + smooth) / (torch.max(m1, m2).sum() + smooth))
    return score


def p2p_loss(pred, label):
    # Mean Absolute Error (MAE)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = label.view(num, -1)  # Flatten
    score = torch.mean(torch.abs(m2 - m1))
    return score


def bce_loss(pred, label):
    score = torch.nn.BCELoss()(pred, label)
    return score


def mse_loss(pred, label):
    score = torch.nn.MSELoss()(pred, label)
    return score


def tversky_index(pred, label, alpha=0.7):
    smooth = 1
    num = pred.size(0)
    y_true_pos = label.view(num, -1)
    y_pred_pos = pred.view(num, -1)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1. - y_pred_pos))
    false_pos = torch.sum((1. - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                1 - alpha) * false_pos + smooth)


def focal_tversky(pred, label, gamma=2, alpha=0.7):
    pt_1 = tversky_index(pred, label, alpha)
    score = (1. - pt_1).pow(gamma)
    return score


class VAE_loss(nn.Module):
    def __init__(self, h1=0.1, h2=0.1, mode='mean'):
        super(VAE_loss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.mode = mode

    def loss_vae(self, x, recon_x, mu, logvar):
        b, c, h, w = recon_x.shape
        assert recon_x.shape == x.shape
        rec_flat = recon_x.view(b, -1)
        x_flat = x.view(b, -1)
        # ce = F.binary_cross_entropy(rec_flat, x_flat, reduction=self.mode)
        rec_loss = torch.nn.MSELoss()(rec_flat, x_flat)
        if self.mode == 'mean':
            # KLD = torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)
            KLD = torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / (b*c*h*w)
        elif self.mode == 'sum':
            KLD = torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        else:
            assert 'Calculate mode is warning: ' + self.mode
        return rec_loss * self.h1 + KLD * self.h2

    def forward(self, x, vae_out, mu, logvar):
        vae_score = self.loss_vae(x, vae_out, mu, logvar)
        score = vae_score
        return score


def ent_loss(output, ita=2.0):
    P = torch.sigmoid(output)
    logP = torch.log2(P+1e-6)
    PlogP = P * logP
    ent = -1.0 * PlogP.sum(dim=1)
    ent = ent ** 2.0 + 1e-8
    ent = ent ** ita
    return ent.mean()


class Seg_loss(nn.Module):
    def __init__(self, lossmap):
        super(Seg_loss, self).__init__()
        self.ecc_loss = Eccentricity_loss(sigmoid=False)
        self.tasks = lossmap
        self.tasks = tuple(sorted(self.tasks))
        self.lossmap = {
            'dice': eval('dice_coeff'),
            'bce': eval('bce_loss'),
            'focal': eval('focal_tversky'),
            'L2loss': eval('mse_loss'),
            'jaccard': eval('jaccard_loss'),
            'p2p': eval('p2p_loss'),
            'gdl': eval('GDL_loss'),
            'ecc': eval('self.ecc_loss'),
        }

    def forward(self, logit_pred, label):
        pred = torch.sigmoid(logit_pred)

        score = 0
        for task in self.tasks:
            if task in self.lossmap.keys():
                score += self.lossmap[task](pred=pred, label=label)
        return score


class Content_loss(nn.Module):
    def __init__(self):
        super(Content_loss, self).__init__()

    def forward(self, Content_1, Content_2):
        loss = 0.5 * torch.sum(torch.square(Content_1 - Content_2))
        # loss = torch.sum(-torch.norm(Content_1, 'nuc', dim=[2, 3]) + torch.norm(Content_2, 'nuc', dim=[2, 3])) / \
        #                 (Content_1.shape[0] * Content_2.shape[1])
        return loss


class Style_loss(nn.Module):
    def __init__(self):
        super(Style_loss, self).__init__()

    def gram(self, matrix):
        features = matrix.view(matrix.shape[0], matrix.shape[1], -1)
        gram_matrix = torch.matmul(features, features.permute(0, 2, 1).contiguous())
        return gram_matrix

    def forward(self, Style_1, Style_2, prior):
        channels = Style_1.shape[1]
        size = Style_1.shape[-1] * Style_1.shape[-2]
        gram_1 = self.gram(Style_1)
        gram_2 = self.gram(Style_2)
        if prior == 'same':
            loss = torch.sum(torch.square(gram_1 - gram_2)) / (4. * (channels ** 2) * (size ** 2))
        if prior == 'diff':
            loss = -torch.sum(torch.square(gram_1 - gram_2)) / (4. * (channels ** 2) * (size ** 2))
        return loss


class Eccentricity_loss(nn.Module):
    def __init__(self, sigmoid=False):
        super(Eccentricity_loss, self).__init__()
        self.sigmoid = sigmoid

    def calculate_e(self, vector):
        epsilon = 1e-5
        N, C, H, W = vector.shape
        v = vector.view(N, -1)
        y = torch.arange(1, H+1, 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(N, C, 1, H)\
            .to(vector.device).view(N, -1).float() / H
        x = torch.arange(1, H+1, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N, C, H, 1)\
            .to(vector.device).view(N, -1).float() / H

        A = torch.sum(v * torch.pow(y, 2))
        B = torch.sum(v * torch.pow(x, 2))
        H = torch.sum(v * x * y)
        p = 2 / ((A + B) - torch.sqrt(torch.pow(A - B, 2) + 4 * torch.pow(H, 2)) + epsilon)
        q = 2 / ((A + B) + torch.sqrt(torch.pow(A - B, 2) + 4 * torch.pow(H, 2)) + epsilon)
        e = p / q
        return e

    def forward(self, logit_pred, label):
        if self.sigmoid:
            pred = torch.nn.Sigmoid()(logit_pred)
        else:
            pred = logit_pred
        pred_ecc = self.calculate_e(pred)
        label_ecc = self.calculate_e(label)
        loss = 0.5 * torch.sum(torch.square(pred_ecc - label_ecc))
        return loss


def entropy_loss_func(v, sigmoid=False):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    if sigmoid:
        v = torch.nn.Sigmoid()(v)
    n, c, h, w = v.size()
    loss = -torch.sum(torch.mul(v, torch.log2(v + 1e-20))) / (n * h * w * np.log2(c))
    return loss


def Entropy(input_, sigmoid=True):
    if sigmoid:
        input_ = torch.nn.Sigmoid()(input_)
    C = torch.tensor(input_.size(1))
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy) / torch.log(C)
    return entropy


def IM_loss(logits):
    C = torch.tensor(logits.size(1))
    epsilon = 1e-5
    sigmoid_out = torch.nn.Sigmoid()(logits)
    entropy_loss = Entropy(sigmoid_out, sigmoid=False)
    msigmoid = sigmoid_out.mean(dim=0)
    gentropy_loss = torch.sum(-msigmoid * torch.log(msigmoid + epsilon)) / torch.log(C)
    entropy_loss -= gentropy_loss
    return entropy_loss


def Seg_consistancy_loss(logits):
    sigmoid_out = torch.nn.Sigmoid()(logits)
    loss = 0
    for i in range(1, sigmoid_out.shape[0]):
        loss += torch.nn.MSELoss()(sigmoid_out[i], sigmoid_out[0])
    return loss


def early_stop(now_loss, loss_l, a):
    '''
    GL早停法
    :param now_loss: 当前epoch验证集损失
    :param loss_l: 总的验证集损失
    :param a: GL参数
    :return: bool，是否需要早停
    '''
    if 100 * (now_loss / min(loss_l) - 1) >= a:
        return True
    else:
        return False


class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - self.last_epoch / self.epochs), self.gamma)]

