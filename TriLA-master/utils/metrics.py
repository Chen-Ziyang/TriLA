import cv2
import numpy as np
from medpy.metric import binary


smooth = 1.0


def polar_to_cart(src, mode=cv2.WARP_INVERSE_MAP):
    src = np.array(src)
    value = np.sqrt(((src.shape[0] / 2.0) ** 2.0) + ((src.shape[1] / 2.0) ** 2.0))
    dst = cv2.linearPolar(src, (src.shape[0] / 2, src.shape[1] / 2), value, mode)
    return np.float32(dst)


def data_process(pred, label, threshold=0.5):
    '''
    pred = np.array(pred*255).astype(np.uint8)
    label = np.array(label*255).astype(np.uint8)

    for N in range(pred.shape[0]):
        for C in range(pred.shape[1]):
            _, pred[N][C] = cv2.threshold(pred[N][C], 0, 255, cv2.THRESH_OTSU)
            _, label[N][C] = cv2.threshold(label[N][C], 0, 255, cv2.THRESH_OTSU)
    pred, label = pred // 255, label // 255
    '''
    pred = np.array(pred)
    label = np.array(label)

    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    return pred.astype(np.uint8), label.astype(np.uint8)


def dice_metric(pred, label):
    batch_size = pred.shape[0]
    disc_dices, cup_dices = [], []

    for batch in range(batch_size):
        disc_intersection = (pred[batch][0] * label[batch][0]).sum()
        disc_dice = (2 * disc_intersection + smooth) / (pred[batch][0].sum() + label[batch][0].sum() + smooth)
        cup_intersection = (pred[batch][-1] * label[batch][-1]).sum()
        cup_dice = (2 * cup_intersection + smooth) / (pred[batch][-1].sum() + label[batch][-1].sum() + smooth)

        disc_dices.append(disc_dice)
        cup_dices.append(cup_dice)

    return np.mean(disc_dices), np.mean(cup_dices)


def iou_metric(pred, label):
    batch_size = pred.shape[0]
    disc_ious, cup_ious = [], []

    for batch in range(batch_size):
        disc_intersection = (pred[batch][0] * label[batch][0]).sum()
        disc_iou = (disc_intersection + smooth) / (pred[batch][0].sum() + label[batch][0].sum() - disc_intersection + smooth)
        cup_intersection = (pred[batch][-1] * label[batch][-1]).sum()
        cup_iou = (cup_intersection + smooth) / (pred[batch][-1].sum() + label[batch][-1].sum() - cup_intersection + smooth)

        disc_ious.append(disc_iou)
        cup_ious.append(cup_iou)

    return np.mean(disc_ious), np.mean(cup_ious)


def confusion_matrix(pred, label):
    TP = np.logical_and(pred == 1, label == 1).sum(axis=-1).sum(axis=-1)
    FP = np.logical_and(pred == 0, label == 1).sum(axis=-1).sum(axis=-1)
    TN = np.logical_and(pred == 0, label == 0).sum(axis=-1).sum(axis=-1)
    FN = np.logical_and(pred == 1, label == 0).sum(axis=-1).sum(axis=-1)
    return TP, FP, TN, FN


def calculate_cdr(img):
    cdr_l = []
    for b in range(img.shape[0]):
        tr_disc = img[b, 0, :, :]
        tr_cup = img[b, -1, :, :]
        disc = np.where(tr_disc == 1)[0]
        cup = np.where(tr_cup == 1)[0]
        if disc.size <= 1 or cup.size <= 1:
            cdr_l.append(np.nan)
        else:
            disc_d = np.max(disc) - np.min(disc)
            cup_d = np.max(cup) - np.min(cup)
            cdr_l.append(cup_d / disc_d)
    return np.array(cdr_l)


def hd95_metric(pred, label):
    disc_hd95, cup_hd95 = 0, 0
    for i in range(pred.shape[0]):
        try:
            disc_hd95 += binary.hd95(pred[i][0], label[i][0])
        except:
            disc_hd95 += 100.
        try:
            cup_hd95 += binary.hd95(pred[i][-1], label[i][-1])
        except:
            cup_hd95 += 100.
    disc_hd95, cup_hd95 = disc_hd95/pred.shape[0], cup_hd95/pred.shape[0]
    return disc_hd95, cup_hd95


def calculate_metrics(pred, label):
    # 0.5 for scenario: Magrabia (Source) -> Base1 (Target), and 0.25 for others
    pred, label = data_process(pred, label, threshold=0.25)

    CDR_Pred = calculate_cdr(pred)
    CDR_Label = calculate_cdr(label)
    # print(CDR_Label)
    MAE_CDR = np.mean(np.abs(CDR_Label - CDR_Pred))

    disc_dice, cup_dice = dice_metric(pred, label)
    disc_hd95, cup_hd95 = hd95_metric(pred, label)

    TP, FP, TN, FN = confusion_matrix(pred, label)
    disc_acc = ((TN[:, 0].sum() + TP[:, 0].sum() + smooth) / (TN[:, 0].sum() + TP[:, 0].sum() + FP[:, 0].sum() + FN[:, 0].sum() + smooth))
    disc_miou = 0.5 * (((TP[:, 0].sum() + smooth) / (TP[:, 0].sum() + FN[:, 0].sum() + FP[:, 0].sum() + smooth)) +
                       ((TN[:, 0].sum() + smooth) / (TN[:, 0].sum() + FN[:, 0].sum() + FP[:, 0].sum() + smooth)))

    cup_acc = ((TN[:, -1].sum() + TP[:, -1].sum() + smooth) / (TN[:, -1].sum() + TP[:, -1].sum() + FP[:, -1].sum() + FN[:, -1].sum() + smooth))
    cup_miou = 0.5 * (((TP[:, -1].sum() + smooth) / (TP[:, -1].sum() + FN[:, -1].sum() + FP[:, -1].sum() + smooth)) +
                      ((TN[:, -1].sum() + smooth) / (TN[:, -1].sum() + FN[:, -1].sum() + FP[:, -1].sum() + smooth)))

    return [disc_acc, disc_hd95, disc_dice, disc_miou, cup_acc, cup_hd95, cup_dice, cup_miou, MAE_CDR]
