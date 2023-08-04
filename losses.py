import torch
import torch.nn.functional as F
from dexi_utils import *


def hed_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.1).float()).float()
    num_negative = torch.sum((mask <= 0.).float()).float()

    mask[mask > 0.1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    # mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), targets.float())

    return l_weight*torch.sum(cost)


def Dice_Loss(inputs, targets, l_weight=1.0):
    # targets = targets.long()
    smooth = 1
    inputs = torch.sigmoid(inputs)  # F.sigmoid(inputs)

    input_flat = inputs.view(-1)
    target_flat = targets.view(-1)

    intersecion = input_flat * target_flat
    unionsection = input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth
    loss = unionsection / (2 * intersecion.sum() + smooth)

    loss = loss.sum()
    return l_weight * loss


def Dice_Loss_consistency(preds1, preds2, l_weight=1.0):
    # targets = targets.long()
    smooth = 1
    preds1 = torch.sigmoid(preds1)
    preds2 = torch.sigmoid(preds2)

    preds1_flat = preds1.view(-1)
    preds2_flat = preds2.view(-1)

    intersecion = preds1_flat * preds2_flat
    unionsection = preds1_flat.pow(2).sum() + preds2_flat.pow(2).sum() + smooth
    loss = unionsection / (2 * intersecion.sum() + smooth)
    loss = loss.sum()
    return l_weight * loss


def CE_loss_consistency(preds1, preds2, l_weight=1.0):
    preds2 = preds2.detach()
    preds1 = torch.sigmoid(preds1)
    preds2 = torch.sigmoid(preds2)
    cost = torch.nn.BCELoss(reduction='sum')(preds1, preds2)
    # print(cost)
    return l_weight * cost


def MSE_loss(preds1, preds2, l_weight=1.0):
    # inputs = torch.sigmoid(inputs)  # inputs.shape: torch.Size([1, 1, 512, 512])
    # targets = targets.long()  # targets.shape: torch.Size([1, 1, 512, 512])0
    preds1 = torch.sigmoid(preds1)
    preds2 = torch.sigmoid(preds2)
    cost = torch.nn.MSELoss(reduction='sum')(preds1, preds2)
    # print(cost)
    return l_weight * cost


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs, targets.float())
    return l_weight*cost


# def CE_robust_loss(prediction, labelf, l_weight=1.1):
#     label = labelf.long()        # targets.shape: torch.Size([B, 1, 512, 512])
#     mask = labelf.clone()
#     num_positive = torch.sum(label == 1).float()
#     num_negative = torch.sum(label == 0).float()
#
#     mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
#     mask[label == 0] = 1.1 * num_positive / (num_positive + num_negative)
#     mask[label == 2] = 0
#     cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='sum')
#     return l_weight * cost


def CE_loss(inputs, targets, l_weight=1.1, lamda=1.1):
    targets = targets.long()        # targets.shape: torch.Size([B, 1, 512, 512])
    mask = targets.float()  # mask.shape: torch.Size([B, 1, 512, 512])
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1
    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = lamda * num_positive / (num_positive + num_negative)

    inputs = torch.sigmoid(inputs)  # inputs.shape: torch.Size([B, 1, 512, 512])
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs, targets.float())
    # print(cost)
    return l_weight * cost


def CE_loss_reduction_none(inputs, targets, confidence, l_weight=1.0):
    confidence = confidence.detach()
    targets = targets.long()        # targets.shape: torch.Size([B, 1, H, W])
    mask = targets.float()          # mask.shape: torch.Size([B, 1, H, W])
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1
    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    mask = mask * confidence

    inputs = torch.sigmoid(inputs)      # inputs.shape: torch.Size([B, 1, H, W])
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # print(cost.shape)
    return l_weight*cost


def CE_loss_with_confidence(inputs, targets, confidence, l_weight=1.0):
    confidence = confidence.detach()
    targets = targets.long()        # targets.shape: torch.Size([B, 1, H, W])
    mask = targets.float()          # mask.shape: torch.Size([B, 1, H, W])
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1
    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    mask = mask * confidence

    inputs = torch.sigmoid(inputs)      # inputs.shape: torch.Size([B, 1, H, W])
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs, targets.float())
    return l_weight*cost


def SCE_loss(inputs, targets, l_weight=1.0, alpha=0.1, beta=1.0):  # maybe wrong
    # symmetric cross entropy loss
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1
    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) # 0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]

    inputs = torch.sigmoid(inputs)
    ce = torch.nn.BCELoss(mask, reduction='sum')(inputs, targets.float())

    inputs_d = inputs.detach()
    rce = torch.nn.BCELoss(mask, reduction='sum')(targets.float(), inputs_d)
    loss = l_weight*(alpha * ce + beta * rce)
    return loss





def bdcn_lossORI(inputs, targets, l_weigts=1.1,cuda=False):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    # print(cuda)
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * 1.1 / valid  # balance = 1.1
    weights = torch.Tensor(weights)
    # if cuda:
    weights = weights.cuda()
    inputs = torch.sigmoid(inputs)
    loss = torch.nn.BCELoss(weights, reduction='sum')(inputs.float(), targets.float())
    return l_weigts*loss

def rcf_loss(inputs, label):

    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float() # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0.
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return 1.*torch.sum(cost)

# ------------ cats losses ----------

def bdrloss(prediction, label, radius,device='cpu'):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)



    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()



def textureloss(prediction, label, mask_radius, device='cpu'):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)


def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
    # tracingLoss
    tex_factor,bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)
    # print('bce')
    cost = torch.sum(torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False))
    label_w = (label != 0).float()
    # print('tex')
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

    return cost + bdr_factor * bdrcost + tex_factor * textcost


