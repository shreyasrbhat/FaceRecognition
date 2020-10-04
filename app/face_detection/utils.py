import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bbox_wh_iou(anch_x, anch_y, target_x, target_y):
    inter_area = torch.min(target_x, anch_x) * torch.min(target_y, anch_y)
    union_area = (anch_x * anch_y) + (target_x * target_y) - inter_area
    return inter_area/union_area

def build_targets(pred_boxes, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nS = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nS, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nS, nA, nG, nG).fill_(1)
    iou_scores = FloatTensor(nS, nA, nG, nG).fill_(0)
    tx = FloatTensor(nS, nA, nG, nG).fill_(0)
    ty = FloatTensor(nS, nA, nG, nG).fill_(0)
    tw = FloatTensor(nS, nA, nG, nG).fill_(0)
    th = FloatTensor(nS, nA, nG, nG).fill_(0)
    target_boxes = target[:, 1:5]
    s = target[:, 0].long()
    g_w, g_h = target_boxes[:, 2:4].T
    g_xy = target_boxes[:, 0:2] * nG
    g_i, g_j = g_xy.long().t()
    g_x, g_y = g_xy.t()
    
    ious = torch.stack([bbox_wh_iou(anch_x, anch_y, g_w, g_h) for anch_x, anch_y in anchors])
    best_ious, ind = ious.max(0)
    
    obj_mask[s, ind, g_i, g_j] = 1
    noobj_mask[s, ind, g_i, g_j] = 0
    
    tx[s, ind, g_i, g_j] = g_x - g_x.floor()
    ty[s, ind, g_i, g_j] = g_y - g_y.floor()
    
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[s[i], anchor_ious > ignore_thres, g_i[i], g_j[i]] = 0
    
    tw[s, ind, g_i, g_j] = torch.log(g_w/anchors[ind][:, 0] + 1e-16)
    th[s, ind, g_i, g_j] = torch.log(g_h/anchors[ind][:, 1] + 1e-16)
    
    iou_scores[s, ind, g_i, g_j] = bbox_iou(pred_boxes[s, ind, g_i, g_j], target_boxes,  x1y1x2y2=False)
    
    tconf = obj_mask.float()
    return iou_scores, obj_mask, noobj_mask, tx, ty, tw, th, tconf


def bbox_iou(boxes1, boxes2, x1y1x2y2=False):
    if not x1y1x2y2:
        boxes1_x1, boxes1_x2 = boxes1[:,0] - boxes1[:,2]/2, boxes1[:,0] + boxes1[:,2]/2
        boxes1_y1, boxes1_y2 = boxes1[:,1] - boxes1[:,3]/2, boxes1[:,1]+boxes1[:,3]/2
        boxes2_x1, boxes2_x2 = boxes2[:,0] - boxes2[:,2]/2, boxes2[:,0] + boxes2[:,2]/2
        boxes2_y1, boxes2_y2 = boxes2[:,1] - boxes2[:,3]/2, boxes2[:,1]+boxes2[:,3]/2
    
    else:
        boxes1_x1, boxes1_x2, boxes1_y1, boxes1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        boxes2_x1, boxes2_x2, boxes2_y1, boxes2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
    inter_w =  torch.clamp(torch.min(boxes1_x2, boxes2_x2) - torch.max(boxes1_x1, boxes2_x1) + 1, min=0)
    inter_h = torch.clamp(torch.max(boxes1_y2, boxes2_y2) - torch.min(boxes1_y1, boxes2_y1 ) + 1, min=0)
    
    inter_area = inter_w * inter_h
    
    boxes1_area = torch.abs(boxes1_x1-boxes1_x2+1) * torch.abs(boxes1_y1-boxes1_y2+1)
    boxes2_area = torch.abs(boxes2_x1-boxes2_x2+1) * torch.abs(boxes2_y1-boxes2_y2+1)
    
    iou = inter_area/(boxes1_area + boxes2_area - inter_area)
    
    return iou