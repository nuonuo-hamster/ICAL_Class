import cv2
import numpy as np
from . import load_data

def iou(gt, rp):
    # gt & rp are rect: (x0, y0, x1, y1)
    gt_tlx, gt_tly, gt_brx, gt_bry = gt
    rp_tlx, rp_tly, rp_brx, rp_bry = rp
    
    # gt area + rp area
    gt_area = (gt_brx - gt_tlx) * (gt_bry - gt_tly)
    rp_area = (rp_brx - rp_tlx) * (rp_bry - rp_tly)
    sum_area = gt_area + rp_area
    
    # intersect area of gt & rp
    i_tlx = max(gt_tlx, rp_tlx)
    i_tly = max(gt_tly, rp_tly)
    i_brx = min(gt_brx, rp_brx)
    i_bry = min(gt_bry, rp_bry)
    i_area = (i_brx - i_tlx) * (i_bry - i_tly)
    
    # adjust intersect correct    
    if i_tlx >= i_brx or i_tly >= i_bry:
        return 0
    else:
        return (i_area / float(sum_area - i_area))
    
def img_empty(img):
    area = 1
    for i in img.shape:
        area = area * i
    return area <= 0

def rect_format_opencv(cvrect):
    x, y, w, h = cvrect
    return (x, y, x + w, y + h)

def rect_format_yolo(txtrect):
    x, y, w, h = txtrect
    return (x - int(w/2), y - int(h/2), x + int(w/2), y + int(h/2))    

def rect_format_iou(rect):
    tlx, tly, brx, bry = rect
    x = int((tlx + brx) / 2.0)
    y = int((tly + bry) / 2.0)
    w = int(brx - tlx)
    h = int(bry - tly)
    return [x, y, w, h]

def ssp(x, size, resize):
    ssp_x = np.array([])
    ssp_rects = np.array([])
    for i in range(x.shape[0]):
        img = (x[i, :] * 255).astype('uint8').copy()
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        for _, rect in enumerate(rects):            
            rect_rp = rect_format_opencv(rect)
            roi = img[rect_rp[0]:(rect_rp[2]+1),rect_rp[1]:(rect_rp[3]+1)].copy()
            if img_empty(roi): continue
            roi = cv2.resize(roi, resize[0:-1])
            roi = roi[np.newaxis, :] / 255.
            ssp_x = load_data.data_append(ssp_x, roi)
            ssp_rects = load_data.data_append(ssp_rects, rects)
                            
    return ssp_x, ssp_rects

def ssp_data(x, y, size, resize, iou_thres):
    ssp_x = np.array([])
    ssp_y = np.array([])
    ussp_x = np.array([])
    ussp_y = np.array([])
    gt_y = np.array([])
    rect_y = y[:, 1:y.shape[0]]
    rect_y[:, 0] = rect_y[:, 0] * size[0]
    rect_y[:, 1] = rect_y[:, 1] * size[1]
    rect_y[:, 2] = rect_y[:, 2] * size[0]
    rect_y[:, 3] = rect_y[:, 3] * size[1]
    rect_y = rect_y.astype('uint16')
    
    for i in range(x.shape[0]):
        img = (x[i, :] * 255).astype('uint8').copy()
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        
        rect_gt = rect_format_yolo(rect_y[i])
        for num, rect in enumerate(rects):            
            rect_rp = rect_format_opencv(rect)
            roi = img[rect_rp[0]:(rect_rp[2]+1),rect_rp[1]:(rect_rp[3]+1)].copy()
            if img_empty(roi): continue
            roi = cv2.resize(roi, resize[0:-1])
            roi = roi[np.newaxis, :] / 255.
            if iou(rect_gt, rect_rp) > iou_thres:
                print('#',i, 'ss:', num, '/', len(rects), 'gt:', rect_gt, 'rp:', rect_rp, 'iou:', iou(rect_gt, rect_rp))
                ssp_x = load_data.data_append(ssp_x, roi)
                ssp_y = load_data.data_append(ssp_y, np.array(rect_format_iou(rect_rp))[np.newaxis, :])
                gt_y = load_data.data_append(gt_y, y[i, :][np.newaxis, :])
            else:
                ussp_x = load_data.data_append(ussp_x, roi)
                ussp_y = load_data.data_append(ussp_y, np.array([0]))
                            
    return ssp_x, ssp_y, gt_y, ussp_x, ussp_y