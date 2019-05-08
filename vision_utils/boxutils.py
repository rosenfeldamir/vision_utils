import numpy as np
import time
import cv2
from matplotlib import pyplot as plt
squashImage = True
from numpy import array

def cropImage(img, rect):
    return img[rect[1]:rect[3], rect[0]:rect[2], :]

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def boxCenters(boxes):
    x = (boxes[:, 0:1]+boxes[:, 2:3])/2
    y = (boxes[:, 1:2]+boxes[:, 3:4])/2
    return np.concatenate((x, y), 1)


def plotRectangle(r, color='r', **kwargs):
		if isinstance(r, np.ndarray):
			 # print 'd'
				r = r.tolist()
	 # print r
		points = [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]], ]
	 # print points    
		line = plt.Polygon(points, closed=True, fill=None, edgecolor=color, **kwargs)
		plt.gca().add_line(line)

def plotRectangle_img(img, t, color=(1, 0, 0), thickness=3):
    t = np.asarray(t).astype(int)
    cv2.rectangle(img, tuple(t[0:2]), tuple(t[2:4]), color, thickness)

def sampleTiledRects(img_size, boxSizeRatio=[5], boxOverlap=.5):
    if not isinstance(boxSizeRatio, list):
        boxSizeRatio = [boxSizeRatio]
    all_h = []
    for sizeRatio in boxSizeRatio:
        boxSize = np.asarray(img_size[0:2])/sizeRatio  # height, width
        b = boxSize*(1-boxOverlap)
        topLefts = [(x, y) for x in np.arange(0, img_size[1]-boxSize[1], b[1])
                    for y in np.arange(0, img_size[0]-boxSize[0], b[0])]
        xs, ys = zip(*topLefts)
        xs = np.floor(np.asarray(xs))
        ys = np.floor(np.asarray(ys))
        all_h.append(np.vstack([xs, ys, xs+boxSize[1], ys+boxSize[0]]).T)
    return np.vstack(all_h)

def sampleRandomRects(img_size, boxSizeRatio=5, nBoxes=50):
    '''
    samples random bounding boxes from image.
    '''
    boxSize = int(np.mean(img_size[0:2])/boxSizeRatio)
    center_ys = np.asarray(np.random.randint(
        low=boxSize/2, high=img_size[0]-boxSize/2, size=nBoxes))
    center_xs = np.asarray(np.random.randint(
        low=boxSize/2, high=img_size[1]-boxSize/2, size=nBoxes))
    h = np.vstack([center_xs-boxSize/2, center_ys-boxSize/2,
                   center_xs+boxSize/2, center_ys+boxSize/2]).T
    return h, zip(center_xs, center_ys)


def boxIntersection(b1, b2):
    xmin1 = b1[0]
    xmax1 = b1[2]
    xmin2 = b2[0]
    xmax2 = b2[2]
    ymin1 = b1[1]
    ymax1 = b1[3]
    ymin2 = b2[1]
    ymax2 = b2[3]
    res = np.asarray([max(xmin1, xmin2), max(ymin1, ymin2),
                      min(xmax1, xmax2), min(ymax1, ymax2)])
    return res

def boxArea(b):
    if b[2] < b[0]:
        return 0
    if b[3] < b[1]:
        return 0
    return (b[2]-b[0])*(b[3]-b[1])

def boxesOverlap(boxes1, boxes2):
    boxes1 = np.reshape(np.asarray(boxes1), (-1, 4)).astype(np.float32)
    boxes2 = np.reshape(boxes2, (-1, 4)).astype(np.float32)
    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]
    # print n1
    # print n2
    res = np.zeros((n1, n2))

    for i1, b1 in enumerate(boxes1):
        b1Area = boxArea(b1)
        for i2, b2 in enumerate(boxes2):
            
            curInt = boxIntersection(b1, b2)
            intArea = boxArea(curInt)
            if intArea <= 0:
                continue
            b2Area = boxArea(b2)
            assert(b1Area > 0)
            assert(b2Area > 0)
            assert(b1Area+b2Area-intArea > 0)
            res[i1, i2] = float(intArea)/(b1Area+b2Area-intArea)
    return res


def boxDims(b):
    width = b[2]-b[0]
    height = b[3]-b[1]
    return width, height


def boxAspectRatio(b):
    width, height = boxDims(b)
    width = float(width)
    height = float(height)
    res = max([height/width, width/height])
    return res


def inflateBox(b, f,is_abs=False):
    width, height = boxDims(b)
    if is_abs:
        width+=f/2
        height+=f/2
    else:
        width = (width*f)/2
        height = (height*f)/2
    center_x = (b[0]+b[2])/2
    center_y = (b[1]+b[3])/2
    # print center_x,center_y
    res = [center_x-width, center_y-height, center_x+width, center_y+height]
    return res


def chopLeft(box, p, mode=0):
    if mode == 0:
        box[0] = (1-p)*box[0]+p*box[2]
    else:
        box[0] = box[0]+p
    return box


def chopRight(box, p, mode=0):
    if mode == 0:
        box[2] = (1-p)*box[2]+p*box[0]
    else:
        box[2] = box[2]-p
    return box


def chopTop(box, p, mode=0):
    if mode == 0:
        box[1] = (1-p)*box[1]+p*box[3]
    else:
        box[1] = box[1]+p
    return box


def chopBottom(box, p, mode=0):
    if mode == 0:
        box[3] = (1-p)*box[3]+p*box[1]
    else:
        box[3] = box[3]-p
    return box


def chopAll(box, p, mode=0):
    box = chopBottom(
        chopTop(chopLeft(chopRight(box, p, mode), p, mode), p, mode), p, mode)
    return box


def plotRectangle(r, color='r', **kwargs):
    if isinstance(r, np.ndarray):
       # print 'd'
        r = r.tolist()
   # print r
    points = [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]], ]
   # print points
    line = plt.Polygon(points, closed=True, fill=None, edgecolor=color, lw=2, **kwargs)
    plt.gca().add_line(line)
    # show()

def splitBox(box, p, mode):
    return [chopLeft(box.copy(), p, mode),
            chopRight(box.copy(), p, mode),
            chopTop(box.copy(), p, mode),
            chopBottom(box.copy(), p, mode), chopAll(box.copy(), p, mode)]
    # inflateBox(box.copy(),.8,)]


def relBox2Box(box):
    box[2] = box[2]+box[0]
    box[3] = box[3]+box[1]
    return box

def boxCenter(box):
    w, h = boxDims(box)
    boxCenterX = box[0]+w/2
    boxCenterY = box[1]+h/2
    return boxCenterX, boxCenterY

def makeSquare(box):
    w, h = boxDims(box)
    m = float(max([w, h]))/2
    boxCenterX = box[0]+w/2
    boxCenterY = box[1]+h/2
    newBox = [boxCenterX-m, boxCenterY-m, boxCenterX+m, boxCenterY+m]
    return newBox

def clipBox(targetBox, box):
    targetBox = list(targetBox)
    if len(targetBox) == 2:
        targetBox = [0, 0]+targetBox[::-1]
    newBox = boxIntersection(targetBox, box)
    return newBox

def remove_box_overlaps(h, thresh=.5):
    '''
    greedily remove bounding boxes until there are no more overlaps above a given threshold.
    '''
    keep_boxes = [True]*len(h)
    overlaps = boxesOverlap(h, h)

    # TODO: possibly re-order the boxes so that the neighbors of the most-overlapping box are considered first?
    for ibox in range(len(h)):
        if not keep_boxes[ibox]:
            continue  # already removed this one.
        cur_overlaps = overlaps[ibox]
        cur_overlaps[ibox] = 0
        other_boxes = np.nonzero(cur_overlaps > thresh)[0]
        for j in other_boxes:
            keep_boxes[j] = False
    return keep_boxes

def ptsToBox(xs,ys):
  xmin = min(xs)
  ymin = min(ys)
  xmax = max(xs)
  ymax = max(ys)
  return xmin,ymin,xmax,ymax

def split_boxes(boxes):
    '''
    split np array to column representation
    '''
    a = boxes[:,0:1]
    b = boxes[:,1:2]
    c = boxes[:,2:3]
    d = boxes[:,3:4]
    
    return a,b,c,d

def to_boxes(a,b,c,d):
    '''
    concat column representation to np array. 
    '''
    return np.hstack([a,b,c,d])

def to_CWH(boxes,box_fmt='united'):
    xmin,ymin,xmax,ymax = split_boxes(boxes)
    cx, cy = (xmin + xmax) / 2, (ymax + ymin) / 2
    w = xmax - xmin
    h = ymax - ymin
    if box_fmt=='united':        
        return to_boxes(cx,cy,w,h)
    else:
        return cx,cy,w,h

def bb_int(boxes1,boxes2):
	res = np.zeros((len(boxes1),len(boxes2)))
	for i1,b1 in enumerate(boxes1):
		for i2,b2 in enumerate(boxes2):
			res[i1,i2] = boxArea(boxIntersection(b1,b2))
	return res
def bb_areas(boxes):
	return array([boxArea(b) for b in boxes])