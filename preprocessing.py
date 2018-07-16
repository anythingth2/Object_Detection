
# coding: utf-8

# In[1]:


import json
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import random
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


input_shape = (224,224,3)
number_anchor = 9
anchor_ratios = [(1,1), (1,2), (2,1), (2,2), (2,4), (4,2), (4,4), (4,8), (8,4)]
anchor_default_size = 28


# In[3]:


train_datasets_path = 'train2014/'
annotations_path = 'annotations/'


# In[4]:


annotations_json_path = annotations_path + 'instances_train2014.json'
annotations_json = json.load(open(annotations_json_path,'r'))
print('loading annotaions')


# In[5]:


label_id_dict ={annotations_json['categories'][i]['id']:i for i in range(len(annotations_json['categories']))}


# In[6]:


def get_label_name(category_id):
    return annotations_json['categories'][label_id_dict[annotation['category_id']]]['name']


# In[7]:


datasets = {}

def load_images(size=None):
    global datasets
    datasets = {}
    if size == None:
        size = len(annotations_json['images'])
    for i, image_info in enumerate(annotations_json['images'][:size]):
        img_name = train_datasets_path + image_info['file_name']
        print('{} % ,loading {}'.format((i+1)*100//size,img_name))
        
        img = cv2.imread(img_name)
        img, resize_ratio = resize_by_padding(img,input_shape[1],input_shape[0])
        
        data = {'img':img,
                'resize_ratio':resize_ratio,
                'index':i,
                'original_resolution':(image_info['height'], image_info['width'])
               }
        datasets[image_info['id']] = data


# In[8]:


def resize_by_padding(img,net_width,net_height):
    img_h,img_w = img.shape[:2]

    if img_h > img_w:
        new_h = int(net_height)
        new_w = int(img_w * net_height / img_h)
    else:
        new_w = int(net_width)
        new_h = int(img_h * net_width / img_w)

    resize_ratio = (new_w / img_w, new_h / img_h)
    
    img = img[:,:,::-1]
    img = img/255.
    resized = cv2.resize(img,(new_w,new_h))
    base_img = np.ones((net_height,net_width,3)) * 0.5
    base_img[(net_height-new_h)//2 : (net_height+new_h)//2,(net_width-new_w)//2 : (net_width+new_w)//2,:] = resized
    
    resize_offset = ((net_width - new_w)//2, (net_height - new_h)//2)
    return base_img, (resize_ratio, resize_offset)


# In[9]:


def draw_box(img,bbox_ratio,color,grid=False,centroid=False,debug=False):
    bbox = np.copy(bbox_ratio)
    bbox[[0,2]] *= input_shape[1]
    bbox[[1,3]] *= input_shape[0]
    
    if grid:
        y_grid = np.arange(input_shape[0]//net_height,input_shape[0],input_shape[0]//net_height)
        x_grid = np.arange(input_shape[1]//net_width,input_shape[1],input_shape[1]//net_width)
        img[y_grid] = (0,0,0)
        img[ : , x_grid] = (0,0,0)
        
    if centroid:
        img = cv2.circle(img,(int(bbox[0]),int(bbox[1])),1,color,thickness=1)
    
    coor = [bbox[0] - bbox[2]/2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2]/2, bbox[1] + bbox[3] / 2]
    coor = [int(x) for x in coor]
    coor = tuple(coor)
    if debug:
        print(coor)
    img = cv2.rectangle(img,coor[:2],coor[2:],color,1)
    
    return img


# In[10]:


def convert_resized_bbox(bbox,resize_ratio):
    ratio, offset = resize_ratio
    return [bbox[0]*ratio[0]+offset[0], bbox[1]*ratio[1]+offset[1], bbox[2]*ratio[0], bbox[3]*ratio[1]]
def convert_bbox_to_ratio(bbox,img_shape):
    return [bbox[0]/img_shape[1], bbox[1]/img_shape[0], bbox[2]/img_shape[1], bbox[3]/img_shape[0]]
def convert_cocobbox_to_anchorbbox(bbox):
    return [bbox[0]+(bbox[2]//2), bbox[1]+(bbox[3]//2), bbox[2], bbox[3]]


# In[11]:


def generate_anchor_bbox(stride_size):
    base_anchor = np.array([
            stride_size/2, stride_size/2, 
            anchor_default_size, anchor_default_size
            ])
#     anchors = []
#     for y_box in range(net_height):
#         for x_box in range(net_width):
            
#             anchors.append(tmp_anchors)
#     anchors = np.array(anchors)
#     return anchors
    tmp_anchor = np.copy(base_anchor)
#     tmp_anchor[:2] += x_box*stride_size, y_box*stride_size
    tmp_anchors = []
    for ratio in anchor_ratios:
        tmp = np.copy(tmp_anchor)
        tmp[2:] *= ratio
        tmp = convert_bbox_to_ratio(tmp, input_shape[:2])
        tmp_anchors.append(tmp)
    tmp_anchors = np.array(tmp_anchors)
    return tmp_anchors

def generate_all_anchor_in_image(base_anchor):
    all_anchor = np.empty((net_height, net_width, number_anchor*4))
    for row in range(net_height):
        for column in range(net_width):
            
            anchor = np.copy(base_anchor)
            anchor[:,0] *= column*2
            anchor[:,1] *= row*2
            anchor = np.reshape(anchor,(number_anchor*4,))
            all_anchor[row,column] = anchor
    return all_anchor


# In[12]:


def iou(bbox1,bbox2):
    box_coor1 = [bbox1[0] - bbox1[2] / 2,
                 bbox1[1] - bbox1[3] / 2,
                 bbox1[0] + bbox1[2] / 2,
                 bbox1[1] + bbox1[3] / 2
                ]
    box_coor2 = [bbox2[0] - bbox2[2] / 2,
                 bbox2[1] - bbox2[3] / 2,
                 bbox2[0] + bbox2[2] / 2,
                 bbox2[1] + bbox2[3] / 2
                ]
    
    x_start_right = max(box_coor1[0], box_coor2[0])
    x_end_left = min(box_coor1[2],box_coor2[2])
    y_start_bottom = max(box_coor1[1],box_coor2[1])
    y_end_top = min(box_coor1[3],box_coor2[3])
    
    #check overlap
    if not((x_start_right < x_end_left) and (y_start_bottom < y_end_top)):
        return 0.0
    intersection = abs((x_start_right - x_end_left) * (y_start_bottom - y_end_top))
    union = bbox1[2]*bbox1[3] + bbox2[2]*bbox2[3] - intersection
    return intersection / union
    


# In[24]:


net_height, net_width = 28, 28

def map_gtbox():
    global datasets
    for i, annotation in enumerate(annotations_json['annotations']):
        try:
            datasets[annotation['image_id']]
        except KeyError:
            continue
        class_id = annotation['category_id']
        truth_bbox = annotation['bbox']
        truth_bbox = convert_cocobbox_to_anchorbbox(truth_bbox)

        truth_bbox = convert_resized_bbox(truth_bbox, datasets[annotation['image_id']]['resize_ratio']) 
        truth_bbox = convert_bbox_to_ratio(truth_bbox, input_shape[:2])
        if 'truth_bbox' in datasets[annotation['image_id']]:
            datasets[annotation['image_id']]['truth_bbox'].append(truth_bbox)
        else:
            datasets[annotation['image_id']]['truth_bbox'] = [truth_bbox]
def compute_class_score_and_rgs(iou_thresold,debug=False):
    global datasets
    
    stride_size = input_shape[0] // net_height
    base_anchors = generate_anchor_bbox(stride_size)
    
    cls = np.zeros((len(datasets), net_height, net_width, number_anchor))
    cls[:,:,:] = np.array([0]*number_anchor)
    rgs = np.zeros((len(datasets), net_height, net_width, number_anchor*4))
#     rgs[:,:,:] = base_anchors.reshape((36,))
    
    if debug:
        fig = plt.figure(figsize=(224,224))
        
    for i, key in enumerate(datasets):
        data = datasets[key]
        try:
            truth_bboxes = data['truth_bbox']
        except:
            continue
        for gt_bbox in truth_bboxes:
            n_column = int(gt_bbox[0]*input_shape[0] // stride_size)
            n_row = int(gt_bbox[1]*input_shape[1]  // stride_size)
            anchors = np.copy(base_anchors)
            anchors[:,0] += n_column*stride_size/input_shape[0]
            anchors[:,1] += n_row*stride_size/input_shape[1]
#             rgs[:,:,:] = anchors.reshape((36,))
            if debug:
                img = np.copy(data['img'])
                img = draw_box(img,gt_bbox,(0,255,0),centroid=True)
            best_iou_score = 0
            best_iou_index = 0
            best_iou_anchor = []
            for j, anchor in enumerate(anchors):
                iou_score = iou(gt_bbox,anchor)
                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_iou_index = j
                    best_iou_anchor = np.copy(anchor)
                if iou_score > iou_thresold:
                    cls[i, n_row, n_column, j] = 1
                    if debug:
                        fig.add_subplot(5,4,j+1)                        
                        plt.imshow(draw_box(img, anchor, (255,0,0),centroid=True))
            if best_iou_score > iou_thresold:
                rgs[i, n_row, n_column, best_iou_index*4:best_iou_index*4+4] = best_iou_anchor
    return cls,rgs


# In[25]:


def load_datasets(size, iou_thresold):
    global datasets
    load_images(size)
    map_gtbox()
    cls, rgs = compute_class_score_and_rgs(iou_thresold)
    images = np.array([datasets[key]['img'] for key in datasets])
    del datasets
    return images, cls, rgs

