'''
 my_visualizer.py
 Visualizer for shelf and goods
    Author: Limy
    Created on March 29th, 2020
'''
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from random import random as rand

import os
import cv2
import numpy as np
from utils_io_file import is_image
from utils_io_folder import create_folder
from utils_json import read_json_from_file

bbox_thresh = 0.4

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    if s_poi[1]==e_poi[1]: 
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: 
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: 
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: 
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: 
        return False
    if s_poi[0]<poi[0] and e_poi[1]<poi[1]: 
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) 
    if xseg<poi[0]: 
        return False
    return True  

def isPoiWithinPoly(poi,poly):
    sinsc=0 
    for epoly in poly: 
        for i in range(len(epoly)): 
            s_poi=epoly[i%len(epoly)]
            e_poi=epoly[(i+1)%len(epoly)]
            if isRayIntersectsSegment(poi,s_poi,e_poi):
                sinsc+=1 

    return True if sinsc%2==1 else  False

def draw_shelf_bbox(img, joints, joint_pairs, joint_names, track_id = -1, trigger = False, img_id = -1, img_path=None, pick_results=[]):
    shelf_bbox_pixels = [[[[706, 342],[707, 313],[823, 463],[817, 494]],
                            [[709, 308],[713, 275],[833, 416],[825, 454]],
                            [[713, 266],[716, 226],[849, 365],[838, 408]],
                            [[719, 221],[722, 172],[868, 297],[853, 357]],
                            [[723, 170],[728, 107],[888, 214],[875, 288]],
                            [[729, 104],[738,  20],[934, 110],[887, 213]]],
                            [[[ 818,  496],[ 827,  459],[ 987,  661],[ 964,  698]],
                            [[ 831,  450],[ 839,  415],[1015,  614],[ 994,  650]],
                            [[ 846,  407],[ 851,  368],[1044,  560],[1023,  606]],
                            [[ 858,  356],[ 869,  299],[1088,  484],[1052,  551]],
                            [[ 877,  290],[ 889,  231],[1141,  405],[1098,  474]],
                            [[ 896,  221],[ 924,  115],[1248,  263],[1154,  392]]]]

    color = (0, 255, 0) #green
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, shelf in enumerate(shelf_bbox_pixels):
        for j, pixels in enumerate(shelf):
            pts = np.array(pixels,np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],isClosed = True,color = color,thickness = 2)
            cv2.putText(img,
                        #'{:s} {:.2f}'.format("ID:"+str(track_id), score),
                        '{:s}'.format(str(i+1)+"-"+str(j+1)),
                        (pts[0][0][0]-10, pts[0][0][1]-5),
                        font,
                        fontScale=0.8,
                        color=color,
                        thickness = 2,
                        lineType = cv2.LINE_AA)
    
    if trigger:
        ind_1 = joint_names.index('right elbow')
        ind_2 = joint_names.index('right wrist')
        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]
        x3 = int(x2 + (x2-x1) / 4)
        y3 = int(y2 + (y2-y1) / 4)
        cv2.circle(img, (x3, y3), radius=3, color=(255,255,255), thickness=2)

        for i, shelf in enumerate(shelf_bbox_pixels):
            for j, pixels in enumerate(shelf):
                if isPoiWithinPoly([x3,y3], [pixels]):
                    pts = np.array(pixels,np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(img,[pts],isClosed = True,color = (0,0,255),thickness = 2)
                    
                    memory_img_id = img_id-50 #img of 50 frame before
                    memory_img_path = img_path[:-7] + str(memory_img_id) + img_path[-4:] 
                    if is_image(memory_img_path):    
                        memory_img = cv2.imread(memory_img_path)
                        cropImg = memory_img[y3-30:y3+30,x3-15:x3+45]
                    else:
                        print("No memory img {}.".format(memory_img_id))
                    
                    pick_results.append({'track_id':track_id, 'shelf_id':[i+1,j+1], 'img_id':img_id, 'hand_pos':[x3,y3], 'memory_item_img':cropImg})
                    
                    print("ID:"+str(track_id)+" picks on shelf "+ str(i+1)+ "-" + str(j+1) +" at time "+str(img_id))
                    
                    break
    
    vis_result_x = 30
    vis_result_y = 30 
    for i, result in enumerate(pick_results):
        cv2.putText(img,
            '{:s}'.format("ID:"+str(result['track_id'])+" picks on shelf "+ str(result['shelf_id'][0])+ "-" + str(result['shelf_id'][1]) +" at time "+str(result['img_id'])),
            (vis_result_x, vis_result_y),
            font,
            fontScale=0.8,
            color=(255,255,255),
            thickness = 2,
            lineType = cv2.LINE_AA)
        vis_result_y += 70
        
        img[vis_result_y-60:vis_result_y, vis_result_x:vis_result_x+60] = result['memory_item_img']
        vis_result_y += 30
    return img, pick_results


# def show_boxes_from_python_data(img, dets, classes, output_img_path, scale = 1.0):
#     plt.cla()
#     plt.axis("off")
#     plt.imshow(img)
#     for cls_idx, cls_name in enumerate(classes):
#         cls_dets = dets[cls_idx]
#         for det in cls_dets:
#             bbox = det[:4] * scale
#             color = (rand(), rand(), rand())
#             rect = plt.Rectangle((bbox[0], bbox[1]),
#                                   bbox[2] - bbox[0],
#                                   bbox[3] - bbox[1], fill=False,
#                                   edgecolor=color, linewidth=2.5)
#             plt.gca().add_patch(rect)

#             if cls_dets.shape[1] == 5:
#                 score = det[-1]
#                 plt.gca().text(bbox[0], bbox[1],
#                                '{:s} {:.3f}'.format(cls_name, score),
#                                bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
#     plt.show()
#     plt.savefig(output_img_path)
#     return img


# def find_color_scalar(color_string):
#     color_dict = {
#         'purple': (255, 0, 255),
#         'yellow': (0, 255, 255),
#         'blue':   (255, 0, 0),
#         'green':  (0, 255, 0),
#         'red':    (0, 0, 255),
#         'skyblue':(235,206,135),
#         'navyblue': (128, 0, 0),
#         'azure': (255, 255, 240),
#         'slate': (255, 0, 127),
#         'chocolate': (30, 105, 210),
#         'olive': (112, 255, 202),
#         'orange': (0, 140, 255),
#         'orchid': (255, 102, 224)
#     }
#     color_scalar = color_dict[color_string]
#     return color_scalar


# def draw_bbox(img, bbox, score, classes, track_id = -1, img_id = -1):
#     if track_id == -1:
#         color = (255*rand(), 255*rand(), 255*rand())
#     else:
#         color_list = ['purple', 'yellow', 'blue', 'green', 'red', 'skyblue', 'navyblue', 'azure', 'slate', 'chocolate', 'olive', 'orange', 'orchid']
#         color_name = color_list[track_id % 13]
#         color = find_color_scalar(color_name)

#     if img_id % 10 == 0:
#         color = find_color_scalar('red')
#     elif img_id != -1:
#         color = find_color_scalar('blue')

#     cv2.rectangle(img,
#                   (bbox[0], bbox[1]),
#                   (bbox[0]+ bbox[2], bbox[1] + bbox[3]),
#                   color = color,
#                   thickness = 3)

#     cls_name = classes[0]
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     if track_id == -1:
#         cv2.putText(img,
#                     #'{:s} {:.2f}'.format(cls_name, score),
#                     '{:s}'.format(cls_name),
#                     (bbox[0], bbox[1]-5),
#                     font,
#                     fontScale=0.8,
#                     color=color,
#                     thickness = 2,
#                     lineType = cv2.LINE_AA)
#     else:
#         cv2.putText(img,
#                     #'{:s} {:.2f}'.format("ID:"+str(track_id), score),
#                     '{:s}'.format("ID:"+str(track_id)),
#                     (bbox[0], bbox[1]-5),
#                     font,
#                     fontScale=0.8,
#                     color=color,
#                     thickness = 2,
#                     lineType = cv2.LINE_AA)
#     return img


# def show_boxes_from_standard_json(json_file_path, classes, img_folder_path = None, output_folder_path = None, track_id = -1):
#     dets = read_json_from_file(json_file_path)

#     for det in dets:
#         python_data = det

#         if img_folder_path is None:
#             img_path = os.path.join(python_data["image"]["folder"], python_data["image"]["name"])
#         else:
#             img_path = os.path.join(img_folder_path, python_data["image"]["name"])
#         if is_image(img_path):    img = cv2.imread(img_path)

#         candidates = python_data["candidates"]
#         for candidate in candidates:
#             bbox = np.array(candidate["det_bbox"]).astype(int)
#             score = candidate["det_score"]
#             if score >= bbox_thresh:
#                 img = draw_bbox(img, bbox, score, classes, track_id = track_id)

#         if output_folder_path is not None:
#             create_folder(output_folder_path)
#             img_output_path = os.path.join(output_folder_path, python_data["image"]["name"])
#             cv2.imwrite(img_output_path, img)
#     return True
