
import time
import argparse

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf
import torch 
import torchvision
from torchvision import transforms 

# import Network
from network_mobile_deconv import Network

# detector utils
from detector.detector_yolov3 import *

# person reID utils
from person_reID.person_reID_base import *


# pose estimation utils
from HPE.dataset import Preprocessing
from HPE.config import cfg
from tfflat.base import Tester
from tfflat.utils import mem_info
from tfflat.logger import colorlogger
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

# import GCN utils
from graph import visualize_pose_matching
from graph  .visualize_pose_matching import *

# import my own utils
import sys, os, time
sys.path.append(os.path.abspath("./graph/"))
from utils_json import *
from visualizer import *
from utils_io_file import *
from utils_io_folder import *

import queue
import numpy as np
import time
# from weight_data_read import csv_file_read as cfr

# from weight_main import WeightSensor


flag_visualize = True
flag_nms = False #Default is False, unless you know what you are doing

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def initialize_parameters():
    global video_name, img_id

    global nms_method, nms_thresh, min_scores, min_box_size
    nms_method = 'nms'
    nms_thresh = 1.
    min_scores = 1e-10
    min_box_size = 0.

    global keyframe_interval, enlarge_scale, pose_matching_threshold
    keyframe_interval = 40 # choice examples: [2, 3, 5, 8, 10, 20, 40, 100, ....]
    enlarge_scale = 0.2 # how much to enlarge the bbox before pose estimation
    pose_matching_threshold = 0.5

    global flag_flip
    flag_flip = True

    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    total_time_POSE = 0
    total_time_DET = 0
    total_time_ALL = 0
    total_num_FRAMES = 0
    total_num_PERSONS = 0

    global person_reID_list, person_reID_sim_thresh, person_ID
    person_reID_list = np.empty(shape=[0, 512])
    person_reID_sim_thresh = 0.1
    person_ID = -1
    print(person_ID)

    global pick_results
    pick_results = []
    return

class Vision_module:
    def __init__(self, camera_id, image_folder, shelf_bbox_list, init_timestamp):
        self.camera_id = camera_id
        self.image_folder = image_folder
        self.shelf_bbox_list = shelf_bbox_list
        self.init_timestamp = init_timestamp # 1580250245.19951 - 1/20 

        #self.shelf_visible_map = shelf_visible_map
        self.keypoints_list = []
        self.bbox_dets_list = []
        self.frame_prev = -1
        self.frame_cur = 0
        self.img_id = -1
        self.timestamp = -1
        self.next_id = 0
        self.keyframe_interval = 40
        self.bbox_dets_list_list = []
        self.keypoints_list_list = []

        self.flag_mandatory_keyframe = False

        self.img_paths = get_immediate_childfile_paths(self.image_folder)
        self.num_imgs = len(self.img_paths)
        self.total_num_FRAMES = self.num_imgs
        self.REPEAT = False

        self.joint_names = ['right ankle', 'right knee', 'right pelvis', 'left pelvis',
                            'left knee', 'left ankle', 'right wrist',
                            'right elbow', 'right shoulder', 'left shoulder', 'left elbow', 'left wrist',
                            'upper neck', 'nose', 'head']

    def update_frame(self, pose_estimator, signal):

        global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
        global pick_results
        global person_reID_list, person_ID
        pick_result = None
        
        if self.img_id >= self.num_imgs:
            print('No more frames.')
            return pick_result

        self.img_id += 1
        self.timestamp += 1/20 #fps=20

        #print('Updating camera {} on frame {} with signal {}'.format(self.camera_num, self.img_id, signal))
        if signal > 0:
            img_path = self.img_paths[self.img_id]
            self.bbox_dets_list = []  # keyframe: start from empty
            self.keypoints_list = []  # keyframe: start from empty

            # perform detection at keyframes
            human_candidates = inference_yolov3(img_path)

            num_dets = len(human_candidates)
            print("Get {} detections".format(num_dets))

            # if nothing detected at keyframe, regard next frame as keyframe because there is nothing to track
            if num_dets <= 0:
                print('No detected person.')
                # add empty result
                bbox_det_dict = {"img_id":self.img_id,
                                "det_id":  0,
                                "track_id": None,
                                "imgpath": img_path,
                                "bbox": [0, 0, 2, 2]}
                self.bbox_dets_list.append(bbox_det_dict)

                keypoints_dict = {"img_id":self.img_id,
                                "det_id": 0,
                                "track_id": None,
                                "imgpath": img_path,
                                "keypoints": []}
                self.keypoints_list.append(keypoints_dict)
                return pick_result

            # For each candidate, perform pose estimation
            for det_id in range(num_dets):
                # obtain bbox position
                bbox_det = human_candidates[det_id]

                # enlarge bbox by 20% with same center position
                bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
                bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
                bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
                print(bbox_x1y1x2y2)

                # Keyframe: use provided bbox
                if bbox_invalid(bbox_det):
                    print('INVALID DETECTION.')
                    continue

                # update current frame bbox
                bbox_det_dict = {"img_id":self.img_id,
                                "det_id":det_id,
                                "imgpath": img_path,
                                "bbox":bbox_det}
                self.bbox_dets_list.append(bbox_det_dict)

            self.bbox_dets_list_list.append(self.bbox_dets_list)

            if signal == 2: # add person
                person_reID_add(img_path, self.bbox_dets_list[0]["bbox"], self.img_id, self.camera_id)
            
            elif signal == 3: # remove person    
                for bbox_det_dict in self.bbox_dets_list:
                    person_reID_remove(img_path, self.bbox_dets_list[0]["bbox"], self.img_id, self.camera_id) 

            elif signal == 1: # detect pick action        
                for bbox_det_dict in self.bbox_dets_list:
                    keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]
                    
                    #get track ID
                    track_id = get_track_id_person_reID(img_path, bbox_det_dict["bbox"], self.img_id, self.camera_id)            
                    pick_result = {'track_id':track_id}

                    #check hand location
                    img = cv2.imread(img_path)

                    # right_hand
                    ind_1_r = self.joint_names.index('right elbow')
                    ind_2_r = self.joint_names.index('right wrist')
                    x1_r, y1_r, sure1_r = keypoints[ind_1_r*3:(ind_1_r+1)*3]
                    x2_r, y2_r, sure2_r = keypoints[ind_2_r*3:(ind_2_r+1)*3]
                    x3_r = int(x2_r + (x2_r-x1_r) / 4)
                    y3_r = int(y2_r + (y2_r-y1_r) / 4)

                    # right_hand
                    ind_1_l = self.joint_names.index('left elbow')
                    ind_2_l = self.joint_names.index('left wrist')
                    x1_l, y1_l, sure1_l = keypoints[ind_1_l*3:(ind_1_l+1)*3]
                    x2_l, y2_l, sure2_l = keypoints[ind_2_l*3:(ind_2_l+1)*3]
                    x3_l = int(x2_l + (x2_l-x1_l) / 4)
                    y3_l = int(y2_l + (y2_l-y1_l) / 4)

                    #TO DO: receive shelf info from weight sensor and only detect that shelf 
                    for i, shelf in enumerate(self.shelf_bbox_list):
                        for j, pixels in enumerate(shelf):
                            if isPoiWithinPoly([x3_r,y3_r], [pixels]):
                                pts = np.array(pixels,np.int32)
                                pts = pts.reshape((-1,1,2))
                                
                                memory_img_id = self.img_id-50 #img of 50 frame before
                                memory_img_path = img_path[:-7] + str(memory_img_id) + img_path[-4:] 
                                if is_image(memory_img_path):    
                                    memory_img = cv2.imread(memory_img_path)
                                    cropImg = memory_img[y3_r-30:y3_r+30,x3_r-15:x3_r+45]
                                else:
                                    print("No memory img {}.".format(memory_img_id))
                                
                                cv2.imwrite('./videos/camera{}-frame{}-detect-item.jpg'.format(self.camera_id, self.img_id), cropImg)
                                pick_result = {'track_id':track_id, 'shelf_id':[i+1,j+1], 'img_id':self.img_id, 'hand_pos':[x3_r,y3_r], 'memory_item_img':cropImg} 

                                print("ID:"+str(track_id)+" picks on shelf "+ str(i+1)+ "-" + str(j+1) +" at time "+str(self.img_id))
                                break
                                
                        
                            if isPoiWithinPoly([x3_l,y3_l], [pixels]):
                                pts = np.array(pixels,np.int32)
                                pts = pts.reshape((-1,1,2))
                               
                                memory_img_id = self.img_id-50 #img of 50 frame before
                                memory_img_path = img_path[:-7] + str(memory_img_id) + img_path[-4:] 
                                if is_image(memory_img_path):    
                                    memory_img = cv2.imread(memory_img_path)
                                    cropImg = memory_img[y3_l-30:y3_l+30,x3_l-15:x3_l+45]
                                else:
                                    print("No memory img {}.".format(memory_img_id))
                                
                                cv2.imwrite('./videos/camera{}-frame{}-detect-item.jpg'.format(self.camera_id, self.img_id), cropImg)
                                pick_result = {'track_id':track_id, 'shelf_id':[i+1,j+1], 'img_id':self.img_id, 'hand_pos':[x3_l,y3_l], 'memory_item_img':cropImg} 

                                print("ID:"+str(track_id)+" picks on shelf "+ str(i+1)+ "-" + str(j+1) +" at time "+str(self.img_id))
                                break

        return pick_result

    def add_person_feature(self, pose_estimator, timestamp):
        global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
        global pick_results
        global person_reID_list, person_ID
        pick_result = []

        curr_img_id = get_img_id_from_timestamp(timestamp, self.init_timestamp)
        print(curr_img_id)
        
        if curr_img_id >= self.num_imgs:
            print('No more frames.')
            return pick_result

        img_path = self.img_paths[curr_img_id]
        self.bbox_dets_list = []  # keyframe: start from empty
        self.keypoints_list = []  # keyframe: start from empty

        # perform detection at keyframes
        human_candidates = inference_yolov3(img_path)

        num_dets = len(human_candidates)
        print("Get {} detections".format(num_dets))

        # if nothing detected 
        if num_dets <= 0:
            print('No detected person.')
            return pick_result

        # For each candidate, save bbox
        for det_id in range(num_dets):
            # obtain bbox position
            bbox_det = human_candidates[det_id]

            # enlarge bbox by 20% with same center position
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            print(bbox_x1y1x2y2)

            # Keyframe: use provided bbox
            if bbox_invalid(bbox_det):
                print('INVALID DETECTION.')
                continue

            # update current frame bbox
            bbox_det_dict = {"img_id":curr_img_id,
                            "det_id":det_id,
                            "imgpath": img_path,
                            "bbox":bbox_det}
            self.bbox_dets_list.append(bbox_det_dict)

        # only add one person
        person_reID_add(img_path, self.bbox_dets_list[0]["bbox"], curr_img_id, self.camera_id)
            

    def get_visual_info(self, pose_estimator, timestamp, shelf_id):

        global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
        global pick_results
        global person_reID_list, person_ID
        pick_result = []

        curr_img_id = get_img_id_from_timestamp(timestamp, self.init_timestamp)
        print(curr_img_id)
        
        if curr_img_id >= self.num_imgs:
            print('No more frames.')
            return pick_result

        img_path = self.img_paths[curr_img_id]
        self.bbox_dets_list = []  # keyframe: start from empty
        self.keypoints_list = []  # keyframe: start from empty

        # perform detection at keyframes
        human_candidates = inference_yolov3(img_path)

        num_dets = len(human_candidates)
        print("Get {} detections".format(num_dets))

        # if nothing detected 
        if num_dets <= 0:
            print('No detected person.')
            return pick_result

        # For each candidate, save bbox
        for det_id in range(num_dets):
            # obtain bbox position
            bbox_det = human_candidates[det_id]

            # enlarge bbox by 20% with same center position
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            print(bbox_x1y1x2y2)

            # Keyframe: use provided bbox
            if bbox_invalid(bbox_det):
                print('INVALID DETECTION.')
                continue

            # update current frame bbox
            bbox_det_dict = {"img_id":curr_img_id,
                            "det_id":det_id,
                            "imgpath": img_path,
                            "bbox":bbox_det}
            self.bbox_dets_list.append(bbox_det_dict)

        gondola_num, shelf_num = get_shelf_position(shelf_id)
        shelf_bbox = self.shelf_bbox_list[gondola_num][shelf_num]

        # for each bbox, do pose estimation, check if it is in the shelf bbox
        for bbox_det_dict in self.bbox_dets_list:
            keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]
            
            #get person ID
            person_id = get_track_id_person_reID(img_path, bbox_det_dict["bbox"], curr_img_id, self.camera_id)            
            #pick_result = {'person_id':person_id}

            #check hand location
            img = cv2.imread(img_path)

            # right_hand
            ind_1_r = self.joint_names.index('right elbow')
            ind_2_r = self.joint_names.index('right wrist')
            x1_r, y1_r, sure1_r = keypoints[ind_1_r*3:(ind_1_r+1)*3]
            x2_r, y2_r, sure2_r = keypoints[ind_2_r*3:(ind_2_r+1)*3]
            x3_r = int(x2_r + (x2_r-x1_r) / 4)
            y3_r = int(y2_r + (y2_r-y1_r) / 4)

            # left_hand
            ind_1_l = self.joint_names.index('left elbow')
            ind_2_l = self.joint_names.index('left wrist')
            x1_l, y1_l, sure1_l = keypoints[ind_1_l*3:(ind_1_l+1)*3]
            x2_l, y2_l, sure2_l = keypoints[ind_2_l*3:(ind_2_l+1)*3]
            x3_l = int(x2_l + (x2_l-x1_l) / 4)
            y3_l = int(y2_l + (y2_l-y1_l) / 4)

            if isPoiWithinPoly([x3_r,y3_r], [shelf_bbox]):
                pts = np.array(shelf_bbox,np.int32)
                pts = pts.reshape((-1,1,2))
                
                memory_img_id = curr_img_id-50 #img of 50 frame before
                memory_img_path = img_path[:-7] + str(memory_img_id) + img_path[-4:] 
                if is_image(memory_img_path):    
                    memory_img = cv2.imread(memory_img_path)
                    cropImg = memory_img[y3_r-30:y3_r+30,x3_r-15:x3_r+45]
                else:
                    print("No memory img {}.".format(memory_img_id))
                
                cv2.imwrite('../videos/camera{}-frame{}-person{}r.jpg'.format(self.camera_id, curr_img_id, person_id), cropImg)
                pick_result.append({'person_id':person_id, 'shelf_id':shelf_id, 'img_id':curr_img_id, 'hand':'r', 'hand_pos':[x3_r,y3_r], 'memory_item_img':cropImg})

                print("ID:"+str(person_id)+" picks on shelf "+ str(shelf_id) +" at time "+str(curr_img_id))
        
            if isPoiWithinPoly([x3_l,y3_l], [shelf_bbox]):
                pts = np.array(pixels,np.int32)
                pts = pts.reshape((-1,1,2))
                
                memory_img_id = curr_img_id-50 #img of 50 frame before
                memory_img_path = img_path[:-7] + str(memory_img_id) + img_path[-4:] 
                if is_image(memory_img_path):    
                    memory_img = cv2.imread(memory_img_path)
                    cropImg = memory_img[y3_l-30:y3_l+30,x3_l-15:x3_l+45]
                else:
                    print("No memory img {}.".format(memory_img_id))
                
                cv2.imwrite('../videos/camera{}-frame{}-person{}l.jpg'.format(self.camera_id, curr_img_id, person_id), cropImg)
                pick_result.append({'person_id':person_id, 'shelf_id':shelf_id, 'img_id':curr_img_id, 'hand':'l', 'hand_pos':[x3_l,y3_l], 'memory_item_img':cropImg})

                print("ID:"+str(person_id)+" picks on shelf "+ str(i+1)+ "-" + str(j+1) +" at time "+str(self.img_id))

        return pick_result

def get_shelf_position(shelf_id):
    gondola_id = shelf_id//(12*6)
    shelf_id = (shelf_id%(12*6))//12
    return gondola_id, shelf_id

def get_img_id_from_timestamp(timestamp, init_timestamp):
    return int((timestamp - init_timestamp)*10) #or it may not align well...


def get_track_id_person_reID(img_path, bbox_det, img_id, camera_id):
    global person_reID_list, person_reID_sim_thresh
    x1,y1,x2,y2 = xywh_to_x1y1x2y2(bbox_det)

    img = cv2.imread(img_path )
    cropImg = img[int(y1):int(y2),int(x1):int(x2)]
    cv2.imwrite('../videos/camera{}-frame{}-detect-person.jpg'.format(camera_id, img_id), cropImg)
    
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    cropImg_Tensor = img_transforms(cropImg).unsqueeze(0)
    person_reID_feature = extract_person_feature(cropImg_Tensor)

    # if ADD_PERSON or len(person_reID_list) == 0 :
    #     person_reID_list = np.append(person_reID_list,person_reID_feature, axis=0)
    #     track_id = len(person_reID_list)
    #     return track_id
    
    # if REMOVE_PERSON:
    #     sim_all = np.dot(person_reID_feature, person_reID_list.T)
    #     person_reID_list = np.delete(person_reID_list, np.argmax(sim_all), axis=0)

    sim_all = np.dot(person_reID_feature, person_reID_list.T)
    if max(sim_all[0]) > person_reID_sim_thresh:
        track_id = np.argmax(sim_all[0])+1 
    print(np.argmax(sim_all[0])+1, max(sim_all))

    return track_id

def person_reID_add(img_path, bbox_det, img_id, camera_id):
    global person_reID_list, person_reID_sim_thresh
    x1,y1,x2,y2 = xywh_to_x1y1x2y2(bbox_det)

    img = cv2.imread(img_path )
    cropImg = img[int(y1):int(y2),int(x1):int(x2)]
    cv2.imwrite('../videos/camera{}-frame{}-add-person.jpg'.format(camera_id, img_id), cropImg)

    
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    cropImg_Tensor = img_transforms(cropImg).unsqueeze(0)
    person_reID_feature = extract_person_feature(cropImg_Tensor)
    person_reID_list = np.append(person_reID_list,person_reID_feature, axis=0)
    print(len(person_reID_list), person_reID_list)
    return 


def person_reID_remove(img_path, bbox_det, img_id, camera_id):
    global person_reID_list, person_reID_sim_thresh
    x1,y1,x2,y2 = xywh_to_x1y1x2y2(bbox_det)

    img = cv2.imread(img_path )
    cropImg = img[int(y1):int(y2),int(x1):int(x2)]
    cv2.imwrite('./videos/camera{}-frame{}-remove-person.jpg'.format(camera_id, img_id), cropImg)

    
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    cropImg_Tensor = img_transforms(cropImg).unsqueeze(0)
    person_reID_feature = extract_person_feature(cropImg_Tensor)

    sim_all = np.dot(person_reID_feature, person_reID_list.T)
    person_reID_list = np.delete(person_reID_list, np.argmax(sim_all), axis=0)
    return 





def get_track_id_SGCN(bbox_cur_frame, bbox_list_prev_frame, keypoints_cur_frame, keypoints_list_prev_frame):
    assert(len(bbox_list_prev_frame) == len(keypoints_list_prev_frame))

    min_index = None
    min_matching_score = sys.maxsize
    global pose_matching_threshold
    # if track_id is still not assigned, the person is really missing or track is really lost
    track_id = -1

    for det_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        # check the pose matching score
        keypoints_dict = keypoints_list_prev_frame[det_index]
        keypoints_prev_frame = keypoints_dict["keypoints"]
        pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame)

        if pose_matching_score <= pose_matching_threshold and pose_matching_score <= min_matching_score:
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None
    else:
        track_id = bbox_list_prev_frame[min_index]["track_id"]
        return track_id, min_index


def get_track_id_SpatialConsistency(bbox_cur_frame, bbox_list_prev_frame):
    thresh = 0.3
    max_iou_score = 0
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_cur_frame)
        boxB = xywh_to_x1y1x2y2(bbox_prev_frame)
        iou_score = iou(boxA, boxB)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index

    if max_iou_score > thresh:
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else:
        return -1, None


def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B):
    if keypoints_A == [] or keypoints_B == []:
        print("graph not correctly generated!")
        return sys.maxsize

    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("graph not correctly generated!")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    start = time.time()
    flag_match, dist = pose_matching(data_A, data_B)
    end = time.time()
    return dist



def get_iou_score(bbox_gt, bbox_det):
    boxA = xywh_to_x1y1x2y2(bbox_gt)
    boxB = xywh_to_x1y1x2y2(bbox_det)

    iou_score = iou(boxA, boxB)
    #print("iou_score: ", iou_score)
    return iou_score


def is_target_lost(keypoints, method="max_average"):
    num_keypoints = int(len(keypoints) / 3.0)
    if method == "average":
        # pure average
        score = 0
        for i in range(num_keypoints):
            score += keypoints[3*i + 2]
        score /= num_keypoints*1.0
        print("target_score: {}".format(score))
    elif method == "max_average":
        score_list = keypoints[2::3]
        score_list_sorted = sorted(score_list)
        top_N = 4
        assert(top_N < num_keypoints)
        top_scores = [score_list_sorted[-i] for i in range(1, top_N+1)]
        score = sum(top_scores)/top_N
    if score < 0.6:
        return True
    else:
        return False


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox_from_keypoints(keypoints_python_data):
    if keypoints_python_data == [] or keypoints_python_data == 45*[0]:
        return [0, 0, 2, 2]

    num_keypoints = len(keypoints_python_data)
    x_list = []
    y_list = []
    for keypoint_id in range(int(num_keypoints / 3)):
        x = keypoints_python_data[3 * keypoint_id]
        y = keypoints_python_data[3 * keypoint_id + 1]
        vis = keypoints_python_data[3 * keypoint_id + 2]
        if vis != 0 and vis!= 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    scale = enlarge_scale # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox_in_xywh


def enlarge_bbox(bbox, scale):
    assert(scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x=0
        max_x=2
        min_y=0
        max_y=2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def inference_keypoints(pose_estimator, test_data):
    cls_dets = test_data["bbox"]
    # nms on the bboxes
    if flag_nms is True:
        cls_dets, keep = apply_nms(cls_dets, nms_method, nms_thresh)
        test_data = np.asarray(test_data)[keep]
        if len(keep) == 0:
            return -1
    else:
        test_data = [test_data]

    # crop and detect pose
    pose_heatmaps, details, cls_skeleton, crops, start_id, end_id = get_pose_from_bbox(pose_estimator, test_data, cfg)
    # get keypoint positions from pose
    keypoints = get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id)
    # dump results
    results = prepare_results(test_data[0], keypoints, cls_dets)
    return results


def apply_nms(cls_dets, nms_method, nms_thresh):
    # nms and filter
    keep = np.where((cls_dets[:, 4] >= min_scores) &
                    ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
    cls_dets = cls_dets[keep]
    if len(cls_dets) > 0:
        if nms_method == 'nms':
            keep = gpu_nms(cls_dets, nms_thresh)
        elif nms_method == 'soft':
            keep = cpu_soft_nms(np.ascontiguousarray(cls_dets, dtype=np.float32), method=2)
        else:
            assert False
    cls_dets = cls_dets[keep]
    return cls_dets, keep


def get_pose_from_bbox(pose_estimator, test_data, cfg):
    cls_skeleton = np.zeros((len(test_data), cfg.nr_skeleton, 3))
    crops = np.zeros((len(test_data), 4))

    batch_size = 1
    start_id = 0
    end_id = min(len(test_data), batch_size)

    test_imgs = []
    details = []
    for i in range(start_id, end_id):
        test_img, detail = Preprocessing(test_data[i], stage='test')
        test_imgs.append(test_img)
        details.append(detail)

    details = np.asarray(details)
    feed = test_imgs
    for i in range(end_id - start_id):
        ori_img = test_imgs[i][0].transpose(1, 2, 0)
        if flag_flip == True:
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
    feed = np.vstack(feed)

    res = pose_estimator.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])[0]
    res = res.transpose(0, 3, 1, 2)

    if flag_flip == True:
        for i in range(end_id - start_id):
            fmp = res[end_id - start_id + i].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            res[i] += fmp
            res[i] /= 2

    pose_heatmaps = res
    return pose_heatmaps, details, cls_skeleton, crops, start_id, end_id


def get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id):
    res = pose_heatmaps
    for test_image_id in range(start_id, end_id):
        r0 = res[test_image_id - start_id].copy()
        r0 /= 255.
        r0 += 0.5

        for w in range(cfg.nr_skeleton):
            res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])

        border = 10
        dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
        dr[:, border:-border, border:-border] = res[test_image_id - start_id][:cfg.nr_skeleton].copy()

        for w in range(cfg.nr_skeleton):
            dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)

        for w in range(cfg.nr_skeleton):
            lb = dr[w].argmax()
            y, x = np.unravel_index(lb, dr[w].shape)
            dr[w, y, x] = 0
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.output_shape[1] - 1))
            y = max(0, min(y, cfg.output_shape[0] - 1))
            cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
            cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

        # map back to original images
        crops[test_image_id, :] = details[test_image_id - start_id, :]
        for w in range(cfg.nr_skeleton):
            cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[1] * (crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
            cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[0] * (crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]
    return cls_skeleton


def prepare_results(test_data, cls_skeleton, cls_dets):
    cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)

    cls_scores = 1
    dump_results = []
    cls_skeleton = np.concatenate(
        [cls_skeleton.reshape(-1, cfg.nr_skeleton * 3), (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
        axis=1)
    for i in range(len(cls_skeleton)):
        result = dict(image_id=test_data['img_id'],
                      category_id=1,
                      score=float(round(cls_skeleton[i][-1], 4)),
                      keypoints=cls_skeleton[i][:-1].round(3).tolist())
        dump_results.append(result)
    return dump_results


def is_keyframe(img_id, interval=10):
    if img_id % interval == 0:
        return True
    else:
        return False


def pose_to_standard_mot(keypoints_list_list, dets_list_list):
    openSVAI_python_data_list = []

    num_keypoints_list = len(keypoints_list_list)
    num_dets_list = len(dets_list_list)
    assert(num_keypoints_list == num_dets_list)

    for i in range(num_dets_list):

        dets_list = dets_list_list[i]
        keypoints_list = keypoints_list_list[i]

        if dets_list == []:
            continue
        img_path = dets_list[0]["imgpath"]
        img_folder_path = os.path.dirname(img_path)
        img_name =  os.path.basename(img_path)
        img_info = {"folder": img_folder_path,
                    "name": img_name,
                    "id": [int(i)]}
        openSVAI_python_data = {"image":[], "candidates":[]}
        openSVAI_python_data["image"] = img_info

        num_dets = len(dets_list)
        num_keypoints = len(keypoints_list) #number of persons, not number of keypoints for each person
        candidate_list = []

        for j in range(num_dets):
            keypoints_dict = keypoints_list[j]
            dets_dict = dets_list[j]

            img_id = keypoints_dict["img_id"]
            det_id = keypoints_dict["det_id"]
            track_id = keypoints_dict["track_id"]
            img_path = keypoints_dict["imgpath"]

            bbox_dets_data = dets_list[det_id]
            det = dets_dict["bbox"]
            if  det == [0, 0, 2, 2]:
                # do not provide keypoints
                candidate = {"det_bbox": [0, 0, 2, 2],
                             "det_score": 0}
            else:
                bbox_in_xywh = det[0:4]
                keypoints = keypoints_dict["keypoints"]

                track_score = sum(keypoints[2::3])/len(keypoints)/3.0

                candidate = {"det_bbox": bbox_in_xywh,
                             "det_score": 1,
                             "track_id": track_id,
                             "track_score": track_score,
                             "pose_keypoints_2d": keypoints}
            candidate_list.append(candidate)
        openSVAI_python_data["candidates"] = candidate_list
        openSVAI_python_data_list.append(openSVAI_python_data)
    return openSVAI_python_data_list


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat] # check line intersection
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

def isPoiWithinPoly(poi,poly): # check point locate in which box
    sinsc=0 
    for epoly in poly: 
        for i in range(len(epoly)): 
            s_poi=epoly[i%len(epoly)]
            e_poi=epoly[(i+1)%len(epoly)]
            if isRayIntersectsSegment(poi,s_poi,e_poi):
                sinsc+=1 

    return True if sinsc%2==1 else  False



if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', '-v', type=str, dest='video_dir_path', default='videos')
    parser.add_argument('--model', '-m', type=str, dest='test_model', default="weights/mobile-deconv/snapshot_296.ckpt")
    args = parser.parse_args()
    args.bbox_thresh = 0.4

    # initialize pose estimator
    initialize_parameters()
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights(args.test_model)

    # init camera
    enter_camera_index = 4
    exit_camera_index = 3

    gondola_to_camera_map = {1:3, 2:3, 3:5, 4:1, 5:4, 5:3}
    gondola_bbox_list = [[],[],[],[[[[706, 342],[707, 313],[823, 463],[817, 494]],
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
                    [[ 896,  221],[ 924,  115],[1248,  263],[1154,  392]]]],[],[],[],[]]
    camera_instance_list = []
    for i in range(8): #8 cameras
        image_folder = args.video_dir_path + '/' + str(i) + '/'
        camera_instance_list.append(Vision_module(i, image_folder, gondola_bbox_list[i]))


    Weight_sensor_number = 360

    detected_weight_event_queue = [queue.Queue(0) for kk in
                                   range(Weight_sensor_number)]  # event sotor queue of each sensor
    total_detected_queue = queue.Queue(0)  # number changed_weight timestamp #total queue of detected event
    enter_signal_queue = queue.Queue(0) 
    exit_signal_queue = queue.Queue(0)  



    weight_value_list = [[] for jj in range(Weight_sensor_number)]
    timestamp_list = [[] for jj in range(Weight_sensor_number)]
    weight_sensor_list = []
    count = 0
    # init weight sensor
    for gn in range(5):  # gondola
        for sn in range(6):  # shelf
            for pn in range(12):  # plate
                tmp_gn = gn + 1
                tmp_sn = sn + 1
                tmp_pn = pn + 1

                timestamp, weight_value = cfr(tmp_gn, tmp_sn, tmp_pn)
                sensor_number = (tmp_gn - 1) * 6 * 12 + (tmp_sn - 1) * 12 + tmp_pn
                weight_value_list[sensor_number - 1].extend(list(weight_value))
                timestamp_list[sensor_number - 1].extend(list(timestamp))

                item_dict = {'1': [10, 10, 2]}  # {'item_number': [item_total_number, weight, standard deviation]}
                if len(weight_value) > 20:
                    initial_val = weight_value[0:20]
                    initial_ts = timestamp[0:20]
                else:
                    initial_val = np.arange(0, 0.2, 0.01)
                    initial_ts = np.arange(0, 200, 10)

                count = count + 1
                weight_sensor_list.append(WeightSensor(sensor_number, item_dict, initial_val, initial_ts))
            
    
    print(len(weight_value))
    for time_coun in np.arange(20, len(weight_value)):
        signal = np.zeros(8)
        for sensor_num in range(Weight_sensor_number):
            
            if time_coun < len(weight_value_list[sensor_num]) - 1:
                update_wv = np.array(weight_value_list[sensor_num][time_coun:time_coun + 1])
                update_ts = np.array(timestamp_list[sensor_num][time_coun: time_coun + 1])
                weight_sensor_list[sensor_num].value_update(total_detected_queue, detected_weight_event_queue,
                                                            update_wv, update_ts)
                #print('load data')

        if time_coun == 21:
            enter_signal_queue.put('1')
            print('Add one person')

        # time.sleep(0.1)

        tmp_st = WeightSensor.weight_change_value[124]  # sensor 2 5 5
        # print(WeightSensor.weight_change_value[sensor_number])
        while not total_detected_queue.empty():
            info = total_detected_queue.get()
            print(info)
            weight_sensor_index = info[0]
            gondola_index = (weight_sensor_index // (6*12)) + 1    
            camera_index = gondola_to_camera_map[gondola_index]
            signal[camera_index] = 1 
            print('\n')

        while not enter_signal_queue.empty():
            enter_signal_queue.get()
            print('CUSTOMER IS COMING IN.')
            signal[enter_camera_index] = 2

        while not exit_signal_queue.empty():
            exit_signal_queue.get()
            print('CUSTOMER IS LEAVING.')
            signal[exit_camera_index] = 3

        # update camera frames    
        print(time_coun)
        for i in range(8):
            camera_instance_list[i].update_frame(pose_estimator, signal[i])
