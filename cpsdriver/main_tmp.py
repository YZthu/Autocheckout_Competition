import queue
import numpy as np
import time

from weight_main import WeightSensor
from weight_main import weight_based_item_estimate

from vision_main import *

from collections import defaultdict

import sys
import logging
import pdb

#from cpsdriver.clients import (
from clients import (
    CpsMongoClient,
    CpsApiClient,
    TestCaseClient,
)
from cli import parse_configs
from log import setup_logger

from options import cpsdriver_args
print (cpsdriver_args)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



def main(args=None):
    args = parse_configs(args)
    setup_logger(args.log_level)
    mongo_client = CpsMongoClient(args.db_address)
    api_client = CpsApiClient()
    test_client = TestCaseClient(mongo_client, api_client)
    #test_client.load(f"{args.command}-{args.sample}")
    logger.info(f"Available Test Cases are {test_client.available_test_cases}")
    test_client.set_context(args.command, load=False)
    generate_receipts(test_client, args.command)

def load_product_locations(test_client,Weight_sensor_number):
    productList = test_client.list_products()
    out_sensor_product_info = []
    weight_sensor_info = [[] for jj in range(Weight_sensor_number)]
    for aProduct in productList:
        item_info = [aProduct.name, aProduct.weight, aProduct.price]
        allFacings = test_client.find_product_facings(aProduct.product_id)
        if len(allFacings) == 0:
            out_sensor_product_info.append(item_info)
            continue
        for aFacing in allFacings:
            for plateLoc in aFacing.plate_ids:
                sensor_number = (plateLoc.gondola_id - 1) * 7 * 15 + (plateLoc.shelf_index- 1) * 15 + plateLoc.plate_index -1
                weight_sensor_info[sensor_number].append(item_info)
    return weight_sensor_info, out_sensor_product_info

def get_sensor_batch(test_client, start_time, batch_length, Weight_sensor_number):
    DOWNFACTOR=1
    if start_time <= 0:
        # the first time, we don't know when the timestamps start, so let's find out
        first_data = test_client.find_first_after_time("plate_data",0.0)[0]
        start_time = first_data.timestamp

    batch_data = test_client.find_all_between_time("plate_data", start_time, start_time+batch_length)
    batch_size = len(batch_data)
    if batch_size == 0:
        return None, -1
    weight_update_data = [np.empty((0,2)) for jj in range(Weight_sensor_number)]
    currentTime = start_time
    # only take up to a multiple of 10 samples so we can downsample to 6Hz

    for rawData in batch_data:
        currentTime = rawData.timestamp
        startShelf = rawData.plate_id.shelf_index
        startPlate = rawData.plate_id.plate_index
        gondolaId = rawData.plate_id.gondola_id
        dataSize = rawData.data.shape
        nSamples = dataSize[0]
        nShelves = dataSize[1]
        nPlates = dataSize[2]
        if nShelves > 7:
            new_shelve = 7
        else:
            new_shelve = nShelves
        if nPlates > 15:
            new_plate = 15
        else:
            new_plate = nPlates

        ts = np.array(range(nSamples))*(1.0/60) + currentTime # the timestamps in this packet
        ts = ts.reshape((nSamples,1))
        for jj in range(new_shelve-1):
            for kk in range(new_plate -1):
                weightData = (rawData.data[:,jj,kk]).reshape(nSamples,1)
                if not(np.isnan(weightData).all()):
                    sensor_number = (gondolaId - 1) * 7 * 15 + (startShelf+jj) * 15 + startPlate + kk
                    updateData = np.hstack((ts,weightData))
                    prevData = weight_update_data[sensor_number]
                    weight_update_data[sensor_number] = np.vstack((prevData, updateData))
                    if sensor_number == 244:
                        pass
                        #print(weightData)
    return weight_update_data, currentTime   
    # down sampled
    """
    for nn in range(Weight_sensor_number):
        # we might lose a few samples but doing it right would involve generators
        d = weight_update_data[nn]
        newLength = DOWNFACTOR*(d.shape[0]//DOWNFACTOR)

        d = d[0:newLength,:]
        meanTs = np.mean(d[:,0].reshape(-1,DOWNFACTOR), axis=1)
        meanW = np.mean(d[:,1].reshape(-1,DOWNFACTOR),axis=1)
        weight_update_data[nn] = np.hstack((meanTs.reshape(-1,1), meanW.reshape(-1,1)))
    return weight_update_data, currentTime            
    """
def customer_shopping_list_update(current_customer_shopping_list, changed_weight, changed_item_info):
    new_shopping_list =[]

    item_name = changed_item_info[0]
    item_number = changed_item_info[1]
    item_price = changed_item_info[2]
    item_per_weight = changed_item_info[3]

    if (changed_weight > 5) & (item_number == 0):
        # changed weight >0 but customer shoping list is empty
        new_shopping_list = current_customer_shopping_list
        return new_shopping_list

    if (changed_weight > 5) & (item_number > 0):
        #return
        for tmp_kk in range(current_customer_shopping_list):
            current_shopping_list_item_info = current_customer_shopping_list[tmp_kk]
            current_item_name = current_shopping_list_item_info[0]
            current_item_number = current_shopping_list_item_info[1]
            current_item_price = current_shopping_list_item_info[2]
            current_item_per_weight = current_shopping_list_item_info[3]

            if current_item_name == item_name:
                new_current_item_number = current_item_number - item_number
                if new_current_item_number > 0:
                    new_current_shopping_list_item_info =[current_item_name, new_current_item_number, current_item_price, current_item_per_weight]
                    new_shopping_list.append(new_current_shopping_list_item_info)
            else:
                new_shopping_list.append(current_shopping_list_item_info)
        return new_shopping_list
    else:
        shopping_list_length = len(current_customer_shopping_list)
        if shopping_list_length < 1:
            #empty shoppng list
            new_shopping_list.append(changed_item_info)
            
        if (changed_weight < -5) & (shopping_list_length > 0):
            #pick up
            item_write_flag = False

            for tmp_kk in range(shopping_list_length):
                current_shopping_list_item_info = current_customer_shopping_list[tmp_kk]
                current_item_name = current_shopping_list_item_info[0]
                current_item_number = current_shopping_list_item_info[1]
                current_item_price = current_shopping_list_item_info[2]
                current_item_per_weight = current_shopping_list_item_info[3]

                if current_item_name == item_name:
                    new_current_item_number = current_item_number + item_number
                    new_current_shopping_list_item_info =[current_item_name, new_current_item_number, current_item_price, current_item_per_weight]
                    new_shopping_list.append(new_current_shopping_list_item_info)
                    item_write_flag = True
                else:
                    new_shopping_list.append(current_shopping_list_item_info)
            if item_write_flag == False:
                # new item for this customer
                new_shopping_list.append(changed_item_info)
        return new_shopping_list
    #return current_customer_shopping_list

def return_weight_sensor_item_updata(sensor_number_list, changed_item_info, sensor_info):
    item_name = changed_item_info[0]
    item_number = changed_item_info[1]
    item_price = changed_item_info[2]
    item_per_weight = changed_item_info[3]

    new_item_info = [item_name, item_per_weight, item_price]

    if len(sensor_number_list) > 0:
        for tmp_sensor_num in range(len(sensor_number_list)):
            tmp_sensor_number = sensor_number_list[tmp_sensor_num]
            new_sensor_item_info = sensor_info[tmp_sensor_number]
            exist_flag = False
            if len(new_sensor_item_info) > 0:
                for old_item_num in range(len(new_sensor_item_info)):
                    old_item_info = new_sensor_item_info[old_item_num]
                    old_item_name = old_item_info[0]
                    if old_item_name == item_name:
                        #return to where it taked
                        exist_flag = True
                        break
            if exist_flag == False:
                #retuen to another plate
                new_sensor_item_info.append(new_item_info)
                sensor_info[tmp_sensor_number] = new_sensor_item_info
    return sensor_info

def print_receipt(customer_id, current_shopping_list):
    
    print('###********  BestTeamEver store receipt  ********###')
    print('customer ID is: %d '%customer_id)
    total_money = 0
    print('------------------------------------------------------')
    if len(current_shopping_list) > 0:
        print('name                     * price * number * total')
        for kk in range(len(current_shopping_list)):
            tmp_item = current_shopping_list[kk]
            print(tmp_item)
            print('------------------------------------------------------')
            item_name = tmp_item[0]
            if len(item_name) > 26:
                print(item_name[0:26],end="")
            else:
                need_len = 26 - len(item_name)
                print(item_name,end="")
                print(' '*need_len, end="")
            print(''.rjust(5, ' '),end="")
            item_price = tmp_item[2]
            print(item_price,end="")
            print(''.rjust(5, ' '),end="")
            print(tmp_item[1],end="")
            print(''.rjust(5, ' '),end="")
            this_item_money = tmp_item[1]*tmp_item[2]
            print(this_item_money)
            total_money = total_money + this_item_money
    print('------------------------------------------------------')
    print('Total  -------------------------------------  %f'%(total_money))
    print('---------------------------end------------------------')

def generate_receipts(test_client, case_name):
    Weight_sensor_number = 630

    detected_weight_event_queue = [queue.Queue(0) for kk in range(Weight_sensor_number)]  # event sotor queue of each sensor
    total_detected_queue = queue.Queue(0)  # number changed_weight timestamp #total queue of detected event
    merged_detected_queue = queue.Queue(0)
    enter_signal_queue = queue.Queue(0) 
    exit_signal_queue = queue.Queue(0)  
    customer_shopping_list = defaultdict(list)
    nCustomers = 0
    #ground truth data read
    sensor_info, out_info = load_product_locations(test_client, Weight_sensor_number)
    #### VISION STUFF ####
    person_ID = 0
    initialize_parameters()
    print(person_ID)
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights("weights/mobile-deconv/snapshot_296.ckpt")
    enter_camera_index = 4
    exit_camera_index = 3
    
    # video_path = '../videos/'+case_name+'/'
    # video_file_names = get_immediate_childfile_names(video_path)
    # if case_name == 'cps-test-01':
    #     video_fps = 20
    #     gondola_bbox_list = [[[[706, 342],[707, 313],[823, 463],[817, 494]],
    #                     [[709, 308],[713, 275],[833, 416],[825, 454]],
    #                     [[713, 266],[716, 226],[849, 365],[838, 408]],
    #                     [[719, 221],[722, 172],[868, 297],[853, 357]],
    #                     [[723, 170],[728, 107],[888, 214],[875, 288]],
    #                     [[729, 104],[738,  20],[934, 110],[887, 213]]],
    #                     [[[ 818,  496],[ 827,  459],[ 987,  661],[ 964,  698]],
    #                     [[ 831,  450],[ 839,  415],[1015,  614],[ 994,  650]],
    #                     [[ 846,  407],[ 851,  368],[1044,  560],[1023,  606]],
    #                     [[ 858,  356],[ 869,  299],[1088,  484],[1052,  551]],
    #                     [[ 877,  290],[ 889,  231],[1141,  405],[1098,  474]],
    #                     [[ 896,  221],[ 924,  115],[1248,  263],[1154,  392]]]]
    # else:
    #     video_fps = 10 
    #     gondola_bbox_list = [[[[1310,  617],[1566, 1004],[1499, 1123],[1275,  741]],
    #                     [[1333,  533],[1601,  901],[1536, 1041],[1299,  672]],
    #                     [[1355,  449],[1641,  808],[1572,  950],[1316,  586]],
    #                     [[1370,  341],[1693,  668],[1624,  860],[1329,  494]],
    #                     [[1396,  212],[1779,  517],[1670,  720],[1351,  373]],
    #                     [[1424,   84],[1911,  365],[1814,  591],[1377,  240]]],

    #                     [[[1534,  968],[1820, 1396],[1706, 1493],[1463, 1112]],
    #                     [[1573,  877],[1880, 1306],[1771, 1424],[1512, 1015]],
    #                     [[1609,  784],[1937, 1219],[1840, 1347],[1545,  926]],
    #                     [[1654,  657],[2061, 1099],[1926, 1258],[1590,  832]],
    #                     [[1708,  494],[2214,  950],[2042, 1163],[1620,  696]],
    #                     [[1782,  367],[2365,  793],[2130,  989],[1674,  545]]],

    #                     [[[1118, 1316],[1469,  870],[1510,  996],[1191, 1445]],
    #                     [[1058, 1198],[1450,  752],[1491,  920],[1146, 1353]],
    #                     [[1008, 1082],[1439,  640],[1478,  799],[1097, 1237]],
    #                     [[ 957,  974],[1415,  550],[1448,  696],[1041, 1123]],
    #                     [[ 800,  795],[1349,  401],[1422,  593],[ 982, 1010]]],

    #                     [[[1452,  892],[1639,  657],[1657,  774],[1491, 1010]],
    #                     [[1422,  774],[1631,  560],[1646,  700],[1467,  918]],
    #                     [[1387,  681],[1611,  461],[1641,  597],[1430,  836]],
    #                     [[1375,  556],[1603,  365],[1626,  500],[1424,  705]],
    #                     [[1232,  455],[1594,  272],[1609,  414],[1368,  631]],
    #                     [[1176,  272],[1568,  158],[1594,  349],[1262,  511]]]]
    
    #for file_name in video_file_names: #generate images from videos
    #    print(file_name)
    #    if file_name[-3:]=='mp4':
    #        video_to_images(video_path + file_name)         

    gondola_to_camera_map = {0:3, 1:3, 2:5, 3:5} #update here with new video names
    camera_instance_list = []

    # people get into store
    current_customer_id = 0
    #nCustomers = 1


    weight_sensor_list = [WeightSensor(jj+1, {'1':[10,10,2]}, np.array([]), np.array([])) for jj in range(Weight_sensor_number)]
    
    buffer_info = []
    pre_system_time = 0
    pre_timestamp = 0
    pre_sensor_num = 0
    moreData, next_time = get_sensor_batch(test_client, -1, 2.0, Weight_sensor_number)
    
    tmp_data = moreData[48]
    tmp_data_ts = tmp_data[:,0]
    initial_timestamp = tmp_data_ts[0]
    # for i in range(len(video_file_names)): #add cameras
    #     image_folder = video_path + str(i) + '/'
    #     camera_instance_list.append(Vision_module(i, image_folder, gondola_bbox_list, initial_timestamp, video_fps))

    personEntered = False # TODO: replace with results from target
    while moreData is not None:
        for sensor_number in range(Weight_sensor_number):
            update_data = moreData[sensor_number]
            
            if update_data.shape[0] == 0:
                continue # no data loaded from the batch
            update_wv = update_data[:,1]
            update_ts = update_data[:,0]
            weight_sensor_list[sensor_number].value_update(total_detected_queue, detected_weight_event_queue, update_wv, update_ts)
            

        moreData,next_time = get_sensor_batch(test_client, next_time, 2, Weight_sensor_number)
        ### VISION STUFF ###
        camera_signal = np.zeros(8)
        if not personEntered:
            enter_signal_queue.put(update_ts[0])
            print('Add one person')
            personEntered = True

        # while not enter_signal_queue.empty():
        #     enter_timestamp = enter_signal_queue.get()
        #     print('CUSTOMER IS COMING IN.')
        #     nCustomers += 1
        #     camera_instance_list[enter_camera_index].add_person_feature(pose_estimator, enter_timestamp)

        while not total_detected_queue.empty():
            tmp_info = total_detected_queue.get()
            tmp_timestamp = tmp_info[2]
            tmp_weight_sensor_index = tmp_info[0]
            

            new_event = False
            if (abs(pre_timestamp - tmp_timestamp) > 2) | (abs(pre_sensor_num- tmp_weight_sensor_index) > 4):
                new_event = True
            if new_event:
                if len(buffer_info) > 0:
                    merged_detected_queue.put(buffer_info)
                    print('merged it')
                    buffer_info = []
            if len(buffer_info) < 1:
                pre_system_time = update_ts[1]
            buffer_info.append(tmp_info)
            pre_timestamp = tmp_timestamp
            pre_sensor_num = tmp_weight_sensor_index
            total_detected_queue.task_done()
            print(new_event)


        now_time = update_ts[1]
        
        if now_time - pre_system_time > 2:
            if len(buffer_info)>0:
                #print(now_time - pre_system_time)
                merged_detected_queue.put(buffer_info)
                print('merged it')
                buffer_info = []
            pre_system_time = update_ts[1]
            
            
        while not merged_detected_queue.empty():
            detected_event = merged_detected_queue.get()

            # add vision info here
            # tmp_weight_sensor = detected_event[0]
            # tmp_weight_sensor_index = tmp_weight_sensor[0]
            # tmp_timestamp = tmp_weight_sensor[2]
            # gondola_num, shelf_num = get_shelf_position(tmp_weight_sensor_index)
            # camera_index = gondola_to_camera_map[gondola_num]     
            # vision_info = camera_instance_list[camera_index].get_visual_info(pose_estimator, tmp_timestamp, tmp_weight_sensor_index)
            # print('vision info')
            # print(vision_info)
            # print('detected change') 
            # print(tmp_info)
            # if len(vision_info) < 1:
            #     pass
            #     #continue
            # if len(vision_info) > 0:
            #     for people_num in range(len(vision_info)):
            #         tmp_people_info = vision_info[people_num]
            #         people_id = tmp_people_info['person_id']
            # else:
            #     people_id = 0
            # current_customer_id = people_id-1

            weight_sensor_index = detected_event[0][0]
            print('in the merged queue')
            print(detected_event)
            sensor_number_list =[]
            total_changed_weight = 0
            event_timestamp =0
            for kk in range(len(detected_event)):
                sub_event = detected_event[kk]
                sensor_number_list.append(sub_event[0])
                total_changed_weight = total_changed_weight + sub_event[1]
                event_timestamp = sub_event[2]
            
            current_shopping_list = customer_shopping_list[current_customer_id]
            item_fin_name, item_fin_number, item_fin_price,  item_per_weight  = weight_based_item_estimate(current_shopping_list, sensor_number_list, total_changed_weight, sensor_info, out_info)
            changed_item_info = [item_fin_name, item_fin_number, item_fin_price,  item_per_weight]
            print('detected changed item info')
            print(total_changed_weight)
            print(changed_item_info)
            if item_fin_number < 0.5:
                continue

            new_list = customer_shopping_list_update(current_shopping_list, total_changed_weight, changed_item_info)
            #print(new_list)
            customer_shopping_list[current_customer_id] = new_list
            if total_changed_weight > 5:
                #return_weight_sensor_item_updata(sensor_number_list, changed_item_info)
                pass
            if len(customer_shopping_list)> 0:
                for (customer_id,  shopping_list) in customer_shopping_list.items():
                    print_receipt(customer_id, shopping_list)
    for kk in range(len(weight_sensor_list)):
        if len(weight_sensor_list[kk].value) > 100:
            print(kk+1)
            
if __name__ == "__main__":
    main(cpsdriver_args)
