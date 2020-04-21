import queue
import numpy as np
import time
import threading

from cvs_data_read import csv_file_read as cfr
from cvs_data_read import ground_truth_read as gtr

from weight_main import WeightSensor
from weight_main import weight_based_item_estimate

from vision_main import *

lock = threading.Lock()

Weight_sensor_number = 360
customer_shopping_list =[]
#each shopping list formate  item_fin_name, item_fin_number, item_fin_price,  item_per_weight
#ground truth data read
sensor_info, out_info = gtr('test1', Weight_sensor_number)


def allocate_customer_id():
    #if some people getinto the store, use the function to allocate the customer id
    global customer_shopping_list
    length = len(customer_shopping_list)
    customer_shopping_list.append([])
    return length

def customer_shopping_list_update(customer_id, changed_weight, changed_item_info):
    global customer_shopping_list
    new_shopping_list =[]

    item_name = changed_item_info[0]
    item_number = changed_item_info[1]
    item_price = changed_item_info[2]
    item_per_weight = changed_item_info[3]
    current_customer_shopping_list = customer_shopping_list[customer_id]

    if (changed_weight > 5) & (item_number == 0):
        # changed weight >0 but customer shoping list is empty
        new_shopping_list = customer_shopping_list[customer_id]

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
                    new_current_shopping_list_item_info =[current_item_number, new_current_item_number, current_item_price, current_item_per_weight]
                    new_shopping_list.append(new_current_shopping_list_item_info)
            else:
                new_shopping_list.append(current_shopping_list_item_info)
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
                    new_current_shopping_list_item_info =[current_item_number, new_current_item_number, current_item_price, current_item_per_weight]
                    new_shopping_list.append(new_current_shopping_list_item_info)
                    item_write_flag = True
                else:
                    new_shopping_list.append(current_shopping_list_item_info)
            if item_write_flag == False:
                # new item for this customer
                new_shopping_list.append(changed_item_info)
    #update the shopping list
    lock.acquire()
    try:
        customer_shopping_list[customer_id] = new_shopping_list
    finally:
        lock.release()

def return_weight_sensor_item_updata(sensor_number_list, changed_item_info):
    item_name = changed_item_info[0]
    item_number = changed_item_info[1]
    item_price = changed_item_info[2]
    item_per_weight = changed_item_info[3]

    global sensor_info

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
                new_sensor_item_info.extend(new_item_info)
                sensor_info[tmp_sensor_number] = new_sensor_item_info
            


def print_receipt(customer_id):
    global customer_shopping_list
    print('###********  BestTeamEver store receipt  ********###')
    print('customer ID is: %d '%customer_id)
    current_shopping_list = customer_shopping_list[customer_id]
    total_money = 0
    print('------------------------------------------------------')
    if len(current_shopping_list) > 0:
        print('name                     * price * number * total')
        for kk in range(len(current_shopping_list)):
            tmp_item = current_shopping_list[kk]
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

if __name__ == "__main__":
    
    #global Weight_sensor_number

    detected_weight_event_queue = [queue.Queue(0) for kk in
                                   range(Weight_sensor_number)]  # event sotor queue of each sensor
    total_detected_queue = queue.Queue(0)  # number changed_weight timestamp #total queue of detected event
    merged_detected_queue = queue.Queue(0)
    enter_signal_queue = queue.Queue(0) 
    exit_signal_queue = queue.Queue(0)  


    # init camera
    global person_ID
    person_ID=0
    initialize_parameters()
    print(person_ID)
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights("weights/mobile-deconv/snapshot_296.ckpt")
    enter_camera_index = 4
    exit_camera_index = 3
    initial_timestamp = 1580250245.19951 - 1/20 

    gondola_to_camera_map = {1:103, 2:103, 3:105, 4:105}
    gondola_bbox_list = [[[[1310,  617],[1566, 1004],[1499, 1123],[1275,  741]],
                        [[1333,  533],[1601,  901],[1536, 1041],[1299,  672]],
                        [[1355,  449],[1641,  808],[1572,  950],[1316,  586]],
                        [[1370,  341],[1693,  668],[1624,  860],[1329,  494]],
                        [[1396,  212],[1779,  517],[1670,  720],[1351,  373]],
                        [[1424,   84],[1911,  365],[1814,  591],[1377,  240]]],

                        [[[1534,  968],[1820, 1396],[1706, 1493],[1463, 1112]],
                        [[1573,  877],[1880, 1306],[1771, 1424],[1512, 1015]],
                        [[1609,  784],[1937, 1219],[1840, 1347],[1545,  926]],
                        [[1654,  657],[2061, 1099],[1926, 1258],[1590,  832]],
                        [[1708,  494],[2214,  950],[2042, 1163],[1620,  696]],
                        [[1782,  367],[2365,  793],[2130,  989],[1674,  545]]]

                        [[[1118, 1316],[1469,  870],[1510,  996],[1191, 1445]],
                        [[1058, 1198],[1450,  752],[1491,  920],[1146, 1353]],
                        [[1008, 1082],[1439,  640],[1478,  799],[1097, 1237]],
                        [[ 957,  974],[1415,  550],[1448,  696],[1041, 1123]],
                        [[ 800,  795],[1349,  401],[1422,  593],[ 982, 1010]]]

                        [[[1452,  892],[1639,  657],[1657,  774],[1491, 1010]],
                        [[1422,  774],[1631,  560],[1646,  700],[1467,  918]],
                        [[1387,  681],[1611,  461],[1641,  597],[1430,  836]],
                        [[1375,  556],[1603,  365],[1626,  500],[1424,  705]],
                        [[1232,  455],[1594,  272],[1609,  414],[1368,  631]],
                        [[1176,  272],[1568,  158],[1594,  349],[1262,  511]]]]
    # test-case-01
    # gondola_bbox_list = [[],[],[],[[[[706, 342],[707, 313],[823, 463],[817, 494]],
    #                 [[709, 308],[713, 275],[833, 416],[825, 454]],
    #                 [[713, 266],[716, 226],[849, 365],[838, 408]],
    #                 [[719, 221],[722, 172],[868, 297],[853, 357]],
    #                 [[723, 170],[728, 107],[888, 214],[875, 288]],
    #                 [[729, 104],[738,  20],[934, 110],[887, 213]]],
    #                 [[[ 818,  496],[ 827,  459],[ 987,  661],[ 964,  698]],
    #                 [[ 831,  450],[ 839,  415],[1015,  614],[ 994,  650]],
    #                 [[ 846,  407],[ 851,  368],[1044,  560],[1023,  606]],
    #                 [[ 858,  356],[ 869,  299],[1088,  484],[1052,  551]],
    #                 [[ 877,  290],[ 889,  231],[1141,  405],[1098,  474]],
    #                 [[ 896,  221],[ 924,  115],[1248,  263],[1154,  392]]]],[],[],[],[]]
    camera_instance_list = []
    for i in range(8): #8 cameras
        image_folder = '../cps-test-videos' + '/' + str(i) + '/'
        camera_instance_list.append(Vision_module(i, image_folder, gondola_bbox_list[i], initial_timestamp))

    # people get into store
    current_customer_id = allocate_customer_id()

    # init weight sensor
    weight_value_list = [[] for jj in range(Weight_sensor_number)]
    timestamp_list = [[] for jj in range(Weight_sensor_number)]
    weight_sensor_list = []
    count = 0
    test_name ='test1'
    for gn in range(5):  # gondola
        for sn in range(6):  # shelf
            for pn in range(12):  # plate
                tmp_gn = gn + 1
                tmp_sn = sn + 1
                tmp_pn = pn + 1

                timestamp, weight_value = cfr(test_name, tmp_gn, tmp_sn, tmp_pn)
                sensor_number = (tmp_gn - 1) * 6 * 12 + (tmp_sn - 1) * 12 + tmp_pn
                weight_value_list[sensor_number - 1].extend(list(weight_value))
                timestamp_list[sensor_number - 1].extend(list(timestamp))

                item_dict = {'1': [10, 10, 2]}  # {'item_number': [item_total_number, weight, standard deviation]}
                if len(weight_value) > 50:
                    initial_val = weight_value[0:50]
                    initial_ts = timestamp[0:50]
                else:
                    initial_val = np.arange(0, 0.5, 0.01)
                    initial_ts = np.arange(0, 500, 10)

                count = count + 1
                weight_sensor_list.append(WeightSensor(sensor_number, item_dict, initial_val, initial_ts))
    

    
    pre_timestamp = 0
    buffer_info = []
    pre_system_time = time.time()
    for time_coun in np.arange(50, len(weight_value) - 10, 10):
        for sensor_num in range(Weight_sensor_number):

            if time_coun < len(weight_value_list[sensor_num]) - 10:
                update_wv = np.array(weight_value_list[sensor_num][time_coun:time_coun + 10])
                update_ts = np.array(timestamp_list[sensor_num][time_coun: time_coun + 10])
                weight_sensor_list[sensor_num].value_update(total_detected_queue, detected_weight_event_queue,
                                                            update_wv, update_ts)
        
        camera_signal = np.zeros(8)

        if time_coun == 50:
            enter_signal_queue.put(update_ts[0])
            print('Add one person.')

        time.sleep(0.1)

        tmp_st = WeightSensor.weight_change_value[124]  # sensor 2 5 5
        # print(WeightSensor.weight_change_value[sensor_number])


        while not enter_signal_queue.empty():
            enter_timestamp = enter_signal_queue.get()
            print('CUSTOMER IS COMING IN.')
            allocate_customer_id()
            camera_instance_list[enter_camera_index].add_person_feature(pose_estimator, enter_timestamp)

        # while not exit_signal_queue.empty():
        #     exit_signal_queue.get()
        #     print('CUSTOMER IS LEAVING.')
        #     camera_signal[exit_camera_index] = 3

        while not total_detected_queue.empty():
            tmp_info = total_detected_queue.get()
            tmp_timestamp = tmp_info[2]
            tmp_weight_sensor_index = tmp_info[0]
            print(tmp_info)

            # add vision info here
            gondola_num, shelf_num = get_shelf_position(tmp_weight_sensor_index)
            camera_index = gondola_to_camera_map[gondola_num+1]     
            vision_info = camera_instance_list[camera_index].get_visual_info(pose_estimator, tmp_timestamp, tmp_weight_sensor_index)
            print(len(vision_info))
            
            
            new_event = False
            if abs(pre_timestamp - tmp_timestamp) > 2:
                new_event = True
            if new_event:
                if len(buffer_info) > 0:
                    merged_detected_queue.put(buffer_info)
                    buffer_info = []
            if len(buffer_info) < 1:
                pre_system_time = time.time()
            buffer_info.append(tmp_info)
            pre_timestamp = tmp_timestamp
            
        now_time = time.time()
        
        if now_time - pre_system_time > 1:
            if len(buffer_info)>0:
                #print(now_time - pre_system_time)
                merged_detected_queue.put(buffer_info)
                buffer_info = []
            pre_system_time = time.time()
            
        while not merged_detected_queue.empty():
            detected_event = merged_detected_queue.get()

            #print(detected_event)
            weight_sensor_index = detected_event[0][0]
            # gondola_index = (weight_sensor_index // (6*12)) + 1    
            # camera_index = gondola_to_camera_map[gondola_index]
            # camera_signal[camera_index] = 1 
            # print('\n')

            # # update camera frames    
            # for i in range(8):
            #     pick_result = camera_instance_list[i].update_frame(pose_estimator, camera_signal[i])
            #     print(pick_result)
            #     if pick_result:
            #         print(pick_result)
            #         current_customer_id = pick_result['track_id']
            # print(current_customer_id)

            print(detected_event)
            sensor_number_list =[]
            total_changed_weight = 0
            event_timestamp =0
            for kk in range(len(detected_event)):
                sub_event = detected_event[kk]
                sensor_number_list.append(sub_event[0])
                total_changed_weight = total_changed_weight + sub_event[1]
                event_timestamp = sub_event[2]
            
            item_fin_name, item_fin_number, item_fin_price,  item_per_weight  = weight_based_item_estimate(customer_shopping_list[current_customer_id], sensor_number_list, total_changed_weight, sensor_info, out_info)
            changed_item_info = [item_fin_name, item_fin_number, item_fin_price,  item_per_weight]
            customer_shopping_list_update(current_customer_id, total_changed_weight, changed_item_info)
            if total_changed_weight > 5:
                return_weight_sensor_item_updata(sensor_number_list, changed_item_info)

            if len(customer_shopping_list)> 0:
                for custom_dd in range(len(customer_shopping_list)):
                    print_receipt(custom_dd)

            #weight_based_item_info =[event_timestamp, item_fin_name, item_fin_number, item_fin_price]
            #print(weight_based_item_info)

        # # update camera frames    
        # print(time_coun)
        # for i in range(8):
        #     camera_instance_list[i].update_frame(pose_estimator, camera_signal[i])
