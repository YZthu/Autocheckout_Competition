import queue
import numpy as np
import time
import threading

from cvs_data_read import csv_file_read as cfr
from cvs_data_read import ground_truth_read as gtr

from weight_main import WeightSensor
from weight_main import weight_based_item_estimate

lock = threading.Lock()

customer_shopping_list =[]
#each shopping list formate  item_fin_name, item_fin_number, item_fin_price,  item_per_weight

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
    Weight_sensor_number = 360

    detected_weight_event_queue = [queue.Queue(0) for kk in
                                   range(Weight_sensor_number)]  # event sotor queue of each sensor
    total_detected_queue = queue.Queue(0)  # number changed_weight timestamp #total queue of detected event
    merged_detected_queue = queue.Queue(0)

    #ground truth data read
    sensor_info, out_info = gtr('test1', Weight_sensor_number)

    # people get into store
    current_customer_id = allocate_customer_id()

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

        time.sleep(0.1)

        tmp_st = WeightSensor.weight_change_value[124]  # sensor 2 5 5
        # print(WeightSensor.weight_change_value[sensor_number])

        while not total_detected_queue.empty():
            tmp_info = total_detected_queue.get()
            tmp_timestamp = tmp_info[2]
            
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
            if len(customer_shopping_list)> 0:
                for custom_dd in range(len(customer_shopping_list)):
                    print_receipt(custom_dd)

            weight_based_item_info =[event_timestamp, item_fin_name, item_fin_number, item_fin_price]
            print(weight_based_item_info)