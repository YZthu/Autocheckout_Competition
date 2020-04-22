from cvs_data_read import csv_file_read as cfr
import queue
import numpy as np
import time

Weight_sensor_number = 360


item_weight_list =[] #this list provide the weight value and standard deviation of each item 


class WeightSensor:
    weight_change_value = [[] for i in range(Weight_sensor_number) ]
    weight_change_ts = [[] for i in range(Weight_sensor_number) ]
    
    def __init__(self, number,item_dict, value, timestamp):
        self.number = number #integer
        self.item = item_dict #{'item_number': [total_number, weight, standard deviation]}
        self.value = value # initial weight value(numpy array)
        self.timestamp = timestamp #the timestamp of weight value (numpy array)
        # weight sensor sampling rate is 60 Hz, keep all the weight value in the last 20 seconds

        
    def weight_change_detection(self, total_detected_queue, detected_weight_event_queue):
        pre_val = self.value[0]
        CONTINUE_TH = 5
        continue_count = 0
        state = 0

        change_flag = 0
        pre_weight = 0
        later_weight = 0
        con_start_loc =0
        con_stop_loc =0
        changed_weight =0
        ts_loc = 0

        for kk in np.arange(1, len(self.value)):

            now_val = self.value[kk]
            if np.abs(now_val - pre_val) < 15:
                continue_count = continue_count+1
                change_flag = 0
            else:
                change_flag =1
                continue_count =0
            pre_val = now_val

            if (change_flag ==1) & (state ==1):
                con_start_loc = kk-1
                state =2
                continue_count = 0

            #print('%d, %d'%(continue_count, state))

            if (continue_count > CONTINUE_TH) & (state != 2):
                if kk>6:
                    pre_weight = np.mean(self.value[(kk-7): (kk-2)])
                else:
                    pre_weight = np.mean(self.value[0: (kk - 2)])
                #print('%f, pre weight'%pre_weight)
                if state ==0:
                    state =1

            if (state ==2):
                if ((kk - con_start_loc) > 20):
                    #print(kk - con_start_loc)
                    state =0
                    pre_val = self.value[kk]
                    continue


            if (continue_count > CONTINUE_TH) & (state ==2):
                if kk+2 < len(self.value):
                    later_weight = np.mean(self.value[(kk-3):(kk+2)])
                else:
                    later_weight = np.mean(self.value[(kk - 5):(kk)])
                con_stop_loc = kk-CONTINUE_TH
                #pre_weight = np.mean(self.value[(con_stop_loc - 10): (con_stop_loc - 5)])
                #print('%f, pre weight, %f, later weight' % (pre_weight, later_weight))

                #print(kk)
                #print(later_weight)
                #print(pre_weight)
                changed_weight = later_weight - pre_weight
                if (np.abs(changed_weight) < 15):
                    continue
                ts_loc = int(np.round((con_start_loc+con_stop_loc)/2))
                time_stamp = self.timestamp[ts_loc]


                # repeat detection
                detected_len = len(self.weight_change_ts[self.number-1])
                repeat_flag = 0
                if detected_len < 1:
                    repeat_flag = 0
                else:
                    for detected_number in range(detected_len):
                        current_ts = self.weight_change_ts[self.number-1][detected_number]
                        if np.abs(current_ts - time_stamp) < 1: # if interval less than 1 second, we consider it is same event
                            repeat_flag =1
                            continue

                if repeat_flag ==1:
                    state = 0
                    continue

                #print('%f time_stamp'%time_stamp)
                self.weight_change_value[self.number-1].append(changed_weight)
                self.weight_change_ts[self.number-1].append(time_stamp)
                info_list =[self.number, changed_weight, time_stamp]
                detected_weight_event_queue[self.number-1].put(info_list)
                total_detected_queue.put(info_list)

                # plot
                #plt.figure(self.number)
                #plt.plot(self.value)
                #plt.legend(str(self.number))
                #plt.title(str(self.number) + ' ' + str(con_start_loc) + '   ' + str(con_stop_loc) + '  ' + str(
                #    changed_weight) +'  '+ str(kk))
                #plt.show()

                state = 0
        #print(changed_weight)
        #print(state)
        #print(ts_loc)
    
    
    def value_update(self, total_detected_queue, detected_weight_event_queue, val, timestamp):
        self.value = np.concatenate((self.value,val),axis = 0)
        self.timestamp = np.concatenate((self.timestamp, timestamp), axis=0)
        if len(self.value) > 20*20: #store 20 seconds data
            del_num = len(self.value) - 20*20
            del_array = np.arange(del_num)
            self.value = np.delete(self.value, del_array, axis = 0)
            self.timestamp = np.delete(self.timestamp, del_array, axis = 0)
        self.weight_change_detection(total_detected_queue, detected_weight_event_queue)

def weight_based_item_estimate(current_shopping_list, sensor_number, changed_weight, weight_sensor_item_info_queue, out_sensor_item_info):
    #return or pick up
    if changed_weight > 5:
        #estimate the item number and category from the temporarry shopping list
        if len(current_shopping_list) < 1:
            #current customer shopping list is empty, return fail
            print('Error, detected return action but shopping list is empty')
            item_fin_name =[]
            item_fin_number = 0
            item_fin_price = 0
            item_per_weight = 0
        else:
            item_condidate = []
            for tmp_kk in range(len(current_shopping_list)):
                tmp_condite = current_shopping_list[tmp_kk]
                #shopping list formate  item_fin_name, item_fin_number, item_fin_price,  item_per_weight
                #item_condidate format item_name, item_weight, item_price
                tmp_item_info =[tmp_condite[0], tmp_condite[3], tmp_condite[2]]
                item_condidate.extend(tmp_item_info)

            item_fin_name, item_fin_number, item_fin_price, item_per_weight = most_similar_item_estimation( item_condidate, np.abs(changed_weight), 3)
    else:
        item_condidate = []
        if changed_weight < -5:
            sensor_total_number = len(sensor_number)
            if sensor_total_number == 1:
                #only one sensor weight changed
                item_condidate = weight_sensor_item_info_queue[sensor_number[0]-1]
            else:
                for tmp_s_num in range(len(sensor_number)):
                    tmp_sensor_num = sensor_number[tmp_s_num-1]
                    tmp_item_info = weight_sensor_item_info_queue[tmp_sensor_num-1]
                    item_condidate.extend(tmp_item_info)
            
        if len(item_condidate) > 0:  #the weight sensor contian useful info
            item_fin_name, item_fin_number, item_fin_price,  item_per_weight = most_similar_item_estimation(item_condidate, np.abs(changed_weight), 3)
        else:
            item_condidate = out_sensor_item_info
            item_fin_name, item_fin_number, item_fin_price,  item_per_weight = most_similar_item_estimation(item_condidate, np.abs(changed_weight), 1)
    return item_fin_name, item_fin_number, item_fin_price,  item_per_weight 

def most_similar_item_estimation(item_condidate, changed_weight, maximum_item_number):
    item_weight = []
    item_name =[]
    item_price =[]
    for kk in range(len(item_condidate)):
        tmp_item = item_condidate[kk]
        item_name.extend([tmp_item[0]])
        item_weight.extend([tmp_item[1]])
        item_price.extend([tmp_item[2]])
    condidate_weight = np.array(item_weight)
    if maximum_item_number > 1:
        for kk in range(maximum_item_number):
            if kk == 0:
                continue
            else:
                condidate_weight = np.append(condidate_weight, (kk+1)* item_weight)
    weight_difference = np.abs(condidate_weight - changed_weight)
    item_weight_diff = min(weight_difference)
    item_index = np.argmin(weight_difference)
    real_index = item_index % len(item_condidate)
    item_fin_name = item_name[real_index]
    item_fin_number = int(np.round(changed_weight/item_weight[real_index]))
    item_fin_price = item_price[real_index]
    item_per_weight = item_weight[real_index]
    return item_fin_name, item_fin_number, item_fin_price, item_per_weight

if __name__ == "__main__":

    detected_weight_event_queue = [queue.Queue(0) for kk in range(Weight_sensor_number)]  # event sotor queue of each sensor
    total_detected_queue = queue.Queue(0)  # number changed_weight timestamp #total queue of detected event

    weight_value_list =[ [] for jj in range(Weight_sensor_number)]
    timestamp_list = [ [] for jj in range(Weight_sensor_number)]
    weight_sensor_list = [ ]
    count = 0
    for gn in range(5): # gondola
        for sn in range(6): # shelf
            for pn in range(12): # plate
                tmp_gn = gn +1
                tmp_sn = sn+1
                tmp_pn = pn + 1

                timestamp, weight_value = cfr('test1', tmp_gn, tmp_sn, tmp_pn)
                sensor_number = (tmp_gn - 1) * 6 * 12 + (tmp_sn-1) * 12 + tmp_pn
                weight_value_list[sensor_number-1].extend(list(weight_value))
                timestamp_list[sensor_number-1].extend(list(timestamp))

                item_dict = {'1': [10, 10, 2]}  # {'item_number': [item_total_number, weight, standard deviation]}
                if len(weight_value) >50:
                    initial_val = weight_value[0:50]
                    initial_ts = timestamp[0:50]
                else:
                    initial_val = np.arange(0, 0.5, 0.01)
                    initial_ts = np.arange(0, 500, 10)

                count = count +1
                weight_sensor_list.append(WeightSensor( sensor_number, item_dict, initial_val, initial_ts))



    for time_coun in np.arange( 50, len(weight_value)-10, 10):
        for sensor_num in range(Weight_sensor_number):

            if time_coun < len(weight_value_list[sensor_num]) -10:

                update_wv = np.array(weight_value_list[sensor_num][time_coun:time_coun+10])
                update_ts = np.array(timestamp_list[sensor_num][time_coun: time_coun+10])
                weight_sensor_list[sensor_num].value_update(total_detected_queue, detected_weight_event_queue, update_wv, update_ts)

        time.sleep(0.1)

        tmp_st = WeightSensor.weight_change_value[124] # sensor 2 5 5
        #print(WeightSensor.weight_change_value[sensor_number])
        while not total_detected_queue.empty():
            print(total_detected_queue.get())
            print('\n')
