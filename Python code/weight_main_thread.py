from weight_data_read import csv_file_read as cfr
import queue
import numpy as np
import time

Weight_sensor_number = 348


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

        
    def weight_change_detection(self):
        pre_val = self.value[0]
        CONTINUE_TH = 5
        continue_count = 0
        state = 0
        change_flag = 0
        pre_weight = 0
        later_weight = 0
        con_start_loc =0
        con_stop_loc =0

        for kk in np.arange(1, len(self.value)):

            now_val = self.value[kk]
            if np.abs(now_val - pre_val) < 3:
                continue_count = continue_count+1
                change_flag = 0
            else:
                change_flag =1
                continue_count =0
            pre_val = now_val

            if (change_flag ==1) & (state ==1):
                con_stop_loc = kk
                state =2
            #print('%d, %d'%(continue_count, state))

            if (continue_count > CONTINUE_TH) & (state ==0):
                pre_weight = np.mean(self.value[(kk-7): (kk-2)])
                #print('%f, pre weight'%pre_weight)
                state =1


            if (continue_count > CONTINUE_TH) & (state ==2):
                later_weight = np.mean(self.value[(kk-5):kk])
                pre_weight = np.mean(self.value[(con_stop_loc - 10): (con_stop_loc - 5)])
                #print('%f, pre weight, %f, later weight' % (pre_weight, later_weight))
                con_start_loc = kk-5
                #print(kk)
                #print(later_weight)
                #print(pre_weight)
                changed_weight = later_weight - pre_weight
                ts_loc = int(np.round((con_start_loc+con_stop_loc)/2))
                time_stamp = self.timestamp[ts_loc]

                # repeat detection
                detected_len = len(self.weight_change_ts)
                repeat_flag = 0
                if detected_len < 1:
                    repeat_flag = 0
                else:
                    for detected_number in range(detected_len):
                        current_ts = self.weight_change_ts[detected_number]
                        if np.abs(current_ts - time_stamp) < 1: # if interval less than 1 second, we consider it is same event
                            repeat_flag =1
                            break

                if repeat_flag ==1:
                    state = 0
                    continue

                #print('%f time_stamp'%time_stamp)
                self.weight_change_value[self.number].append(changed_weight)
                self.weight_change_ts[self.number].append(time_stamp)
                #print(changed_weight)
                state = 0
                
    
    
    def value_update(self, val, timestamp):
        self.value = np.append(self.value, val)
        self.timestamp = np.append(self.timestamp, timestamp)
        if len(self.value) > 60*20: #store 20 seconds data
            del_num = len(self.value) - 60*20
            del_array = np.arange(del_num)
            self.value = np.delete(self.value, del_array, axis = 0)
            self.timestamp = np.delete(self.timestamp, del_array, axis = 0)
        self.weight_change_detection()
            
if __name__ == "__main__":
    gn = 2
    sn = 5
    pn = 5
    timestamp, weight_value = cfr(gn, sn, pn)
    print(len(weight_value))
    #print(weight_value)
    sensor_number = (gn - 1) * 6 * 12 + sn * 12 + pn
    item_dict = {'1': [10, 10, 2]}  # {'item_number': [item_total_number, weight, standard deviation]}
    initial_val = weight_value[0:50]

    initial_ts = timestamp[0:50]
    test_sensor = WeightSensor( sensor_number, item_dict, initial_val, initial_ts)


    for time_coun in np.arange( 50, len(weight_value)-10, 10):
        update_wv = weight_value[time_coun:time_coun+10]
        update_ts = timestamp[time_coun: time_coun+10]
        test_sensor.value_update(update_wv, update_ts)

        time.sleep(0.1)

        tmp_st = WeightSensor.weight_change_value[sensor_number]
        #print(WeightSensor.weight_change_value[sensor_number])
        if len(tmp_st) > 0:
            print(tmp_st)