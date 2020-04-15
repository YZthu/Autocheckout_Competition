import queue
import numpy as np
import time
from weight_data_read import csv_file_read as cfr

from weight_main import WeightSensor

if __name__ == "__main__":
    Weight_sensor_number = 360

    detected_weight_event_queue = [queue.Queue(0) for kk in
                                   range(Weight_sensor_number)]  # event sotor queue of each sensor
    total_detected_queue = queue.Queue(0)  # number changed_weight timestamp #total queue of detected event
    merged_detected_queue = queue.Queue(0)

    weight_value_list = [[] for jj in range(Weight_sensor_number)]
    timestamp_list = [[] for jj in range(Weight_sensor_number)]
    weight_sensor_list = []
    count = 0
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
            buffer_info.extend(tmp_info)
            new_event = False
            if abs(pre_timestamp - tmp_timestamp) > 2:
                new_event = True
            if new_event:
                merged_detected_queue.put(buffer_info)
                buffer_info = []
                pre_timestamp = tmp_timestamp
            
        while not merged_detected_queue.empty():
            print(merged_detected_queue.get())
            print('\n')