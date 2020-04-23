import numpy as np
from cvs_data_read import ground_truth_read as gtr
from weight_main import weight_based_item_estimate
Weight_sensor_number = 360
sensor_info, out_info = gtr('test1', Weight_sensor_number)

#detected_event = [[123, -21.866723632812636, 1580250255.4365394], [124, -59.09319661458335, 1580250255.589082], [125, -27.650744628906295, 1580250255.4365394]]
detected_event = [[58, -558.7344238281248, 1580250261.7786217]]
print(detected_event)
sensor_number_list =[]
total_changed_weight = 0
event_timestamp =0
for kk in range(len(detected_event)):
    sub_event = detected_event[kk]
    sensor_number_list.append(sub_event[0])
    total_changed_weight = total_changed_weight + sub_event[1]
    event_timestamp = sub_event[2]
            
item_fin_name, item_fin_number, item_fin_price = weight_based_item_estimate(sensor_number_list, total_changed_weight, sensor_info, out_info)
weight_based_item_info =[event_timestamp, item_fin_name, item_fin_number, item_fin_price]
print(weight_based_item_info)