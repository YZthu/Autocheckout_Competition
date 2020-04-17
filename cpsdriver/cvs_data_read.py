import pandas as pd
import numpy as np
import os

def csv_file_read(test_name, gn, sn, pn):
    file_path = '../csv/'
    file_name = file_path + test_name + '/gondola_' + str(gn) + '_shelf_' + str(sn) + '_plate_' + str(pn) + '.csv'
    if not os.path.exists(file_name):
        down_ts =[]
        down_wv =[]
        return np.array(down_ts), np.array(down_wv)
    weight_data = pd.read_csv(file_name)
    timestamp = np.array(list(weight_data.iloc[:, 0]))
    weight_value = np.array(list(weight_data.iloc[:, 1]))


    down_ts = []
    down_wv = []

    for jj in np.arange(0, len(weight_value)-3, 3):
        #print(jj)
        tmp_ts = timestamp[jj: jj + 3]
        tmp_wv = weight_value[jj: jj + 3]
        #print(tmp_wv)

        mean_ts = np.mean(tmp_ts)
        mean_wv = np.mean(tmp_wv)
        #print(mean_wv)
        down_ts.append( mean_ts)
        down_wv.append( mean_wv)


    return np.array(down_ts), np.array(down_wv)

def ground_truth_read(test_name, weight_sensor_total_number):
    file_path = '../csv/ground_truth/'
    file_name = file_path + test_name + '-all_product' + '.csv'
    if not os.path.exists(file_name):
        down_ts =[]
        down_wv =[]
        return np.array(down_ts), np.array(down_wv)
    item_weight_location = pd.read_csv(file_name)
    item_name = list(item_weight_location.iloc[:, 0])
    gondola = np.array(list(item_weight_location.iloc[:, 1]))
    shelf = np.array(list(item_weight_location.iloc[:, 2]))
    plate = np.array(list(item_weight_location.iloc[:, 3]))
    item_weight = np.array(list(item_weight_location.iloc[:, 4]))
    item_price = np.array(list(item_weight_location.iloc[:, 5]))

    weight_sensor_info = [[] for jj in range(weight_sensor_total_number)]
    out_sensor_product_info =[]

    for tmp_sensor in range(len(item_name)):
        tmp_gondola = gondola[tmp_sensor]
        tmp_shelf = shelf[tmp_sensor]
        tmp_plate = plate[tmp_sensor]
        if tmp_gondola < 1:
            tmp_product_info = [item_name[tmp_sensor], item_weight[tmp_sensor], item_price[tmp_sensor]]
            out_sensor_product_info.append(tmp_product_info)
            continue

        #tmp_info_list = [item_name[tmp_sensor], gondola[tmp_sensor], shelf[tmp_sensor], plate[tmp_sensor], item_weight[tmp_sensor], item_price[tmp_sensor] ]
        tmp_info_list = [item_name[tmp_sensor], item_weight[tmp_sensor], item_price[tmp_sensor] ]
        tmp_sensor_number = (tmp_gondola-1)* 6 * 12 + (tmp_shelf - 1) * 12 + tmp_plate
        #print(tmp_sensor_number)
        weight_sensor_info[tmp_sensor_number].append(tmp_info_list)
    return weight_sensor_info, out_sensor_product_info