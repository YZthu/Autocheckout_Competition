import numpy as np
impo
a =[]
for gn in range(5):  # gondola
    for sn in range(6):  # shelf
        for pn in range(12):  # plate
            tmp_gn = gn + 1
            tmp_sn = sn + 1
            tmp_pn = pn + 1
            sensor_number = (tmp_gn - 1) * 6 * 12 + (tmp_sn - 1) * 12 + tmp_pn
            a.append(sensor_number)

print(a)