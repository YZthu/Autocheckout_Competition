clear all;
close all;
clc;

data_path ='..\csv\';
gn = 1;
sn = 1;
pn = 1;
s1=[];
for gn=1:5
    for sn=1:6
        for pn=1:12
tmpfilename = strcat(data_path, 'gondola_',num2str(gn), '_shelf_' ,num2str(sn), '_plate_',num2str(pn),'.csv');

if exist(tmpfilename, 'file') ==0
        all_data =0;
        time_stamp_us = 0;
record_time_us = 0;
        ttttt='file error'
        return
    end
    
    fil = fopen(tmpfilename, 'rt');
    
    all_data = [];
    T =[];
    W = [];
    flag = 1;
    while feof(fil) ~= 1
        tmp_num = [];
        tmp_lin = fgetl(fil);
        unit = split(tmp_lin, ",");
        tmp_ts = str2double(char(unit(1)));
        tmp_val = str2double(char(unit(2)));
        if flag
            T0 = tmp_ts;
            flag =0;
        end
        T = [T, tmp_ts-T0];
        W = [W, tmp_val];
    end
    
    w1= mean(W(1:100));
    w2 = mean(W(1900:2000));
    if abs(w1-w2) > 10
        figure
        plot(T,W);
        s1 =W;
        title([num2str(gn), '  ', num2str(sn),'  ', num2str(pn)]);
    end
        end
    end
end
    