kk= ones(1,10)/10;
count = 0;
newW=[];
for tt=1:10:size(s1,2)-10
    count = count+1;
    newW(count) = s1(tt:tt+9)*kk';
end

pre_val = newW(1);
continue_count =0;
state =0;
change_flag = 0;
test_c =[]
hist_con=[]
result=[];
for jj=2:size(newW,2)
    now_val = newW(jj);
    if abs(pre_val - now_val) < 3
        pre_val = now_val;
        continue_count = continue_count +1;
    else
        continue_count = 0;
        change_flag = 1;
        pre_val = now_val;
    end
    test_c =[test_c, continue_count];
    result=[result; continue_count, state]
    if continue_count > 10 & state ==0
                pre_constant = mean(newW(jj-10:jj));
                hist_con = [hist_con, pre_constant];
                state = 1;            
    end
    if change_flag ==1 & state ==1
        stop1 =jj;
        state =2;
    end
    if continue_count > 10 & state ==2
                new_constant = mean(newW(jj-10:jj));
                stop2 = jj-10;
                state =3;
    end
    
    if state ==3
        changed_weight = abs(new_constant- pre_constant);
        locat = (stop1 + stop2)/2;
        if new_constant> pre_constant
            change_state =1;
        else
            change_state = -1;
        end
        state = 0;
    end
end
                
    
    
        