
%자르는 한칸 크기
sampling_wide=1000;
%다음 인덱스 시작점 까지의 거리
step_forward=100;
%열 번호
row_num=0;
%데이터 셋 크기 선언을 해주기 위한 반복문
for i=1:length(var_names)

    expression=strcat("s=length(",var_names{i},".ori_rpm);");
    eval(expression);
    cnt=0;
    for j=1:step_forward:s-sampling_wide-1
        if j+sampling_wide>s
            break;
        end
        cnt=cnt+1;
    end
row_num=row_num+cnt;
end

td_data=zeros(row_num,8);

%데이터 셋 만들기 시작
tmp_row_num=1;
for i=1:length(var_names)
    %데이터별 rpm 불러오기 
    expression=strcat("rpm=",var_names{i},".ori_rpm;");
    eval(expression);

    tmp=strfind(expression,'norm');
    if isempty(tmp)==1
        fault=1;
    else
        fault=0;
    end


    %인덱스를 나누어서 나눈 것들 별로 계산
    
    for j=1:step_forward:length(rpm)-sampling_wide-1
        disp(tmp_row_num);

        k=j+sampling_wide;

        if k>length(rpm)
            break;
        end

        %RPM을 짤라줌
        rpm_sample=rpm(j:k);
        %평균
        td_data(tmp_row_num,1)=mean(rpm_sample)-20;
        %분산
        td_data(tmp_row_num,2)=var(rpm_sample);
        %왜도
        td_data(tmp_row_num,3)=skewness(rpm_sample);
        %kurtosis
        td_data(tmp_row_num,4)=kurtosis(rpm_sample)-2.95;
        %max
        td_data(tmp_row_num,5)=max(rpm_sample)-20;
        %min
        td_data(tmp_row_num,6)=min(rpm_sample)-20;
        %median
        td_data(tmp_row_num,7)=median(rpm_sample)-20;
        


      
        td_data(tmp_row_num,end)=fault;
        tmp_row_num=tmp_row_num+1;

    end
    

end

save('report_train.mat','td_data')
disp('---------Finish-----------');






