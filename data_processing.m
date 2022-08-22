ds_rates = [2,5,10,20,50,100,200];

for i = 1:length(var_names)
    expression = strcat(var_names{i},".mse=zeros(length(ds_rates),1);");
    eval(expression);
    for ds_idx = 1:length(ds_rates)
        % set downsampling rate
        ds_str = [var_names{i} '.ds' num2str(ds_rates(ds_idx))];
        tmp_sample_freq = floor(sample_freq/ds_rates(ds_idx));
        % get downsampled data
        expression = strcat(ds_str,"=downsample(",var_names{i},".original,",num2str(ds_rates(ds_idx)),");");
        eval(expression);
        % get period
        expression = strcat(ds_str,"_period=pulseperiod(",ds_str,", tmp_sample_freq);");
        eval(expression);
        % get no of pulses
        expression = strcat("tmp_length=length(",ds_str,"_period);");
        eval(expression);
        expression = strcat("tmp_length2=length(",var_names{i},".ori_period);");
        eval(expression);
        % check pulse identification failure
        if tmp_length < tmp_length2 -1
            break;
        % get MSE if there is no failure
        elseif tmp_length == tmp_length2 -1                
            expression = strcat(ds_str,"_rpm=1./",ds_str,"_period/6;");
            eval(expression);
            expression = strcat(var_names{i},".mse(",num2str(ds_idx),")=immse(",var_names{i},".ori_rpm(1:end-1),",ds_str,"_rpm);");
            eval(expression);
        else
            expression = strcat(ds_str,"_rpm=1./",ds_str,"_period/6;");
            eval(expression);
            expression = strcat(var_names{i},".mse(",num2str(ds_idx),")=immse(",var_names{i},".ori_rpm,",ds_str,"_rpm);");
            eval(expression);
        end
    end
end