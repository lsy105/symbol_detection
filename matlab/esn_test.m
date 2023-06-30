load fpga/test_rc_wifi_data.mat
 
cfg = {};
cfg.w_in = w_in/2^15;   % Q15 format
cfg.w_x = w_x/2^15;     % Q15 format
cfg.w_out = w_out/2^5;  % 10Q5 format

cfg.num_neurons = size(w_x, 1);
cfg.num_outputs = size(w_out, 1);

din = inputs.' / 2^15;  % Q15 format

y = esn_predict(din, cfg).';


var(predict)

var(y)
