
load fpga/test_rc_wifi_float.mat

cfg = {};
cfg.w_in = w_in;
cfg.w_x = w_x;
cfg.w_out = w_out;
cfg.num_neurons = size(w_x,1);
cfg.num_outputs = size(w_out,1);

din = inputs.';

y = esn_predict(din, cfg).';


var(predict)

var(y)