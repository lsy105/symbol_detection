% Calculate prediction using ESN update cycles
function yout = esn_predict(xin, cfg)

% Pre-allocation
state = zeros(cfg.num_neurons, 1);
yout = zeros(cfg.num_outputs, size(xin,2));

% ESN update iterations
for n = 1:size(xin,2)
    activation = cfg.w_x * state + cfg.w_in * xin(:, n);
    state = tanh(activation);
    yout(:, n) = cfg.w_out * [state; xin(:, n)];
end


end %function
