% import tanh_intercept, tanh_slope
load tanh_lut.mat

% convert to fixed points
tanh_intercept = floor(tanh_intercept*2^18)/2^18;
tanh_slope = floor(tanh_slope*2^10)/2^10;

% input data width (table entries)
dwidth = 8;
% fraction width
fwidth = 8; 

dlen = 2^(dwidth+fwidth);
x = (0:dlen-1);
% real range = [-8, 8]
xf = x / 2^(dwidth+fwidth-3);

xh = floor(x/2^fwidth);
xl = x - xh*2^fwidth;

tanh_hw = tanh_intercept(xh+1) + xl/2^(dwidth+fwidth-3) .* tanh_slope(xh+1);
tanh_err = tanh(xf) - tanh_hw;

figure
plot(xf, tanh_err);
xlabel('Input'); ylabel('Error');
title('Tanh Implementation Error')

