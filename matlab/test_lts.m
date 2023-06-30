
% Load Rx packets
load data/S1/wifi_rx_time_cp.mat

% remove non-packets
v1 = var(wifi_rx_time_cp(1,:));
pkt_idx = find(var(wifi_rx_time_cp.') > 0.5*v1);
rx_time_cp = wifi_rx_time_cp(pkt_idx, :);

% Preprocess LTS
ss = conj(flipud(lts));

% Check packet starts using LTS
total_packets = 0;
for pidx=1:2:size(rx_time_cp, 1)
  rx = rx_time_cp(pidx:pidx+1,:).';
  corr = filter(ss, 1, rx(:));
  % [max_corr, max_idx] = max(abs(corr));
  corr_mag = abs(corr);
  max_idx = find(corr_mag > mean(corr_mag)*10);
  max_corr = corr_mag(max_idx);
  if ~isempty(max_idx)
     disp(int32([pidx-2; max_idx; max_corr*1e3].'))
     total_packets = total_packets + 1;
  end
end

fprintf('\nDetect %d packets\n', total_packets)
