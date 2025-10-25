% Load the data
load('E:\Cutting Tool Paper\Dataset\cutting tool data\AE_ALL.mat');  % Replace with the actual filename

% Extract the first row of the data
data = AE_ALL.BF;

% Define the sampling frequency
Fs = 1e6;  % 1 MHz

% Compute and plot the Continuous Wavelet Transform (CWT)
cwt(data, 'amor', Fs);
