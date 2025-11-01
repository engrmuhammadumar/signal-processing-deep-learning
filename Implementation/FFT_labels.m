% Define the path to save FFT images
output_folder = 'TF_channel4_FFT';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Number of samples to process
num_samples = 40;

% Sampling frequency (Adjust this according to your data)
Fs = 1e6; 

for i = 1:num_samples
    % Extract data row
    data_row = AE_ALL.TF{4, 1}(i, :);
    % Normalize data row
    data_row_normalized = data_row - mean(data_row);

    % Compute FFT
    L = length(data_row_normalized); % Length of the signal
    Y = fft(data_row_normalized); % Compute the FFT
    P2 = abs(Y / L); % Two-sided spectrum
    P1 = P2(1:L/2+1); % Single-sided spectrum
    P1(2:end-1) = 2 * P1(2:end-1); % Adjust amplitude for single-sided spectrum
    f = Fs * (0:(L/2)) / L; % Frequency axis

    % Plot FFT
    figure('Visible', 'off', 'Position', [100, 100, 600, 400]); % Set figure size
    plot(f / 1000, 10 * log10(P1), 'b'); % Plot frequency in kHz, magnitude in dB
    xlabel('Frequency (kHz)');
    ylabel('Magnitude (dB)');
    axis tight; % Remove extra white space around the plot
    set(gca, 'XTick', [], 'YTick', []); % Remove x and y axis ticks for cleaner look
    
    % Save the figure
    saveas(gcf, fullfile(output_folder, ['Sample_' num2str(i + 120) '_FFT.png']));
    close(gcf); % Close the figure to save memory
end
