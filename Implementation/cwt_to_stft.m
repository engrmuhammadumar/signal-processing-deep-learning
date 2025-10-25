% Define the output directory
outputDir = 'E:\1 Paper Work\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_stft_data\BFI\';

% Check if the directory exists, if not, create it
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Sampling frequency
fs = 1e6; % 1 MHz

% Loop to generate and save STFT images
for i = 1:40
    % Extract data row and normalize
    data_row = AE_ALL.BFI{1, 1}(i, :);
    data_row_normalized = data_row - mean(data_row);
    
    % Perform STFT
    window_length = 1000; % Define a larger window length for better frequency resolution
    overlap = round(0.8 * window_length); % Define overlap
    nfft = 2048; % Increase FFT points for better frequency detail
    [S, F, T] = stft(data_row_normalized, fs, 'Window', hamming(window_length), 'OverlapLength', overlap, 'FFTLength', nfft);
    
    % Convert STFT to decibel scale for better visualization
    S_dB = 10 * log10(abs(S) + eps); % Add a small epsilon to avoid log(0)
    
    % Create figure for STFT
    figure('Visible', 'off'); % Prevent figure from showing
    
    % Plot STFT in dB scale
    surf(T, F, S_dB, 'EdgeColor', 'none');
    axis tight;
    view(0, 90); % Set view to 2D
    
    % Set frequency and time limits if desired
    ylim([0, 1e5]); % Set frequency limit to 0 - 100 kHz for better visibility

    % Adjust figure properties
    ax = gca;
    ax.XLabel = [];
    ax.YLabel = [];
    ax.Title = [];
    ax.XTickLabel = {}; % Remove x-axis numbers
    ax.YTickLabel = {}; % Remove y-axis numbers
    colorbar('off'); % Remove colorbar
    
    % Set colormap to grayscale or any other colormap
    colormap(jet); % Change 'jet' to any other colormap you prefer
    
    % Save figure as PNG
    saveas(gcf, [outputDir 'Sample_' num2str(i) '.png']); % Adjust file name if needed
    
    % Close figure to save memory
    close(gcf);
end
