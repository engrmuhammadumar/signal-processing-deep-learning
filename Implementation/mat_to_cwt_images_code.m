for i = 1:40
    data_row = AE_ALL.BF{1, 1}(i, :);
    data_row_normalized = data_row - mean(data_row);
    figure('Visible', 'off'); % Prevent figure from showing
    cwt(data_row_normalized, 'amor', 1e6);
    ax = gca;
    ax.XLabel = [];
    ax.YLabel = [];
    ax.Title = [];
    ax.XTickLabel = {}; % Remove x-axis numbers
    ax.YTickLabel = {}; % Remove y-axis numbers
    colorbar('off');
    saveas(gcf, ['E:/1 Paper Work/Cutting Tool Paper/Dataset/cutting tool data/test_data_40_images/BFI/Sample_' num2str(i) '.png']); % Adjust file name to start from 81
    close(gcf); % Close the figure to save memory
end


