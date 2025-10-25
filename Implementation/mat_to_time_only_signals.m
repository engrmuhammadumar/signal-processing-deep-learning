for i = 1:40
    % Extract the data row from your cell array
    data_row = AE_ALL.N{, 1}(i, :);
    
    % Normalize the data (remove the mean)
    data_row_normalized = data_row - mean(data_row);
    
    % Create a figure without displaying it
    figure('Visible', 'off'); % Prevent figure from showing
    
    % Plot the normalized data with thicker lines and darker color
    plot(data_row_normalized, 'k', 'LineWidth', 1); % 'k' is for black color
    
    % Add labels and title with larger font size
    xlabel('Sample', 'FontSize', 10);
    ylabel('Magnitude', 'FontSize', 10);
    title('Normal', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Remove axis tick numbers (labels) for both x and y axes
    xticks([]); % Remove x-axis numbers
    yticks([]); % Remove y-axis numbers
    
    % Save the plot as an image
    saveas(gcf, ['E:\2 Paper MCT\Final Codes 4 Classes\Techincal Backgound\Time Signal/Sample_' num2str(i) '.png']);
    
    % Close the figure to save memory
    close(gcf);
end
