% Define the input and output paths
input_path = 'E:\1 Paper Work\Cutting Tool Paper\Dataset\cutting tool data\test_data_160_images\N';
output_path_gaussian = 'E:\1 Paper Work\Cutting Tool Paper\Dataset\cutting tool data\test_data_160_images\N_Gaussian_Test';

% Create output directory if it does not exist
if ~exist(output_path_gaussian, 'dir')
    mkdir(output_path_gaussian);
end

% Apply Gaussian filter and adjust contrast for each image
for i = 1:40
    % Read the image
    filename = fullfile(input_path, ['Sample_' num2str(i) '.png']);
    image = imread(filename);
    
    % Apply Gaussian Blur
    gaussian_blur = imgaussfilt(image, 1); % Adjust sigma value as needed
    
    % Increase contrast (optional step for better visual distinction)
    gaussian_blur_contrast = imadjust(gaussian_blur, stretchlim(gaussian_blur), []);
    
    % Save the filtered images with enhanced contrast
    imwrite(gaussian_blur_contrast, fullfile(output_path_gaussian, ['Sample_' num2str(i) '.png']));
end

disp('Filtering, contrast adjustment, and saving of images completed.');
