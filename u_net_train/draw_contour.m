%remove all and close all
close all;
clear all;

%data path and file names
images_folder_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\images';
images_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\images\\*.jpg';

%data path and file names
labels_folder_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\labels';
labels_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\labels\\*.jpg';

%data path and file names
prediction_folder_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\prediction';
prediction_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\prediction\*.jpg';

%path for saving
save_folder_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\contour_prediction';

%path for saving
image_contour_folder_path = 'C:\\Users\\berkan.lafci\\Desktop\\safams\\deep_learning\\u_net\\berkan_pytorch\\predictions\\contour_label';

%image list
list_of_images = dir(images_path);

%label list
list_of_labels = dir(labels_path);

%prediction list
list_of_predictions = dir(prediction_path);

for i = 1: length(list_of_images)
    
    %read images
    image_name = list_of_images(i).name;
    image = imread(sprintf('%s\\%s',images_folder_path,image_name));
    
    %read images
    label_name = list_of_labels(i).name;
    label = imread(sprintf('%s\\%s',labels_folder_path,label_name));
    
    %read images
    prediction_name = list_of_predictions(i).name;
    prediction = imread(sprintf('%s\\%s',prediction_folder_path,prediction_name));
    
    %create path to the segmentation image
    prediction_contour_image_path = sprintf('%s\\%s',save_folder_path,image_name);
    
    %create path to the segmentation image
    image_contour_image_path = sprintf('%s\\%s',image_contour_folder_path,image_name);
    
    prediction_binary = (prediction>127);
    label_binary = (label>127);
    
    %get the contour of prediction
    [C,~,~,~] = bwboundaries(prediction_binary); 
    cropcoordinates_orig = C{1};
    
    cropcoordinates_orig = fliplr(cropcoordinates_orig);
    
    %get the contour of prediction
    [D,~,~,~] = bwboundaries(label_binary); 
    cropcoordinates_label = D{1};
    
    cropcoordinates_label = fliplr(cropcoordinates_label);
    
    %display the image and draw the contour on top of it
    figure; 
    imagesc(image);
    axis off;
    hold on; 
    axis image;
    colormap gray;
    plot(cropcoordinates_orig(:,1),cropcoordinates_orig(:,2),'Linewidth',2, 'Color',[0.451 0.941 0.902]);
    
    %save the image in segmentation folder
    saveas(gcf,prediction_contour_image_path);
    
    %display the image and draw the contour on top of it
    figure; 
    imagesc(image);
    axis off;
    hold on; 
    axis image;
    colormap gray;
    plot(cropcoordinates_label(:,1),cropcoordinates_label(:,2),'Linewidth',2, 'Color',[0.451 0.941 0.902]);
    
    %save the image in segmentation folder
    saveas(gcf,image_contour_image_path);
    
end