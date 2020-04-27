%remove all and close all
close all;
clear all;

%data path and file names
images_folder_path = 'I:\RnD\R07 Algorithm Development\023 Semi- and Fully-automatic mouse segmentation\Deep_Learning\Models_Data_\clean_u_net\u_net_train\_data_kidney_liver_train_256_256\foraugumentation_512scanner\inputs_aug';
images_path = 'I:\RnD\R07 Algorithm Development\023 Semi- and Fully-automatic mouse segmentation\Deep_Learning\Models_Data_\clean_u_net\u_net_train\_data_kidney_liver_train_256_256\foraugumentation_512scanner\inputs_aug\*.jpg';

% images_folder_path1 = 'H:\reconstruction\ReconStack\Wholemosedata\Scan_5\withoutzerophase\inputjpeg800_fv30';
% images_path1 = 'H:\reconstruction\ReconStack\Wholemosedata\Scan_5\withoutzerophase\inputjpeg800_fv30\*.jpg';


%data path and file names
% labels_folder_path = 'H:\reconstruction\ReconStack\Wholemosedata\Scan_5\targets';
% labels_path = 'H:\reconstruction\ReconStack\Wholemosedata\Scan_5\targets\*.jpg';

%data path and file names
prediction_folder_path800 = 'I:\RnD\R07 Algorithm Development\023 Semi- and Fully-automatic mouse segmentation\Deep_Learning\Models_Data_\clean_u_net\u_net_train\_data_kidney_liver_train_256_256\foraugumentation_512scanner\targets_aug';
prediction_path800 ='I:\RnD\R07 Algorithm Development\023 Semi- and Fully-automatic mouse segmentation\Deep_Learning\Models_Data_\clean_u_net\u_net_train\_data_kidney_liver_train_256_256\foraugumentation_512scanner\targets_aug\*.jpg';
% 
% %data path and file names
% prediction_folder_path700 = 'H:\reconstruction\ReconStack\Wholemosedata\Scan_5\withoutzerophase\Prediction800_fv30';
% prediction_path700 ='H:\reconstruction\ReconStack\Wholemosedata\Scan_5\withoutzerophase\Prediction800_fv30\*.jpg';

%path for saving
save_folder_path =  'I:\RnD\R07 Algorithm Development\023 Semi- and Fully-automatic mouse segmentation\Deep_Learning\Models_Data_\clean_u_net\u_net_train\_data_kidney_liver_train_256_256\foraugumentation_512scanner\contours800_fv30';
mkdir(save_folder_path)
% 
% save_folder_path_rough =  'H:\reconstruction\ReconStack\Wholemosedata\Scan_5\withoutzerophase\contours800_fv30'
% mkdir(save_folder_path_rough)

% %path for saving
% label_contour_folder_path =  'I:\RnD\R07 Algorithm Development\023 Semi- and Fully-automatic mouse segmentation\Deep_Learning\Models_Data_\clean_u_net\u_net_test\predictions\contour_label_resized200epoch_wholemouse';
% mkdir(label_contour_folder_path)
% 
% labelcontour_overlay_folder_path =  'I:\RnD\R07 Algorithm Development\023 Semi- and Fully-automatic mouse segmentation\Deep_Learning\Models_Data_\clean_u_net\u_net_test\predictions\label_overlay_resized200epoch_wholemouse';
% mkdir(labelcontour_overlay_folder_path)

%image list
list_of_images = dir(images_path);

%label list
% list_of_labels = dir(labels_path);

%prediction list
list_of_predictions = dir(prediction_path800);

for i = 1: length(list_of_images)
    
    %read images
    image_name = list_of_images(i).name;
    image = imread(sprintf('%s\\%s',images_folder_path,image_name));
%     image1 = imread(sprintf('%s\\%s',images_folder_path1,image_name));
    
%     %read images
%     label_name = list_of_labels(i).name;
%     label = imread(sprintf('%s\\%s',labels_folder_path,label_name));
    
    %read images
    prediction_name = list_of_predictions(i).name;
%     prediction700= imread(sprintf('%s\\%s',prediction_folder_path700,prediction_name));
    prediction800 = imread(sprintf('%s\\%s',prediction_folder_path800,prediction_name));

 
     %create path to the segmentation image
     prediction_image_path0 = sprintf('%s\\%s',save_folder_path,prediction_name);
   % prediction_image_path = sprintf('%s\\%s',save_folder_path_rough,prediction_name);
%     %create path to the segmentation image
     %label_contour_image_path = sprintf('%s\\%s',label_contour_folder_path,image_name);
    
%     %create path to the segmentation image
%     label_overlay_image_path = sprintf('%s\\%s',labelcontour_overlay_folder_path,image_name);
    
   % prediction_binary700 = (prediction700>127);
    prediction_binary800 = (prediction800>127);
%     label_binary = (label>127);
    
    %get the contour of prediction
%     [C,~,~,~] = bwboundaries(prediction_binary700); 
%     cropcoordinates_orig = C{1};    
%      cropcoordinates_orig = fliplr(cropcoordinates_orig);
    
        %get the contour of prediction
    [E,~,~,~] = bwboundaries(prediction_binary800); 
    cropcoordinates_orig1 = E{1};    
  %    cropcoordinates_orig1 = flipud(cropcoordinates_orig1);
      cropcoordinates_orig1 = fliplr(cropcoordinates_orig1);
    
%     %get the contour of prediction
%     [D,~,~,~] = bwboundaries(label_binary); 
%     cropcoordinates_label = D{1};    
%     cropcoordinates_label = fliplr(cropcoordinates_label);
    
    %display the image and draw the contour on top of it
    figure; 
    imagesc(image);
    axis off;
    hold on; 
    axis image;
    colormap gray;
%     plot(cropcoordinates_orig(:,1),cropcoordinates_orig(:,2),'Linewidth',2, 'Color',[0.451 0.941 0.902]); %cyan
    plot(cropcoordinates_orig1(:,1),cropcoordinates_orig1(:,2),'Linewidth',2, 'Color',[0 ,1,0]); %green
%        plot(cropcoordinates_label(:,1),cropcoordinates_label(:,2),'Linewidth',2, 'Color',[1 0 0]); %red
    %save the image in segmentation folder
    saveas(gcf,prediction_image_path0);
    
%     D = bwdist(prediction/255);
%     % soft edge
%     Dn = mat2gray(D);
%     Dn_inv = abs(Dn-1);
%     mask_soft = Dn_inv.^10;
%     figure;imshow(im2double(image).*mask_soft);
%     saveas(gcf,prediction_soft_contour_image_path)
    
%     figure;imshow(image.* (prediction/255));
%     saveas(gcf,prediction_overlay_image_path)
    
    %display the image and draw the contour on top of it
%     figure; 
%     imagesc(image1);
%     axis off;
%     hold on; 
%     axis image;
%     colormap gray;
%     plot(cropcoordinates_orig(:,1),cropcoordinates_orig(:,2),'Linewidth',2, 'Color',[1 0 0]);
%     
%     %save the image in segmentation folder
%     saveas(gcf,prediction_image_path);
%     
%         
%     figure;imshow(image.* (label/255));
%     saveas(gcf,label_overlay_image_path)
    
end