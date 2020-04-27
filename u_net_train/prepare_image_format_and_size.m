
clear all;
close all;

%% load the image

%path to dataset
data_dir = 'H:\reconstruction\Testdata-wholemouse\Source\Scan_106';

%just read tif files
image_folder_path = sprintf('%s\\%s\\%s',data_dir,'\recon715\raw');

%just read tif files
image_folder_path_tif = sprintf('%s\\%s\\%s',data_dir,'\recon715\raw','*.bin');

%labeled images
image_list = dir(image_folder_path_tif);

% %just read tif files
% label_folder_path = sprintf('%s\\%s\\%s',data_dir,'targets');
% 
% %just read tif files
% label_folder_path_png = sprintf('%s\\%s\\%s',data_dir,'targets','*.png');
% 
% %labeled images
% label_list = dir(label_folder_path_png);

%store the images after segmentation
% inputs_path_300_300 = sprintf('%s\\%s',data_dir,'inputs_300_300');

%store the images after segmentation
inputs_path_256_256 = sprintf('%s\\%s',data_dir,'inputs700_256_256');

%store the images after segmentation
% inputs_path_128_128 = sprintf('%s\\%s',data_dir,'inputs_128_128');
% 
% %store the images after segmentation
% inputs_path_64_64 = sprintf('%s\\%s',data_dir,'inputs_64_64');
% 
% %store the images after segmentation
% inputs_path_28_28 = sprintf('%s\\%s',data_dir,'inputs_28_28');

% %store the images after segmentation
% targets_path_300_300 = sprintf('%s\\%s',data_dir,'targets_300_300');
% 
% %store the images after segmentation
% targets_path_256_256 = sprintf('%s\\%s',data_dir,'targets_256_256');
% 
% %store the images after segmentation
% targets_path_128_128 = sprintf('%s\\%s',data_dir,'targets_128_128');
% 
% %store the images after segmentation
% targets_path_64_64 = sprintf('%s\\%s',data_dir,'targets_64_64');
% 
% %store the images after segmentation
% targets_path_28_28 = sprintf('%s\\%s',data_dir,'targets_28_28');

% %create segmentation folder
% mkdir(inputs_path_300_300);

%create segmentation folder
mkdir(inputs_path_256_256);

% %create segmentation folder
% mkdir(inputs_path_128_128);

% %create segmentation folder
% mkdir(inputs_path_64_64);
% 
% %create segmentation folder
% mkdir(inputs_path_28_28);
% 
% %create segmentation folder
% mkdir(targets_path_300_300);
% 
% %create segmentation folder
% mkdir(targets_path_256_256);
% 
% %create segmentation folder
% mkdir(targets_path_128_128);
% 
% %create segmentation folder
% mkdir(targets_path_64_64);
% 
% %create segmentation folder
% mkdir(targets_path_28_28);

for i = 3:length(image_list)
    
%     %read the name of the label
%     label_name = label_list(i).name;
%     
%     %create path to the label
%     label_path = sprintf('%s\\%s',label_folder_path,label_name);
    
    %read the name of the image
    image_name = image_list(i).name;
    
    %create path to the image
    image_path = sprintf('%s\\%s',image_folder_path,image_name);
    fid = fopen(image_path, 'r');
    A = fread(fid,[332,332],'float64');
    image_org_1=A;
    %read the image
%     image_org_1 = read(image_path,881792);
    
    % cut 0s around the image
    %image_org_1 = image_org_1(25:274,25:274);
    
     image_org = uint8(((double(image_org_1) - double(min(min(image_org_1)))) / double((max(max(image_org_1)) - min(min(image_org_1))))) * 255);
    
%     %read label
%     label_org = uint8(imread(label_path)*255);
%     
%     %label_org = label_org(25:274,25:274);
%     
%     %label 300x300
%     label_org_300 = imresize(label_org,[300,300]);
%     
    %image 128x128
    image_256_256 = imresize(image_org,[256,256]);
    
%     %label 300x300
%     label_256_256 = imresize(label_org,[256,256]);
%     
%     %image 128x128
%     image_128_128 = imresize(image_org,[128,128]);
%     
%     %image 128x128
%     image_64_64 = imresize(image_org,[64,64]);
%     
%     %image 128x128
%     image_28_28 = imresize(image_org,[28,28]);
%     
%     %image 128x128
%     label_128_128 = imresize(label_org,[128,128]);
%     
%     %image 128x128
%     label_64_64 = imresize(label_org,[64,64]);
%     
%     %image 128x128
%     label_28_28 = imresize(label_org,[28,28]);
    
    %remove the extension of the image name to save in a different format
%     image_name = image_name(1:end-4);
    
    %remove the extension of the image name to save in a different format
    %label_name = label_name([1:20 end-15:end-4]);
%     
%     %store the images after segmentation
%     image_path_to_save_300_300 = sprintf('%s\\%s%s',inputs_path_300_300,image_name,'.jpg');
    
    %store the images after segmentation
    image_path_to_save_256_256 = sprintf('%s\\%s%s',inputs_path_256_256,image_name,'.jpg');
    
%     %store the images after segmentation
%     image_path_to_save_128_128 = sprintf('%s\\%s%s',inputs_path_128_128,image_name,'.jpg');
%     
%     %store the images after segmentation
%     image_path_to_save_64_64 = sprintf('%s\\%s%s',inputs_path_64_64,image_name,'.jpg');
%     
%     %store the images after segmentation
%     image_path_to_save_28_28 = sprintf('%s\\%s%s',inputs_path_28_28,image_name,'.jpg');
%     
%     %store the images after segmentation
%     label_path_to_save_300_300 = sprintf('%s\\%s%s',targets_path_300_300,image_name,'.jpg');
%     
%     %store the images after segmentation
%     label_path_to_save_256_256 = sprintf('%s\\%s%s',targets_path_256_256,image_name,'.jpg');
%     
%     %store the images after segmentation
%     label_path_to_save_128_128 = sprintf('%s\\%s%s',targets_path_128_128,image_name,'.jpg');
%     
%     %store the images after segmentation
%     label_path_to_save_64_64 = sprintf('%s\\%s%s',targets_path_64_64,image_name,'.jpg');
%     
%     %store the images after segmentation
%     label_path_to_save_28_28 = sprintf('%s\\%s%s',targets_path_28_28,image_name,'.jpg');
%     
%     %save the image
%     imwrite(image_org, image_path_to_save_300_300);
    
    %save the image
%     fwrite(image_256_256,[256:256],'uint16');
    imwrite(image_256_256, image_path_to_save_256_256);
    
%     %save the image
%     imwrite(image_128_128, image_path_to_save_128_128);
%     
%     %save the image
%     imwrite(image_64_64, image_path_to_save_64_64);
% 
%     %save the image
%     imwrite(image_28_28, image_path_to_save_28_28);
%     
%     %save the image
%     imwrite(label_org_300, label_path_to_save_300_300);
%     
%     %save the image
%     imwrite(label_256_256, label_path_to_save_256_256);
%     
%     %save the image
%     imwrite(label_128_128, label_path_to_save_128_128);
%     
%     %save the image
%     imwrite(label_64_64, label_path_to_save_64_64);
%     
%     %save the image
%     imwrite(label_28_28, label_path_to_save_28_28);
    
end

