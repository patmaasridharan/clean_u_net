"""
Read the predictions results and calculate dice scores
"""

# import the required libraries
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2

#data path and file names
label_path = '/home/psridharan/Unet-Birkan/annotation_correlation/TS/Formated_targets/'

#data path and file names
prediction_path = '/home/psridharan/Unet-Birkan/u_net_test/predictions/prediction_correlationimages/'

image_size = 256   
#main_dir = os.getcwd()                                  # main folder directory (do not change!)
#test_dir = os.path.join(main_dir,'_data_test')		# directory for target labelled data
#data_dir = os.path.join(test_dir, '_data_Berkan_wholemouse_test')                     # data folder name: "_name_{img_size}_{img_size}"
#label_path= os.path.join(data_dir,'targets') 
#prediction_dir = os.path.join(main_dir,'predictions')	# directory for prediction data
#prediction_path = os.path.join(prediction_dir, 'prediction_Berkankidneymodel_Berkantest')


list_of_labels = [f for f in os.listdir(label_path) if f.endswith('.jpg')]
list_of_predictions = [f for f in os.listdir(prediction_path) if f.endswith('.jpg')]

#create a matrix with zeros for dicre score
dice = np.zeros((np.size(list_of_labels)))
accuracy = np.zeros((np.size(list_of_labels)))
precision = np.zeros((np.size(list_of_labels)))
recall = np.zeros((np.size(list_of_labels)))
TPscore =np.zeros((np.size(list_of_labels)))
TNscore =np.zeros((np.size(list_of_labels)))
FPscore =np.zeros((np.size(list_of_labels)))
FNscore =np.zeros((np.size(list_of_labels)))


# loop for calculating dice and saving predictions as png files
for i in range(np.size(list_of_labels)):
    
    #read the labels for dice calculation
    label_name = list_of_labels[i]
    label = plt.imread(label_path +'/'+ label_name)

    label = label/255
    
    #read the predictions for dice calculation
    prediction_name = list_of_predictions[i]
    print(prediction_path + '/'+ prediction_name)
    prediction = plt.imread(prediction_path + '/'+ prediction_name)
    print(prediction.size)
    prediction = prediction/255

    #threshold the intensities
    threshold = 0.5
    pred_logical = prediction >= threshold
    
    #create binary image
    pred_binary = pred_logical*1
    
    #scale the prediction with 2
    scaled_predictions = pred_binary*2
    
    print(label.size)
    print(prediction.size)

    #calculate the difference between label and the prediction
    diff = label - scaled_predictions
    
    
    #decide on TP,FP, TN, FN
    TP = np.size(np.where(diff == -1),1); #true positive
    FP = np.size(np.where(diff == -2),1); #false positive
    TN = np.size(np.where(diff == 0),1); #true negative
    FN = np.size(np.where(diff == 1),1); #false negative
    
    #calculate dice score and write into a matrix
    dice[i] = (2*TP)/(2*TP + FP +FN)
    #accuracy [i] = TP+TN/(TP + TN + FP +FN)
    #precision[i] = TP/(TP+FP)
    #recall[i]= TP/(TP+FN)
    TPscore[i]= TP
    TNscore[i]=TN
    FPscore[i]=FP
    FNscore[i]=FN

x_array = np.array(list_of_labels)
y_array = np.array(dice)
z_array = np.array(list_of_predictions)


print(x_array)

print(y_array)

data = [x_array, y_array , np.array(TPscore), np.array(TNscore), np.array(FPscore), np.array(FNscore), np.array(dice)]

data = np.vstack(data)

data = data.T

with open('scores_TS_Unet.csv','w') as out:
    for row in data:
        for col in row:
            out.write('{0};'.format(col))
        out.write('\n')
