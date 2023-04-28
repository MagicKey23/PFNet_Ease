import os 
import re
import cv2
#Kaney 

def get_data(TP, TN, FP, FN, Total):
    #Calculate The Data 
    TP = round(TP/Total * 100, 1)
    TN = round(TN/Total * 100, 1)
    FP = round(FP/Total * 100, 1) 
    FN = round(FN/Total * 100, 1) 
    Accuracy = round((TP + TN), 1)
    
    print("True positive: " + str(TP) + "%")
    print("True negative: " + str(TN) + "%")
    print("False positive: " + str(FP) + "%")
    print("False negative: " + str(FN) + "%")
    print("Accuracy: " + str(Accuracy) + "%")
    
    
    
def get_mask_area(results_path, project_name, exp_name, test_path):
    print("Calculate Mask Area Accuracy...")
    #Go Through The Folder and Load the .txt file then read all of the true positive text file 
    #Read image_classification txt 
    #Regex
    find_negative = re.compile("negative", re.IGNORECASE)
    find_false_positive = re.compile("false_positive", re.IGNORECASE)
    
    #Data Variables 
    
    mask_area = 0
    mask_sum = 0
    count_file = 0 
    gt_total_area = 0
    
    #Open Image Classification File 
    files = open(os.path.join(results_path, project_name, exp_name, "./imageclass/posImages.txt"), 'r')
        
    #Remove \n
    cleaned_files = [line.strip() for line in files]
    # Path to original img and ground truth
    output_path = os.path.join(results_path, project_name, exp_name, 'masks')
    original_path = os.path.join(test_path, 'gts')
    
    
    for file in cleaned_files:
        #Make sure file doesn't have any negative string or false_positive string
        if re.search(find_negative, file) is None and re.search(find_false_positive, file) is None:
            count_file += 1 
            
            #Read File Output
            img_output = cv2.imread(os.path.join(output_path, file))
            #Read Original Image 
            img_original = cv2.imread(os.path.join(original_path, file))
            
            
            #Get Edges
            edges_output = cv2.Canny(img_output, 100,200)
            edges_original = cv2.Canny(img_original, 100,200)
            #Find contour
            contours_output, ___ = cv2.findContours(edges_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_original, ___ = cv2.findContours(edges_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #Gets Area 
            area_output = cv2.contourArea(contours_output[0])
            gt_total_area = cv2.contourArea(contours_original[0])
   
         
            if(len(contours_original) > 1 or gt_total_area < 100):
                #Defective Head Contour
                count_file -= 1
            else:
                #calculate Percentage
                if  area_output <= gt_total_area:
                    try:
                        percentage = round((area_output/gt_total_area) * 100, 2)
                        mask_sum  += percentage
                    except ZeroDivisionError as e: 
                        print("Error: Cannot divide by zero")
                else:
                     try:
                        
                        
                        percentage = round((1-(area_output-gt_total_area)/gt_total_area) * 100, 2)
                        if(percentage < 1):
                            percentage = 0
                               
                        mask_sum  +=  percentage
                     except ZeroDivisionError as e: 
                        print("Error: Cannot divide by zero")
    try:
        mask_area = round(mask_sum/count_file, 2)
    except ZeroDivisionError as e: 
        print("Error: Cannot divide by zero")
        
    files.close()   
    
    print("The average accuracy for contour area is: " + str(mask_area) + '%')
   