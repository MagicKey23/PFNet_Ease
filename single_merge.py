# import the necessary packages
import numpy as np
import cv2
import os
        
def load_image():
        #Kaney
        #Load Contour
        img = cv2.imread("") 
        #Load Original Image
        original_img = cv2.imread("")
        
        
        edges = cv2.Canny(img, 100,200)
        
        cv2.imshow("before",edges)
        cv2.waitKey(0)
        
        # Find contours of the edges
        contours, ___ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
        
        # Calculate the area of each contour and add it up
        total_area = 0
        
        
        print("Number of contour is: " + str(len(contours))) 
        
        
        count = 1
        # Create new edges 
        new_edges = np.zeros_like(edges)
        
        #add threshold
        
        min_threshold = 50
        
        for contour in contours:
            area = cv2.contourArea(contour)
            print("The contour area: " + str(count) + ": " + str(area))
            count += 1
            if area > min_threshold:
                cv2.drawContours(new_edges, [contour], 0, 255, -1)
            total_area += area
         
        print("The Total Area is: " + str(total_area))

     
   

        #New Contour
        
        cv2.imshow('Filtered', new_edges)
        cv2.waitKey(0)
        
        #New edge 
        
        new_edges = cv2.Canny(new_edges, 100, 200)
        
        cv2.imshow("After", new_edges)
        
        
        cv2.waitKey(0)

        #Kernel for dilation
        kernel = np.ones((3,3), np.uint8)
        #Apply dilation 
        edges = cv2.dilate(new_edges, kernel, iterations=1)

        rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # RGB for matplotlib, BGR for imshow() !
        rgb *= np.array((0,0,5),np.uint8) # bgr

        add = cv2.bitwise_or(original_img, rgb)

        print("Printing out result...")
        cv2.imwrite(os.path.join("outline.png"), add)


load_image()


