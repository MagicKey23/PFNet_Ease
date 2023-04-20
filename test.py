import time
import datetime
import platform

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
import numpy as np

import re

import settings
from config import *
from misc import *
from PFNet import PFNet
import cv2

from calculate_data import get_data
from calculate_data import get_mask_area

#Martin I am using the platform module to check what OS the script is being ran on, then we can decide how to connect to a GPU
currOS = platform.system()

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

opt = settings.get_config()

print(opt)

results_path = opt.result_path
check_mkdir(results_path)
project_name = opt.project_name
exp_name = opt.exp_name

args = {
    'scale': opt.img_size,
    'save_results': opt.save_results
}

print(torch.__version__)

img_transform = transforms.Compose([

    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

to_pil = transforms.ToPILImage()


to_test = OrderedDict([
                        (exp_name , test_path)
                       ])

results = OrderedDict()

def main():
    
    #Store prediction data (Not Implement Yet)
    total_img = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
   
    
    #net = PFNet(backbone_path).to('cpu')
    #Martin - depending on what operating system we are using, choose how to connect to GPU
    if currOS == 'Darwin':
        net = PFNet(backbone_path).to('mps')
    else:
        net = PFNet(backbone_path).cuda(device_ids[0])

    net.load_state_dict(torch.load(opt.load_weight))
    print('Load {} succeed!'.format(opt.load_weight))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, project_name, name))
                
            #added by Martin
            check_mkdir(os.path.join(results_path, project_name, name)+'/imageclass/')
            check_mkdir(os.path.join(results_path, project_name, name) + '/imageclass/')

            p = open(os.path.join(results_path, project_name, name)+'/imageclass/posImages.txt','w')
            n = open(os.path.join(results_path, project_name, name)+'/imageclass/negImages.txt','w')



            #img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpg') or f.endswith('.JPG')]
            
            #Compatible to all images  #Kaney
            
            img_list = []
            
            for f in os.listdir(image_path):
                if not f.startswith("desktop") and not f.startswith("."):
                    img_list.append(os.path.splitext(f))
                    
            #Total Img  
            total_img = len(img_list)
            
            for idx, img_name in enumerate(img_list):
        
                #original_img = cv2.imread(os.path.join(image_path, img_name + '.png')) #Change to numpy load Kaney
                
                print(os.path.join(image_path, img_name[0] + img_name[1]))
                
                original_img = cv2.imread(os.path.join(image_path, img_name[0] + img_name[1]))
         
                img = Image.fromarray(original_img).convert('RGB') # Convert to Image object 
                
                w, h = img.size
                 
          
                #Martin - depending on what operating system we are using, choose how to connect to GPU
                if currOS == 'Darwin':
                    img_var = Variable(img_transform(img).unsqueeze(0)).to('mps')
                else:
                    img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                _, _, _, prediction = net(img_var)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))

               

                #added by Martin
                # result = edge_generator('background_head_1.png')
                #img = cv2.imread(os.path.join(os.path.join(results_path, exp_name, name, img_name + '.png')))
                # original_contour = cv2.imread('background_head_1contour.jpg')
                #original_img = cv2.imread(os.path.join(image_path, img_name + '.png'))
                
                result = cv2.Canny(prediction, 100, 200) #kaney/martin
                edges = cv2.Canny(prediction, 100, 200)
                
                # Kernel for dilation
                kernel = np.ones((3, 3), np.uint8) #kaney
                # Apply dilation
                result = cv2.dilate(result, kernel, iterations=1) #kaney

                rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)  # RGB for matplotlib, BGR for imshow() ! kaney
                rgb *= np.array((0, 0, 1), np.uint8)  # bgr coded by kaney

                add = cv2.bitwise_or(original_img, rgb) #martin
                
                if args['save_results']:
                
                    mask_path = os.path.join(results_path, project_name, name, "masks")
                    outline_path = os.path.join(results_path, project_name, name, "outlines")
                    check_mkdir(mask_path)
                    check_mkdir(outline_path)
                    print("Saving outlines : " + os.path.join(results_path, project_name, name, "outlines", img_name[0] + img_name[1]))
                    cv2.imwrite(os.path.join(results_path, project_name, name, "outlines", img_name[0] + img_name[1]), add)#martin
                    print("Saving masks : " + os.path.join(outline_path, "outlines", img_name[0] + img_name[1])) #Kaney
                    Image.fromarray(prediction).convert('L').save(os.path.join(results_path, project_name, name, "masks", img_name[0] + img_name[1]))#Kaney

                # Find contours of the edges / Martin
                contours, ___ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print("Number of target segmentation detect: " + str(len(contours)))
                
                #Split string 
                
                search_negative = re.compile("negative", re.IGNORECASE)
                
                
                if len(contours) == 0:
                    #Write Negative Image 
                    n.write(img_name[0] + img_name[1] + '\n')  
                    #Write False Negative if not contains "Negative" string
                    if re.search(search_negative, img_name[0]) is None:
                        false_negative = false_negative + 1
                    else:
                        true_negative = true_positive + 1
                elif len(contours) > 1:
                    #Write False Positive
                    p.write(img_name[0] + '_false_positive' + img_name[1] + '\n') #Kaney  Adding this so it would count 2 more contour as false positive     
                    false_positive = false_positive + 1
                else:
                    #Write Positive Image 
                    p.write(img_name[0] + img_name[1] + '\n')
                    #Write False positive if contains "negative" string
                    if re.search(search_negative, img_name[0]) is not None:
                        false_positive = false_positive + 1
                    else: 
                        true_positive = true_positive + 1
                
              
            n.close()
            p.close()
            print(('{}'.format(project_name)))
            print("{}'s average Time Is: {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is: {:.1f} fps".format(name, 1 / mean(time_list)))
            if opt.display_accuracy == True:
                get_data(true_positive, true_negative, false_positive, false_negative, total_img)
            if opt.display_area == True:
                get_mask_area(results_path, project_name, exp_name,test_path)

            end = time.time()
       
    
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
 
if __name__ == '__main__':
    main()
