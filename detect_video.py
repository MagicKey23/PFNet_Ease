
import time
import datetime
import platform

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

from config import *
from misc import *
from PFNet import PFNet
import cv2 
from settings import get_config

#Martin I am using the platform module to check what OS the script is being ran on, then we can decide how to connect to a GPU
currOS = platform.system()
#Kaney Args Parameter
opt = get_config()
print(opt)

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])


args = {
    'scale': opt.img_size,
    'save_results': True
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()



def main():
   #path to save output to video
   #count = 1
    save_path = "./output_video/"
    check_mkdir(save_path)
    if currOS == 'Darwin':
        net = PFNet(backbone_path).to('mps')
    else:
        net = PFNet(backbone_path).cuda(device_ids[0])

   
    net.load_state_dict(torch.load(opt.load_weight))
    print('Load {} succeed!'.format(opt.load_weight))
    
    video = cv2.VideoCapture(opt.load_video)
    
    print("Load Video")
    
    #Martin Added the recorder to record the result at the end
    
    if opt.save_video:

        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')# WINDOW1/MAC
        
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        
        out = cv2.VideoWriter(save_path + opt.load_video, fourcc, 20.0, (frame_width, frame_height))
    
    
    print("Load Recorder")
    
    net.eval()
    with torch.no_grad():
    
        start = time.time()

        while video.isOpened():
                    
            ret, frame = video.read()
            
            
            scale_percent = opt.frame_scale # Kaney
            
            if(frame.all != None):
                width  = int(frame.shape[1] * scale_percent / 100) #fix crash bug
                height = int(frame.shape[0] * scale_percent / 100)
            
            dim = (width, height)
            
            resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            else:
            
                img = Image.fromarray(resized).convert("RGB")
                w, h = img.size

                #Martin - depending on what operating system we are using, choose how to connect to GPU
                if currOS == 'Darwin':
                    img_var = Variable(img_transform(img).unsqueeze(0)).to('mps')
                else:
                    img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                _, _, _, prediction = net(img_var)
                
                #Target Segmenation
                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                
                #Image.fromarray(prediction).convert('L').save(os.path.join(save_path, "output_contour_" + str(count) + ".png"))

                    
                #array prediction convert
                #Kaney Adding this so it would show the contour on original image
                result = cv2.Canny(prediction,100,300)
                
                kernel = np.ones((3,3), np.uint8)
                result = cv2.dilate(result, kernel, iterations=1)
                rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)  # RGB for matplotlib, BGR for imshow() !
                rgb *= np.array((0, 0, 1), np.uint8)  # bgr
                
                add = cv2.bitwise_or(rgb, resized) # Martin
                if opt.save_video == True:
                    out.write(add) 
                
                cv2.imshow('frame', add)
                #count += 1
                if cv2.waitKey(1) == ord('q'):
                    break
        
        
        
    if opt.save_video == True:
        out.release()
    video.release()
    cv2.destroyAllWindows()


          
            


if __name__ == '__main__':
    main()
