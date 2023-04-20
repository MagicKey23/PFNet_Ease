import os
import settings
import urllib.request


opt = settings.getConfig()

#Kaney

#Check if backbone exist 

#if not download the backbone

#'https://download.pytorch.org/models/resnet50-19c8e357.pth'

url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
filename = './backbone/resnet/resnet50-19c8e357.pth'

if not os.path.isfile(filename):
    print('resnet50-19c8e357 file is missing')
    print("Downloading resnet50-19c8e357")
    urllib.request.urlretrieve(url, filename)
    print("File downloaded successfully!")


backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = opt.dataset_path
training_path = opt.train_path
testing_path =  opt.test_path

train_path =  os.path.join(datasets_root, training_path)
test_path =  os.path.join(datasets_root, testing_path)
