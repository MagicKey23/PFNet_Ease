#Kaney, Args Parameter

import argparse
import textwrap

def getConfig():
  
    #Format the args parser
    
    parser = argparse.ArgumentParser(prog='Help Tool',
    formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=40))
    
    # Training parameter settings
    parser.add_argument('--img_size',type=int, default=704, help='The size of the input image. Adjust the train scale by specifying a value, for example: 416, 704, or 1024. Default: %(default)s')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in each batch. Adjust as needed. Default: %(default)s')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for. Adjust as needed. Default: %(default)s')
    parser.add_argument('--last_epoch', type=int, default=0, help='The index of the last epoch completed. Default: %(default)s')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate for training. Default: %(default)s' )
    parser.add_argument('--optimizer', type=str, default='SGD', help='The optimizer to use for training. For example: Adam. Default: %(default)s')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='The weight decay for regularization during training. Default: %(default)s')
    parser.add_argument('--snap_shot', type=str, default='', help='A snapshot of the training model to save. Leave blank if not needed. Default: %(default)s')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='The learning rate decay factor.')
    parser.add_argument('--momentum', type=int, default=0.9, help='The momentum for optimizer during training. Default: %(default)s')
    parser.add_argument('--poly_train', type=bool, default=True, help='Set to True for polynomial decay learning rate during training. Default: %(default)s')
    parser.add_argument('--save_point', type=list, default=[1,10,20,30,40,50], help='Epochs at which to save the model weights. Enter as a list, for example: [1,10,20,30,40,50]. Default: %(default)s')
    parser.add_argument('--train_path', type=str, default="train/Train_Mix", help='The path to the training data. Default: %(default)s ')
    parser.add_argument('--dataset_path', type=str, default="./data/", help='The path to the root of the dataset. Default: %(default)s')        
    parser.add_argument('--exp_name', type=str, default="test1", help='Set experiment name. Default: %(default)s')
    parser.add_argument('--project_name', type=str, default="project1", help='Set project name. Default: %(default)s')

    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='The location where you want to save your training result. Default: %(default)s')

    # Test parameter settings
    parser.add_argument('--test_path', type=str, default="test/Train_Mix", help='The path to the test data. Default: %(default)s')
    parser.add_argument('--result_path', type=str, default="./results/", help='The path to the results. Default: %(default)s')
    parser.add_argument('--load_weight', type=str, default="./best.pth", help='The path to the pre-trained weight file. Enter in the format: Path + weight_name.pth. Default: %(default)s')
    parser.add_argument('--frame_scale', type=int, default="100", help='The percentage by which to upscale or downscale the camera capture. Default: %(default)s')
    parser.add_argument('--load_video', type=str, default="videoname.mp4", help='The path to the video file to load. Enter in the format: "./monkey.mp4". Default: %(default)s')
    parser.add_argument('--select_camera', type=int, default="0", help='The index of the camera to use. Enter as an integer value from 0 to 4. Default: %(default)s')
    parser.add_argument('--display_accuracy', type=bool, default=False, help='Display TP, TN, FP, NP, Accuracy. Note: Required Proper Formatting to work. Default: %(default)s')
    parser.add_argument('--display_area', type=bool, default=False, help='Display area accuracy. Note: Required Proper Formatting to work. Default: %(default)s')
    parser.add_argument('--device', type=int, default=0, help='The index of the camera to use for testing. For example: 0, 1, 2, or 3. Default: %(default)s')

    parser.add_argument('--save_video', type=bool, default=False, help='Save result video Default: %(default)s')
    parser.add_argument('--save_results', type=bool, default=True, help='Save infer result Default: %(default)s')

    # Hardware settings
    parser.add_argument('--num_workers', type=int, default=16, help='The number of worker threads to use. Default: %(default)s')

    opt = parser.parse_args()
    
    return opt


if __name__ == '__main__':
    opt = getConfig()
    opt = vars(opt)
    print(opt)