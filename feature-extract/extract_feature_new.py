import torch
from ava import AVA
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as TF
import random
import torch.backends.cudnn as cudnn


def forward_pass(data, model, fc_layer):
    features = []
    # outputs = []

    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        # outputs.append(output.detach().cpu())
    
    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    model.eval()
    with torch.no_grad():
        data = data.cuda()
        _ = model(data)
    
    forward_hook.remove()
    features = torch.cat([x for x in features])

    return features

def options():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg16', type=str)
    parser.add_argument('--ptm_dataset', default='cars', type=str)
    parser.add_argument('--data_class', default='freeway', type=str)
    parser.add_argument('--dataloader_type', default='train', type=str)
    parser.add_argument('--box_type', default='center', type=str)
    # parser.add_argument('--load_model', action='store_true', default=False)

    parser.add_argument('--seed', default=0 , type=int)
    parser.add_argument('--gpu', type=str, default='0')
    return parser

def set_seed(seed):

    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def set_gpu(x, space_hold=1000):
    import os
    import time
    assert torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    torch.backends.cudnn.benchmark = True
    gpu_available = 0
    while gpu_available < space_hold:
        gpu_space_available = gpu_state(x, get_return=True)
        for gpu_id, space in gpu_space_available.items():
            gpu_available += space
        if gpu_available < space_hold:
            gpu_available = 0
            time.sleep(1800) # 间隔30分钟.
    gpu_state(x)

def gpu_state(gpu_id, get_return=False):
    import os
    qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

    results = os.popen(cmd).readlines()
    gpu_id_list = gpu_id.split(",")
    gpu_space_available = {}
    for cur_state in results:
        cur_state = cur_state.strip().split(", ")
        for i in gpu_id_list:
            if i == cur_state[0]:
                if not get_return:
                    print(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')
                else:
                    gpu_space_available[i] = int("".join(list(filter(str.isdigit, cur_state[3])))) - int("".join(list(filter(str.isdigit, cur_state[2]))))
    if get_return:
        return gpu_space_available
    

class MyColorJitterTransform:
    """Rotate by one of the given angles."""

    def __init__(self, factors):
        self.factors = factors
        self.brightness, self.contrast, self.saturation, self.hue = factors

    def __call__(self, x):
        x = TF.adjust_brightness(x, self.brightness)
        x = TF.adjust_contrast(x, self.contrast)
        x = TF.adjust_saturation(x, self.saturation)
        x = TF.adjust_hue(x, self.hue)
        return x
    
class MyGaussianBlurTransform:
    def __init__(self, kernel_size, blur_config_flag):
        self.kernel_size = [kernel_size, kernel_size]
        self.blur_config_flag = blur_config_flag
    def __call__(self, x):
        if blur_config_flag > 0.7:
            return TF.gaussian_blur(x, self.kernel_size)
        else:
            return x
        
        
class MyFlipTransform:
    def __init__(self, p):
        self.p = p
    def __call__(self, x):
        if self.p > 0.5:
            return TF.hflip(x)
        else:
            return x
        
if __name__ == "__main__":

    args = options().parse_args()
    set_seed(args.seed)
    set_gpu(args.gpu)
    torch.cuda.set_device(int(args.gpu))

    # dataloader_type = 'train'
    # data_class = 'freeway'

    label_onehot = [[1, 0], [0, 1]]
    dataset = AVA(args.dataloader_type, args.data_class, args.box_type)


    
    train_dataloader = DataLoader(dataset=dataset, batch_size=1)

    # train_transform = transforms.Compose([
    #                             transforms.ToPILImage(),
    #                             transforms.Resize((250, 250)),
    #                             # transforms.Resize(250),
    #                             transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0)),
    #                             transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
    #                         ])
    test_transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                            ])
                            
    if args.ptm_dataset == 'imagenet':
        if args.model == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif args.model == 'vgg16':
            model = models.vgg16(pretrained=True)
    elif args.ptm_dataset == 'cars':
        if args.model == 'resnet101':
            model = models.resnet101()
        elif args.model == 'vgg16':
            model = models.vgg16()
        model.load_state_dict(torch.load(f'./{args.model}_ft_cars.pth')['model'])
    else:
        raise NotImplementedError
    model.cuda()
    model.eval()
    
    with tqdm(total=len(train_dataloader)) as t:
        for index, (images, labels, info, video_name) in enumerate(tqdm(train_dataloader)):
            video_name = video_name[0]
            if os.path.exists(f'/data/yehj/SAVES/DSTA/{args.model}_features_{args.ptm_dataset}_{args.box_type}_{args.seed}/{args.dataloader_type}/{args.data_class}/{video_name}.npz'):
                continue
            images, labels = images[0], labels[0]   # check batch_size == 1
            video_features = []
            det = []

            color_config = (random.randint(5,15)/10, random.randint(5,15)/10, random.randint(5,15)/10, random.randint(-3,3)/10)
            blur_config = random.choice([3,5,7,9])
            blur_config_flag = random.random()
            flip_config = random.random()
            train_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        MyColorJitterTransform(color_config),
                                        MyGaussianBlurTransform(blur_config, blur_config_flag),
                                        MyFlipTransform(flip_config),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                                    ])
            transform = train_transform if args.dataloader_type == 'train' else test_transform

            for cur_image, cur_label, cur_info in zip(images, labels, info):
                image_boxes = []
                image_boxes.append(cur_image)
                for cur_key in cur_info.keys():
                    values = cur_info[cur_key]
                    xmin,ymin,xmax,ymax = max(values[0][0], torch.tensor(0)), max(values[1][0], torch.tensor(0)), min(values[2][0], torch.tensor(cur_image.shape[1]-1)), min(values[3][0], torch.tensor(cur_image.shape[0]-1))
                    if xmin >= xmax or ymax <= ymin:
                        break
                    image_boxes.append(cur_image[ymin:ymax,xmin:xmax,:])
                    # print(cur_image[ymin:ymax,xmin:xmax,:].shape)
                if len(image_boxes) < 20:
                    break               
                # image_boxes.extend([cur_image[max(values[1][0], torch.tensor(0)): min(values[3][0], torch.tensor(cur_image.shape[1]-1)), max(values[0][0], torch.tensor(0)): min(values[2][0], torch.tensor(cur_image.shape[0]-1)), :] for _, values in cur_info.items()])
                
                image_boxes = torch.stack([transform(np.array(i)) for i in image_boxes]) # 20 x 3 x 224 x 224
                if args.model == 'vgg16':
                    frame_feature = forward_pass(image_boxes, model, model.classifier[6])
                else:
                    frame_feature = forward_pass(image_boxes, model, model.fc)  # 20 x 4096
                video_features.append(frame_feature)
                cur_det = [np.array([values[0][0].item(), values[1][0].item(), values[2][0].item(), values[3][0].item(), 0, 0]) for _, values in cur_info.items()]
                det.append(np.stack(cur_det))   # np.stack(cur_det): 19 x 6

            # saves = {
            #     'data': np.array(torch.stack(video_features)),  # 60 x 20 x 4096
            #     'label': label_onehot[int(labels[0])],  # 2
            #     'det':  np.stack(det), # 60 x 19 x 6
            #     'ID': video_name
            #     }
            if len(det) < 19:
                with open('./error_video.txt', 'a') as f:
                        f.write(f'{args.dataloader_type},{args.data_class},{video_name}\n')
                continue
            assert np.stack(det).shape[1] == 19

            save_dir = Path(f'/data/yehj/SAVES/DSTA/{args.model}_features_{args.ptm_dataset}_{args.box_type}_{args.seed}/{args.dataloader_type}/{args.data_class}/')
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez(os.path.join(save_dir, f'{video_name}'), data=np.array(torch.stack(video_features)), label=label_onehot[int(labels[0])], det=np.stack(det), ID=video_name)
