import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math, time

# self-defined module
import Py_autopilot2_module

debug = True

def time_it(output=''):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            debug and print(output + '{:.6f}s'.format(elapsed))
            return result
        return wrapper
    return decorator

def rgb2hsv(img):
    # cv2.resize, 参数输入是 W×H×C
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV)), (100, 100))
    return resized

def angle2pi(angle):
    return float(angle) * math.pi / 180

def prepare_data(input_path):
    features = []
    # Read Image, RGB mode
    img = plt.imread(input_path)
    # convert image to hsv
    features.append(rgb2hsv(img))
    features = np.array(features).astype('float32')
    return features

def substitute_value(angle):
    # return substitute value when angle is None
    v = 0.0
    return np.array([angle2pi(angle)]).astype('float32') if angle is not None else np.array([angle2pi(v)]).astype('float32')

@time_it('[Metrics][TIME] Load Data: ')
def load_data(model_path, input_path, angle):
    features = prepare_data(input_path)
    #load network
    test_net = torch.load(model_path)
    test_net.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    test_net = test_net.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),               ## shape:(31784, 100, 100, 3) -->(31784, 3,100, 100)
        transforms.Normalize((0.43567735,0.49298695,0.5192303), (0.22272868,0.24110594,0.29045662))     
    ])
    dataset = Py_autopilot2_module.Custom_Dataset(features, substitute_value(angle), transform = transform)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers=0)
    return {
        "dataloader": dataloader,
        "network": test_net,
        "device": device,
    }

@time_it('[Metrics][TIME] Running: ')
def test(dataloader, network, device):
    net_out_rec = []
    steer_rec = []
    test_loss = []
    test_accr_rec = []
    out_res = []
    for i, data in enumerate(dataloader):
        img, steer = data
        img, steer = img.to(device), steer.to(device)

        net_out = network.forward(img).squeeze().float() 
        steer = steer.squeeze()
        steer_pred_angle = net_out.detach().cpu().numpy() / math.pi * 180
        debug and print("i=", i, "Pridect =", steer_pred_angle)
        out_res.append(steer_pred_angle)
        net_out_rec.append(net_out.detach().cpu().numpy() / math.pi * 180 )
        steer_rec.append(steer.detach().cpu().numpy() / math.pi * 180)
        loss = network.steering_acc_loss(net_out.detach().cpu(),  steer.detach().cpu().float())  #double -> float , tensor
        loss_result = loss.detach().cpu().numpy()
        test_loss.append(loss_result)
        # get accuracy
        accr_temp= 1 - np.abs(steer.detach().cpu().numpy() - net_out.detach().cpu().numpy()) / math.pi
        test_accr_rec.append(accr_temp)
    test_loss_mean = np.mean(test_loss)
    debug and print("test loss mean =", test_loss_mean)
    test_accr_mean = np.mean(test_accr_rec)
    debug and print("test accr mean =", test_accr_mean)
    debug and print("steer_rec =", steer_rec)
    return {
        "net_out_rec": net_out_rec,
        "steer_rec": steer_rec,
        "test_loss": test_loss,
        "test_accr_rec": test_accr_rec,
        "res": out_res,
    }
    



def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(add_help=True)

    # 添加命令行参数
    parser.add_argument('--model', '-m', required=True, help='input model path')
    parser.add_argument('--input', '-i', required=True, help='input image path')
    parser.add_argument('--angle', '-a', type=float, help='Set target angle to get loss and accuracy')
    
    # 解析命令行参数
    args = parser.parse_args()

    # 获取命令行参数的值
    model_path = args.model
    input_path = args.input
    angle = args.angle

    # load data
    data = load_data(model_path, input_path, angle)
    dataloader = data["dataloader"]
    network = data["network"]
    device = data["device"]
    # run model test
    res = test(dataloader, network, device)
    
    debug and print("Predicted steering angle: ", res["res"])
    return res["res"]


if __name__ == '__main__':
    res = main()
    print(res)
