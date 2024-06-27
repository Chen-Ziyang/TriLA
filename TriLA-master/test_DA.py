# coding:utf-8
import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from torch.autograd import Variable
from networks.deeplabv3 import DeepLab
from networks.ResUnet_DA import ResUnet
from utils.metrics import calculate_metrics
import datetime


class TestDA:
    def __init__(self, config, test_loader):
        # 数据加载
        self.test_loader = test_loader

        # 模型
        self.model = None
        self.model_type = config.model_type
        self.reload = config.reload

        # 路径设置
        self.target = config.Target_Dataset
        self.result_path = config.result_path
        self.model_path = config.model_path

        # 其他
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.mode = config.mode
        self.device = config.device

        self.build_model()
        self.print_network(self.model)

    def build_model(self):
        if self.model_type == 'Deeplab_Mobile':
            self.model = DeepLab(num_classes=self.out_ch, backbone='mobilenet', output_stride=16,
                                 sync_bn=True, freeze_bn=False).to(self.device)
        elif self.model_type == 'Deeplab_Res':
            self.model = DeepLab(num_classes=self.out_ch, backbone='resnet50', output_stride=16,
                                 sync_bn=True, freeze_bn=False).to(self.device)
        elif self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet='resnet34', num_classes=self.out_ch, pretrained=True,
                                 ).to(self.device)
            # self.model = ResUnet_ISW(resnet='resnet34', num_classes=self.out_ch, pretrained=True,
            #                          IBN=self.IBN).to(self.device)
        else:
            raise ValueError('The model type is wrong!')

        checkpoint = torch.load(self.model_path + '/' + str(self.reload) + '-' + self.model_type + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))
        self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of parameters: {}".format(num_params))

    def test(self):
        print("Testing and Saving the results... Domain Generalization Phase")
        print("--" * 15)
        metrics_y = [[], [], [], [], [], [], [], [], []]
        metric_dict = ['Disc_ACC', 'Disc_HD', 'Disc_Dice', 'Disc_Iou', 'Cup_ACC', 'Cup_HD', 'Cup_Dice', 'Cup_Iou', 'MAE_CDR']

        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                x, y, path = data['data'], data['mask'], data['name']
                x = torch.from_numpy(normalize_image_to_0_1(x)).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x, y = Variable(x).to(self.device), Variable(y).to(self.device)

                if x.shape[0] <= 1:
                    x = torch.cat((x, x), 0)
                    y = torch.cat((y, y), 0)

                seg_logit, _, _ = self.model(x)

                seg_output = torch.sigmoid(seg_logit)
                metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu(), self.target)

                for i in range(len(metrics)):
                    metrics_y[i].append(metrics[i])

                draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
                img_name = path[0].split('/')[-2] + '-' + path[0].split('/')[-1].split('.')[0]
                # Optic Cup
                cv2.imwrite(self.result_path + '/' + img_name + '_OC.png', draw_output[0][1])
                # Optic Disc
                cv2.imwrite(self.result_path + '/' + img_name + '_OD.png', draw_output[0][0])

        test_metrics_y = np.mean(metrics_y, axis=1)
        print_test_metric = {}
        for i in range(len(test_metrics_y)):
            print_test_metric[metric_dict[i]] = test_metrics_y[i]

        with open(self.result_path+'/'+'metrics'+'.txt', 'w', encoding='utf-8') as f:
            f.write('Disc HD95\n')
            f.write(str(metrics_y[1])+'\n')  # Disc HD95
            f.write('Disc Dice\n')
            f.write(str(metrics_y[2])+'\n')  # Disc Dice
            f.write('Cup HD95\n')
            f.write(str(metrics_y[5])+'\n')  # Cup HD95
            f.write('Cup Dice\n')
            f.write(str(metrics_y[6])+'\n')  # Cup Dice

        print("Test Metrics: ", print_test_metric)

