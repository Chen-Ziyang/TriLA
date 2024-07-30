# coding:utf-8
import torch
from torchnet import meter
from torch.autograd import Variable
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from networks.deeplabv3 import DeepLab
from networks.ResUnet_DA import ResUnet
from config import *
from utils.metrics import calculate_metrics
from torch.nn import functional as F
from utils.train_target_aug import *


class TrainNoAdapt:
    def __init__(self, config, train_loader, test_loader):
        # 数据加载
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 模型
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type

        # 损失函数
        self.lossmap = config.lossmap
        self.seg_cost = Seg_loss(self.lossmap)

        # 优化器
        self.optimizer = None
        self.scheduler = None
        self.optim = config.optimizer
        self.lr_scheduler = config.lr_scheduler
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.betas = (config.beta1, config.beta2)

        # 训练设置
        self.reload = config.reload
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # 路径设置
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.savefig = config.savefig

        # 其他
        self.warm_up = -1
        self.valid_frequency = 10   # 多少轮测试一次
        self.mode = config.mode
        self.device = config.device

        self.build_model()
        self.print_network()

    def build_model(self):
        if self.model_type == 'Deeplab_Mobile':
            self.model = DeepLab(num_classes=self.out_ch, backbone='mobilenet', output_stride=16,
                                 sync_bn=True, freeze_bn=False).to(self.device)
        elif self.model_type == 'Deeplab_Res':
            self.model = DeepLab(num_classes=self.out_ch, backbone='resnet50', output_stride=16,
                                 sync_bn=True, freeze_bn=False).to(self.device)
        elif self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet='resnet34', num_classes=self.out_ch, pretrained=True).to(self.device)
        else:
            raise ValueError('The model type is wrong!')

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )

        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        if torch.cuda.device_count() > 1:
            device_ids = list(range(0, torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        if self.lr_scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)
        elif self.lr_scheduler == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        elif self.lr_scheduler == 'Epoch':
            self.scheduler = EpochLR(self.optimizer, epochs=self.num_epochs, gamma=0.9)
        else:
            self.scheduler = None

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        loss_meter = meter.AverageValueMeter()
        # 绘制loss曲线
        metrics_l = {'Disc_ACC': [], 'Disc_HD': [], 'Disc_Dice': [], 'Disc_Iou': [],
                     'Cup_ACC': [], 'Cup_HD': [], 'Cup_Dice': [], 'Cup_Iou': [], 'MAE_CDR': []}
        metric_dict = ['Disc_ACC', 'Disc_HD', 'Disc_Dice', 'Disc_Iou', 'Cup_ACC', 'Cup_HD', 'Cup_Dice', 'Cup_Iou', 'MAE_CDR']

        for epoch in range(self.num_epochs):
            self.model.train()
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Training...")
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            loss_meter.reset()
            metrics_y = [[], [], [], [], [], [], [], [], []]
            for batch, data in enumerate(self.train_loader):
                CS_SS, y = data['data'], data['mask']
                CS_SS = torch.from_numpy(normalize_image_to_0_1(CS_SS)).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)
                y = Variable(y).to(self.device)
                CS_SS = Variable(CS_SS).to(self.device)   # Content: Source, Style: Source
                seg_SS, _, _ = self.model(CS_SS)
                seg_loss_SS = self.seg_cost(seg_SS, y)
                loss = seg_loss_SS

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                loss_meter.add(loss.item())

                seg_output = torch.sigmoid(seg_SS.detach())
                metrics = calculate_metrics(seg_output.cpu(), y.cpu())
                for i in range(len(metrics)):
                    metrics_y[i].append(metrics[i])

            if self.scheduler is not None:
                self.scheduler.step()

            print_train = {}
            train_metrics_y = np.sum(metrics_y, axis=1) / len(self.train_loader)

            for i in range(len(train_metrics_y)):
                metrics_l[metric_dict[i]].append(train_metrics_y[i])
                print_train[metric_dict[i]] = train_metrics_y[i]

            print("Train ———— Total_Loss:{:.8f}".format(loss_meter.value()[0]))
            print("Train Metrics: ", print_train)

            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), self.model_path + '/' + str(epoch + 1) + '-' + self.model_type + '.pth')
            else:
                torch.save(self.model.state_dict(), self.model_path + '/' + str(epoch + 1) + '-' + self.model_type + '.pth')

            if (epoch + 1) % self.valid_frequency == 0:
                self.model.eval()
                metrics_test = [[], [], [], [], [], [], [], [], []]
                with torch.no_grad():
                    for batch, data in enumerate(self.test_loader):
                        x, y = data['data'], data['mask']
                        x = torch.from_numpy(normalize_image_to_0_1(x)).to(dtype=torch.float32)
                        y = torch.from_numpy(y).to(dtype=torch.float32)
                        x, y = Variable(x).to(self.device), Variable(y).to(self.device)

                        if x.shape[0] <= 1:
                            x = torch.cat((x, x), 0)
                            y = torch.cat((y, y), 0)

                        seg_logit, _, _ = self.model(x)
                        seg_output = torch.sigmoid(seg_logit)
                        metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
                        for i in range(len(metrics)):
                            metrics_test[i].append(metrics[i])
                test_metrics_y = np.mean(metrics_test, axis=1)
                print_test_metric = {}
                for i in range(len(test_metrics_y)):
                    print_test_metric[metric_dict[i]] = test_metrics_y[i]
                print("Test Metrics: ", print_test_metric)
