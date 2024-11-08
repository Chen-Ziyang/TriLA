# coding:utf-8
import torch
import torch.nn as nn
from torchnet import meter
from torch.autograd import Variable
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from networks.deeplabv3 import DeepLab
from networks.ResUnet_DA import ResUnet
from networks.LFDA import LFDA
from networks.vae import VAE
from config import *
from utils.metrics import calculate_metrics
from tensorboardX import SummaryWriter


def future_fusion(style, content):
    fusion = content
    for i in [1, 0]:
        s = nn.AdaptiveAvgPool2d(fusion.size()[2:])(style[i])
        fusion = torch.cat((s, fusion), dim=1)
    return fusion


class TrainDA:
    def __init__(self, config, train_loader, target_train_dataloader, test_dataloader):
        # Data loading
        self.train_loader = train_loader               # Source dataset
        self.target_loader = target_train_dataloader   # Target training dataset
        self.test_loader = test_dataloader             # Target test dataset

        # Model
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type

        # Loss function
        self.lossmap = config.lossmap
        self.vae_cost = VAE_loss(h1=1., h2=1., mode='mean')
        self.seg_cost = Seg_loss(self.lossmap)
        self.content_cost = Content_loss()
        self.style_cost = Style_loss()

        # Weighting coefficients
        self.vae_coef = config.vae_coef
        self.content_coef = config.content_coef
        self.style_coef = config.style_coef
        self.output_coef = config.output_coef

        # LFDA Upper bound and lower bound
        self.upbound = config.upbound
        self.lowbound = config.lowbound

        # Optimizer
        self.optimizer = None
        self.scheduler = None
        self.optim = config.optimizer
        self.lr_scheduler = config.lr_scheduler
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.betas = (config.beta1, config.beta2)

        # Training
        self.reload = config.reload
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # Root
        self.log_path = config.log_path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.savefig = config.savefig

        self.valid_frequency = 10

        self.device = config.device

        self.build_model()
        self.print_network()

    def build_model(self):
        if self.model_type == 'Deeplab_Mobile':
            self.model = DeepLab(num_classes=self.out_ch, backbone='mobilenet', output_stride=16,
                                 sync_bn=True, freeze_bn=False).to(self.device)
            self.vae = VAE(in_channels=376, in_dim=(self.image_size // 16, self.image_size // 16),
                           out_dim=(3, self.image_size, self.image_size)).to(self.device)
        elif self.model_type == 'Deeplab_Res':
            self.model = DeepLab(num_classes=self.out_ch, backbone='resnet50', output_stride=16,
                                 sync_bn=True, freeze_bn=False).to(self.device)
        elif self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet='resnet34', num_classes=self.out_ch, pretrained=True).to(self.device)
            self.vae = VAE(in_channels=640, in_dim=(self.image_size // 32, self.image_size // 32),
                           out_dim=(3, self.image_size, self.image_size)).to(self.device)
        else:
            raise ValueError('The model type is wrong!')
        self.lfda = LFDA(image_size=self.image_size, upbound=self.upbound, lowbound=self.lowbound).to(self.device)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )

            self.vae_optimizer = torch.optim.SGD(
                self.vae.parameters(),
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

            self.vae_optimizer = torch.optim.Adam(
                self.vae.parameters(),
                lr=self.lr,
                betas=self.betas
            )
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

            self.vae_optimizer = torch.optim.AdamW(
                self.vae.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        self.lfda_optimizer = torch.optim.SGD(
            self.lfda.parameters(),
            lr=0.1,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        if torch.cuda.device_count() > 1:
            device_ids = list(range(0, torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            self.vae = torch.nn.DataParallel(self.vae, device_ids=device_ids)
            self.lfda = torch.nn.DataParallel(self.lfda, device_ids=device_ids)

        if self.lr_scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)
        elif self.lr_scheduler == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
            self.vae_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        elif self.lr_scheduler == 'Epoch':
            self.scheduler = EpochLR(self.optimizer, epochs=self.num_epochs, gamma=0.9)
            self.vae_scheduler = EpochLR(self.optimizer, epochs=self.num_epochs, gamma=0.9)
        else:
            self.scheduler = None
            self.vae_scheduler = None

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        for p in self.vae.parameters():
            num_params += p.numel()

        # print(model)
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        writer = SummaryWriter(self.log_path.replace('.log', '.writer'))
        best_mean_dice, best_epoch = 0, 0
        loss_meter = meter.AverageValueMeter()
        # 绘制loss曲线
        loss_l, test_loss_l = [], []
        metrics_l = {'Disc_ACC': [], 'Disc_HD': [], 'Disc_Dice': [], 'Disc_Iou': [],
                     'Cup_ACC': [], 'Cup_HD': [], 'Cup_Dice': [], 'Cup_Iou': [], 'MAE_CDR': []}
        metric_dict = ['Disc_ACC', 'Disc_HD', 'Disc_Dice', 'Disc_Iou', 'Cup_ACC', 'Cup_HD', 'Cup_Dice', 'Cup_Iou', 'MAE_CDR']

        for epoch in range(self.num_epochs):
            segconsis_loss, content_loss, style_loss, vae_loss = 0., 0., 0., 0.
            self.model.train()
            self.vae.train()
            self.lfda.train()
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Training...")
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            loss_meter.reset()
            metrics_y = [[], [], [], [], [], [], [], [], []]
            for batch, data in enumerate(self.train_loader):
                self.lfda_optimizer.zero_grad()
                self.vae_optimizer.zero_grad()
                self.optimizer.zero_grad()

                # 1. train generator with images from different domain
                for para in self.model.parameters():
                    para.requires_grad = True
                for para in self.vae.parameters():
                    para.requires_grad = True
                for para in self.lfda.parameters():
                    para.requires_grad = True

                try:
                    _, data_T = next(domain_t_loader)
                except:
                    domain_t_loader = enumerate(self.target_loader)
                    _, data_T = next(domain_t_loader)

                CT_ST = data_T['data']
                CT_ST = torch.from_numpy(normalize_image_to_0_1(CT_ST)).to(dtype=torch.float32)

                CS_SS, y = data['data'], data['mask']
                CS_SS = torch.from_numpy(normalize_image_to_0_1(CS_SS)).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)
                y = Variable(y).to(self.device)

                CS_SS = Variable(CS_SS).to(self.device)   # Content: Source, Style: Source
                CT_ST = Variable(CT_ST).to(self.device)   # Content: Target, Style: Target

                CT_SS, CS_ST = self.lfda(input_S=CS_SS, input_T=CT_ST)
                CT_SS, CS_ST = normalize_image_to_0_1(CT_SS), normalize_image_to_0_1(CS_ST)

                seg_SS, content_SS, style_SS = self.model(CS_SS)
                seg_ST, content_ST, style_ST = self.model(CS_ST)
                seg_TS, content_TS, style_TS = self.model(CT_SS)
                seg_TT, content_TT, style_TT = self.model(CT_ST)

                CS_SS_FMAP_1 = future_fusion(content=content_SS, style=style_SS)
                CS_SS_FMAP_2 = future_fusion(content=content_ST, style=style_TS)
                CT_ST_FMAP_1 = future_fusion(content=content_TT, style=style_TT)
                CT_ST_FMAP_2 = future_fusion(content=content_TS, style=style_ST)

                z_CS_SS_1, recon_CS_SS_1, mu_CS_SS_1, logvar_CS_SS_1 = self.vae(CS_SS_FMAP_1)
                z_CS_SS_2, recon_CS_SS_2, mu_CS_SS_2, logvar_CS_SS_2 = self.vae(CS_SS_FMAP_2)
                z_CT_ST_1, recon_CT_ST_1, mu_CT_ST_1, logvar_CT_ST_1 = self.vae(CT_ST_FMAP_1)
                z_CT_ST_2, recon_CT_ST_2, mu_CT_ST_2, logvar_CT_ST_2 = self.vae(CT_ST_FMAP_2)

                vae_loss_original = (
                                  self.vae_cost(x=CS_SS, vae_out=recon_CS_SS_1, mu=mu_CS_SS_1, logvar=logvar_CS_SS_1) +
                                  self.vae_cost(x=CT_ST, vae_out=recon_CT_ST_1, mu=mu_CT_ST_1, logvar=logvar_CT_ST_1)
                )
                vae_loss_generate = (
                                  self.vae_cost(x=CS_SS, vae_out=recon_CS_SS_2, mu=mu_CS_SS_2, logvar=logvar_CS_SS_2) +
                                  self.vae_cost(x=CT_ST, vae_out=recon_CT_ST_2, mu=mu_CT_ST_2, logvar=logvar_CT_ST_2)
                )
                vae_loss_total = vae_loss_original + vae_loss_generate
                vae_loss += vae_loss_total.data.item()

                seg_loss_SS = self.seg_cost(seg_SS, y)
                seg_loss_ST = self.seg_cost(seg_ST, y)

                segment_consistent = (
                        torch.nn.BCELoss()(torch.sigmoid(seg_TS), torch.sigmoid(seg_TT.detach())) +
                        torch.nn.BCELoss()(torch.sigmoid(seg_ST), torch.sigmoid(seg_SS.detach()))
                )
                segconsis_loss += segment_consistent.data.item()

                content_loss_same = (self.content_cost(content_ST, content_SS) +
                                     self.content_cost(content_TS, content_TT))
                content_loss_total = content_loss_same
                content_loss += content_loss_total.data.item()

                style_loss_total = 0
                layer_weight = 1. / len(style_SS)
                for i in range(len(style_SS)):
                    style_loss_same = (self.style_cost(style_ST[i], style_TT[i], 'same') +
                                       self.style_cost(style_TS[i], style_SS[i], 'same'))
                    style_loss_total += layer_weight * style_loss_same
                style_loss += style_loss_total.data.item()

                if (epoch+1) % 30 == 0 and self.content_coef < self.style_coef:
                    self.content_coef *= 10.

                loss = (seg_loss_SS + seg_loss_ST                                                        # input-level
                        + self.output_coef * segment_consistent                                          # output-level
                        + self.content_coef * content_loss_total + self.style_coef * style_loss_total    # feature-level
                        + self.vae_coef * vae_loss_original                                              # L_rep
                        )

                loss_meter.add(loss.item())
                loss.backward(retain_graph=True)

                for para in self.model.parameters():
                    para.requires_grad = False

                self.lfda_optimizer.zero_grad()     # Train lfda with the generate loss only
                vae_loss_generate.backward()        # L_LFDA

                for para in self.lfda.parameters():
                    para.requires_grad = False

                self.vae_optimizer.zero_grad()      # Train vae with the original loss only
                vae_loss_original.backward()        # L_rep

                self.optimizer.step()
                self.lfda_optimizer.step()
                self.vae_optimizer.step()

                seg_output = torch.nn.Sigmoid()(seg_SS.detach())
                metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
                for i in range(len(metrics)):
                    metrics_y[i].append(metrics[i])

            train_segconsis_loss = segconsis_loss / len(self.train_loader)
            train_content_loss = content_loss / len(self.train_loader)
            train_style_loss = style_loss / len(self.train_loader)
            train_vae_loss = vae_loss / len(self.train_loader)

            if self.scheduler is not None:
                self.scheduler.step()
            if self.vae_scheduler is not None:
                self.vae_scheduler.step()

            print_train, print_test = {}, {}
            train_metrics_y = np.sum(metrics_y, axis=1) / len(self.train_loader)

            for i in range(len(train_metrics_y)):
                metrics_l[metric_dict[i]].append(train_metrics_y[i])
                print_train[metric_dict[i]] = train_metrics_y[i]

            loss_l.append(loss_meter.value()[0])

            try:
                lfda_data = self.lfda.module.activate(self.lfda.module.fda_beta).data
            except:
                lfda_data = self.lfda.activate(self.lfda.fda_beta).data
            print("Train ———— Total_Loss:{:.8f}, Segconsis_Loss:{:.8f}, Content_Loss:{:.8f}, Style_Loss:{:.8f}, "
                  "VAE_Loss:{:.8f}, FDA_beta:{}".
                  format(loss_meter.value()[0], train_segconsis_loss, train_content_loss, train_style_loss,
                         train_vae_loss, lfda_data))
            print("Train Metrics: ", print_train)

            if (epoch + 1) % self.valid_frequency == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), self.model_path + '/' + str(epoch + 1) + '-' + self.model_type + '.pth')
                    torch.save(self.vae.module.state_dict(), self.model_path + '/' + str(epoch + 1) + '-' + 'vae.pth')
                else:
                    torch.save(self.model.state_dict(), self.model_path + '/' + str(epoch + 1) + '-' + self.model_type + '.pth')
                    torch.save(self.vae.state_dict(), self.model_path + '/' + str(epoch + 1) + '-' + 'vae.pth')

                print('Model Save: ' + str(epoch + 1) + '-' + self.model_type + '.pth')
                print('VAE Save: ' + str(epoch + 1) + '-' + 'vae.pth')

                self.model.eval()
                metrics_test = [[], [], [], [], [], [], [], [], []]
                with torch.no_grad():
                    for batch, data in enumerate(self.test_loader):
                        x, y = data['data'], data['mask']
                        x = torch.from_numpy(normalize_image_to_0_1(x)).to(dtype=torch.float32)
                        y = torch.from_numpy(y).to(dtype=torch.float32)
                        x, y = Variable(x).to(self.device), Variable(y).to(self.device)

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
                writer.add_scalar('Disc Dice', test_metrics_y[2], epoch + 1)
                writer.add_scalar('Cup Dice', test_metrics_y[6], epoch + 1)

                print("===" * 10)

