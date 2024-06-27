from torch import nn
import torch
from networks.docr.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from networks.docr.unet import SaveFeatures, UnetBlock


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.bnin = nn.BatchNorm2d(32)
        layers = list(base_model(pretrained=pretrained).children())[:8]
        base_layers = nn.Sequential(*layers)
        self.res = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]

        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

    def forward(self, input):
        x = F.relu(self.res(input))
        content = x

        x = self.up1(x, self.sfs[3].features.to(x.device))
        x = self.up2(x, self.sfs[2].features.to(x.device))
        x = self.up3(x, self.sfs[1].features.to(x.device))
        x = self.up4(x, self.sfs[0].features.to(x.device))
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))

        seg_output = self.seg_head(head_input)

        return seg_output, content, [self.sfs[0].features, self.sfs[1].features]

    def close(self):
        for sf in self.sfs: sf.remove()


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=True, IBN=True)
    print(model.res)
    model.eval()
    input = torch.rand(2, 3, 512, 512)
    seg_output, content, style = model(input)
    print(seg_output.size(), style[0].size(), style[1].size(), content.size())

