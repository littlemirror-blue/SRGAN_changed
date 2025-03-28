import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # 加载预训练的 VGG16 模型
        vgg = vgg16(pretrained=True)
        # 使用 VGG16 的前 31 层作为损失网络
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # 冻结损失网络的参数
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()  # 均方误差损失
        self.tv_loss = TVLoss()  # 全变差损失

    def forward(self, out_labels, out_images, target_images):
        # 将灰度图像复制为 3 通道
        if out_images.size(1) == 1:
            out_images = out_images.repeat(1, 3, 1, 1)  # 将 1 通道复制为 3 通道
        if target_images.size(1) == 1:
            target_images = target_images.repeat(1, 3, 1, 1)  # 将 1 通道复制为 3 通道

        # Adversarial Loss（对抗损失）
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss（感知损失）
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss（图像损失）
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss（全变差损失）
        tv_loss = self.tv_loss(out_images)

        # 总损失
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)