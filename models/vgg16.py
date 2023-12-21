import torch
import torch.nn as nn

from models.blocks.vgg16_blocks import conv_block, classifier_block


class VGG16(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """https://arxiv.org/pdf/1409.1556.pdf"""
        super(VGG16, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # TODO: инициализируйте сверточные слои модели, используя функцию conv_block
        self.conv1 = conv_block([3, 64], [64, 64])
        self.conv2 = conv_block([64, 128], [128, 128])
        ...  # еще?

        # TODO: инициализируйте полносвязные слои модели, используя функцию classifier_block
        #  (последний слой инициализируется отдельно)
        self.linears = classifier_block([128, 128], [128, 128])  #  [4096, 4096], [4096, 4096])

        # TODO: инициализируйте последний полносвязный слой для классификации с помощью
        #  nn.Linear(in_features=4096, out_features=nrof_classes)
        self.classifier = nn.Linear(in_features=4096, out_features=nrof_classes)

        # raise NotImplementedError

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight)
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
        """
        # raise NotImplementedError
        x = self.conv1(inputs)
        x = self.conv2(x)
        print(x.size())
        # x = x.view(x.size(0), -1)
        print(x.size())
        x = self.linears(x)
        print(x.size())
        output = self.classifier(x)

        return output
