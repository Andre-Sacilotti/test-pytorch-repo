import torch

class RedeLegal(torch.nn.Module):
  def __init__(self):
    super(RedeLegal, self).__init__()

    self.conv1 = torch.nn.Conv2d(1, 30, 4)
    self.bn = torch.nn.BatchNorm2d(30)
    self.pooling = torch.nn.AvgPool2d(24)
    self.conv2 = torch.nn.Conv2d(30, 10, 2)
    self.act = torch.nn.ReLU()
    self.sigmoid = torch.nn.Softmax(dim=1)
  
  def forward(self, img):
    out = self.conv1(img)
    out = self.bn(out)
    out = self.act(out)
    out = self.conv2(out)
    out = self.pooling(out).view((img.size(0), out.size(1)))
    out = self.sigmoid(out)
    return out