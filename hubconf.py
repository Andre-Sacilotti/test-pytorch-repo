from src.models.modelo1 import RedeLegal
from src.models.modelo2 import CNN


def rede_1(pretrained=True):
    return RedeLegal()

def rede_2(pretrained=True):
    return CNN()