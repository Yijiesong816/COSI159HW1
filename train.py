import os
import time


import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter

from PIL import Image
from torchvision.transforms import transforms
from torch.autograd import Variable


class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))

        elapse = time.time() - tik
        print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        return

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """
        self._model.eval()
        accuracy = 0.0
        total = 0.0
        correct_ones = 0

        with torch.no_grad():
            self._model.eval()
            for test_data, test_target in test_loader:
                test_output = self._model(test_data)

                _, predicted = torch.max(test_output.data, dim=1)

                total += test_data.size(0)
                "add up individual correctness"
                correct_ones += (predicted == test_target).sum().item()
        accuracy = (100*correct_ones / total)
        print('Accuracy is %d %%' % accuracy)
        return

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        print("Prediction in progress")
        images, labels = sample
        actual_number = labels[1].numpy()

        
        with torch.no_grad():
            test_output = self._model(images[1])
            pred_y = torch.max(test_output, 1)
        

        print("the index of the image is:", pred_y)
        print('Actual number:', actual_number)
        return


    def load_model(self, path: str) -> None:
        """ load model from a .pth file """

        path = "mnist.pth"
        self._model.load_state_dict(torch.load(path))
        return

