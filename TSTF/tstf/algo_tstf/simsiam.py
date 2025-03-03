import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


class SimSiam(nn.Module):
    def __init__(self, base_encoder, output_size=1024, pred_dim=256):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.encoder = base_encoder
        # build a 3-layer projector
        prev_dim = self.encoder.num_hiddens
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True),  # first layer
                                       nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True),  # second layer
                                       nn.Linear(prev_dim, output_size, bias=False),
                                       nn.BatchNorm1d(output_size, affine=False))  # output layer
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(output_size, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, output_size))  # output layer

    def forward(self, x1, x2):
        # compute features for one view
        x1 = self.encoder(x1)
        z1 = self.projector(x1)
        # x2 = torch.flip(x2, dims=[1])
        x2 = self.encoder(x2)
        z2 = self.projector(x2)

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()
