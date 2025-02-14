import torch
import torch.nn as nn
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjection2D


class TrainableFrequencyResponse(nn.Module):
    def __init__(self, num_detectors, detector_spacing, number_of_projections):
        super(TrainableFrequencyResponse, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.number_of_projections = number_of_projections
        self.number_detectors = num_detectors
        self.half_detector_numbers = num_detectors // 2 + num_detectors % 2
        self.trainable_numbers = self.half_detector_numbers

        # Trainable parameters
        # This will define the response of the filter at different frequencies
        ramp_value = 1.0 / (detector_spacing * detector_spacing)

        self.trainable_sparse_response = nn.Parameter(
            ramp_value * torch.rand(self.trainable_numbers).to(self.device)
        )

        self.filter_1d = None

    def forward(self, x):
        # Use the trainable parameters to define the filter's response

        frq_response = self.trainable_sparse_response

        if self.number_detectors % 2 == 0:
            self.filter_1d = torch.cat((frq_response.flip(dims=[0]), frq_response))
        else:
            self.filter_1d = torch.cat((frq_response.flip(dims=[0]), frq_response[1:]))

        filter_2d = self.filter_1d.repeat(self.number_of_projections, 1)

        return x * filter_2d


class ParReconstruction2D(nn.Module):
    def __init__(self, geometry):
        super(ParReconstruction2D, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.geometry = geometry

        self.filter = TrainableFrequencyResponse(
            geometry.detector_shape[-1],
            geometry.detector_spacing[-1],
            geometry.number_of_projections,
        ).to(self.device)

        self.AT = ParallelBackProjection2D().to(self.device)

    def forward(self, proj):
        proj = proj.clone().to(self.device)
        x = torch.fft.fft(proj, dim=-1, norm="ortho")
        x = self.filter(x)
        proj = torch.fft.ifft(x, dim=-1, norm="ortho").real.float()

        rco = self.AT.forward(proj.contiguous(), **self.geometry)

        return rco
