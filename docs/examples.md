# Examples 

## Cone 3D Reconstruction

```py title="example_cone_3d.py" linenums="1" hl_lines="30-77"
# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt

# TODO: better imports
from pyronn.ct_reconstruction.layers.projection_3d import ConeProjectionFor3D
from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjectionFor3D
from pyronn.ct_reconstruction.geometry.geometry_base import GeometryCone3D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.filters import weights
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.misc.general_utils import fft_and_ifft
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
from pyronn.ct_reconstruction.helpers.filters.filters import shepp_logan_3D


def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    # TODO: Document this function on your own. Could not be documented by the model.
    # TODO: Document this function on your own. Could not be documented by the model.
    volume_size = 256
    volume_shape = [volume_size, volume_size, volume_size]
    volume_spacing = [0.5, 0.5, 0.5]

    # Detector Parameters:
    detector_shape = [400, 600]
    detector_spacing = [1, 1]

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2 * np.pi

    sdd = 1200
    sid = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape=volume_shape,volume_spacing=volume_spacing,
                                detector_shape=detector_shape,detector_spacing=detector_spacing,
                                number_of_projections=number_of_projections,angular_range=angular_range,
                                source_isocenter_distance=sid, source_detector_distance=sdd)
    geometry.set_trajectory(circular_trajectory_3d(geometry.number_of_projections, geometry.angular_range,
                                                   geometry.detector_spacing, geometry.detector_origin,
                                                   geometry.source_isocenter_distance,
                                                   geometry.source_detector_distance,
                                                   True))
    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)

    # ------------------ Call Layers ------------------
    # The following code is the new TF2.0 experimental way to tell
    # Tensorflow only to allocate GPU memory needed rather then allocate every GPU memory available.
    # This is important for the use of the hardware interpolation projector, otherwise there might be not enough memory left
    # to allocate the texture memory on the GPU

    sinogram = ConeProjectionFor3D().forward(phantom, geometry)

    reco_filter = shepp_logan_3D(geometry.detector_shape,geometry.detector_spacing,geometry.number_of_projections)
    x = fft_and_ifft(sinogram, reco_filter)

    reco = ConeBackProjectionFor3D().forward(x, geometry)
    
    plt.figure()
    plt.imshow(np.squeeze(reco)[volume_shape[0]//2], cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig(f'3d_cone_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_cone_3d()
```


## Fan 2D Reconstruction

```py title="example_fan_2d.py" linenums="1" hl_lines="29-72"
# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt

# TODO: better imports
from pyronn.ct_reconstruction.layers.projection_2d import FanProjectionFor2D
from pyronn.ct_reconstruction.layers.backprojection_2d import FanBackProjectionFor2D
from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.filters import weights
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.misc.general_utils import fft_and_ifft
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d


def example_fan_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    # TODO: Document this function on your own. Could not be documented by the model.
    # TODO: Document this function on your own. Could not be documented by the model.
    volume_size = 512
    volume_shape = [volume_size, volume_size]
    volume_spacing = [0.23655,0.23655]

    # Detector Parameters:
    detector_shape = [512]
    detector_spacing = [0.8, 0.8]

    # Trajectory Parameters:
    number_of_projections = 512
    angular_range = 2*np.pi

    sdd = 200
    sid = 100

    # create Geometry class
    geometry = GeometryFan2D(volume_shape, volume_spacing,
                             detector_shape, detector_spacing,
                             number_of_projections, angular_range,
                             sdd, sid)
    geometry.set_trajectory(circular_trajectory_2d(geometry.number_of_projections, geometry.angular_range, True))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    # Add required batch dimension
    phantom = np.expand_dims(phantom,axis=0)
    # ------------------ Call Layers ------------------

    sinogram = FanProjectionFor2D().forward(phantom, geometry)

    #TODO: Add Cosine weighting

    redundancy_weights = weights.parker_weights_2d(geometry)
    sinogram_redun_weighted = sinogram * redundancy_weights
    reco_filter = filters.ram_lak_2D(detector_shape, detector_spacing, number_of_projections)
    x = fft_and_ifft(sinogram, reco_filter)

    reco = FanBackProjectionFor2D().forward(x, geometry)

    plt.figure()
    plt.imshow(np.squeeze(reco), cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig(f'2d_fan_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_fan_2d()

```

## Parallel 2D Reconstruction

```py title="example_parallel_2d.py" linenums="1" hl_lines="26-70"
# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from pyronn.ct_reconstruction.layers.projection_2d import ParallelProjectionFor2D
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjectionFor2D
from pyronn.ct_reconstruction.geometry.geometry_base import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.misc.general_utils import fft_and_ifft
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d
import matplotlib.pyplot as plt

def example_parallel_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    # TODO: Document this function on your own. Could not be documented by the model.
    # TODO: Document this function on your own. Could not be documented by the model.
    volume_size = 256
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = [800]
    detector_spacing = [1]

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2* np.pi

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
    geometry.set_trajectory(circular_trajectory_2d(geometry.number_of_projections, geometry.angular_range, True))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)

    # ------------------ Call Layers ------------------
    sinogram = ParallelProjectionFor2D().forward(phantom, geometry)

    #sinogram = sinogram + np.random.normal(
    #    loc=np.mean(np.abs(sinogram)), scale=np.std(sinogram), size=sinogram.shape) * 0.02

    reco_filter = filters.shepp_logan_2D(geometry.detector_shape, geometry.detector_spacing, geometry.number_of_projections)

    # # one for all
    # x = fft_and_ifft(sinogram, reco_filter)

    # You can also do it step by step
    import torch
    x = torch.fft.fft(torch.tensor(sinogram).cuda(), dim=-1, norm='ortho')
    x = torch.multiply(x, torch.tensor(reco_filter).cuda())
    x = torch.fft.ifft(x, dim=-1, norm='ortho').real

    reco = ParallelBackProjectionFor2D().forward(x, geometry)

    plt.figure()
    plt.imshow(np.squeeze(reco), cmap=plt.get_cmap('gist_gray'))
    plt.axis('off')
    plt.savefig(f'2d_par_reco.png', dpi=150, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    example_parallel_2d()

```
