### Helper functions for Data Preprocessing

=== "STL to Numpy Arrays (.h5 container)"

    ```py
    import os
    import numpy as np
    import h5py
    import stltovoxel
    from PIL import Image
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import tempfile
    from pathlib import Path
    from typing import Iterator, Optional
    import logging
    from logging.handlers import RotatingFileHandler
    import itertools
    from datetime import datetime

    def setup_logging():
        logger = logging.getLogger()
        
        # 如果logger已经有处理器，说明已经被设置过，直接返回
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger


    logger = setup_logging()

    def convert_stl_to_png(stl_path: str, resolution: int = 512, pad: int = 0) -> str:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            png_path = tmp.name
        stltovoxel.convert_file(stl_path, png_path, resolution=resolution, pad=pad, parallel=True)
        return png_path

    def load_png_to_numpy(png_path: str, start_num: int, end_num: int) -> np.ndarray:
        target_size = (512, 512)
        images = []
        
        for i in range(start_num, end_num + 1):
            filename = f"{png_path[:-4]}_{i:03d}.png"
            if not os.path.exists(filename):
                logger.warning(f"File {filename} does not exist. Skipping.")
                continue
            with Image.open(filename) as img:
                resized_img = img.resize(target_size, Image.LANCZOS)
                images.append(np.array(resized_img))
        
        images_array = np.array(images)
        
        if len(images) != 512:
            padded_array = np.zeros((512, 512, 512), dtype=np.uint8)
            padded_array[:len(images)] = images_array[:512]
            return padded_array
        
        return images_array.astype(np.uint8)

    def process_single_stl(stl_file: str) -> np.ndarray:
        png_path = convert_stl_to_png(stl_file)
        array = load_png_to_numpy(png_path, 0, 511)
        
        # Delete PNG files
        for png_file in Path(png_path[:-4]).glob('*.png'):
            png_file.unlink()
        
        return array

    def stl_file_generator(input_dir: str, max_files: Optional[int] = None) -> Iterator[Path]:
        count = 0
        for stl_file in Path(input_dir).rglob('*.stl'):
            yield stl_file
            count += 1
            if max_files is not None and count >= max_files:
                break

    def process_stl_files(input_dir: str, output_dir: str, batch_size: int = 1000, max_files: Optional[int] = None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        num_workers = max(1, multiprocessing.cpu_count() - 1)
        stl_files = stl_file_generator(input_dir, max_files)

        batch_num = 0
        file_count = 0

        while True:
            h5_filename = Path(output_dir) / f'batch_{batch_num:03d}.h5'
            
            with h5py.File(h5_filename, 'w') as h5f:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    batch_files = list(itertools.islice(stl_files, batch_size))
                    if not batch_files:
                        break  # No more files to process

                    futures = {executor.submit(process_single_stl, str(stl_file)): i 
                            for i, stl_file in enumerate(batch_files)}
                    
                    for future in as_completed(futures):
                        i = futures[future]
                        try:
                            array = future.result()
                            h5f.create_dataset(f'array_{i:03d}', data=array, compression='gzip')
                            file_count += 1
                            logger.info(f"Processed file {file_count}")
                        except Exception as e:
                            logger.error(f"Error processing file {file_count + 1}: {e}")

            logger.info(f"Batch {batch_num} saved to {h5_filename}")
            batch_num += 1

        logger.info(f"Total processed files: {file_count}")

    if __name__ == "__main__":
        input_dir = r"D:\datasets\stl2\abc_0000_stl2_v00"
        output_dir = r"D:\datasets\stl2\output_h5"
        batch_size = 10
        max_files = 100  # Set this to the number of files you want to process, or None to process all files
        process_stl_files(input_dir, output_dir, batch_size=batch_size, max_files=max_files)
    ```

=== "Numpy (.h5 container) to Sinograms"

    ```py
    import os
    import h5py
    import numpy as np
    import torch
    from pyronn.ct_reconstruction.geometry.geometry import Geometry
    from pyronn.ct_reconstruction.helpers.filters import weights, filters
    from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
        circular_trajectory_3d,
    )
    from torch.nn import functional as F
    from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D
    from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjection3D
    from scipy.ndimage import convolve, gaussian_filter, rotate
    from scipy.signal import convolve2d
    from scipy.signal import fftconvolve

    from update.beam_harden import simulate_polychromatic_spectrum, apply_beam_hardening4D


    class SinogramNoiseSimulator:
        def __init__(self, sinogram):
            """
            Initialize the SinogramNoiseSimulator with a sinogram.

            :param sinogram: Original sinogram data of shape (1, num_projections, height, width)
            """
            self.sinogram = sinogram

        def add_detector_jitter(self, max_jitter):
            """
            Simulate detector jitter in the sinogram.

            :param max_jitter: Maximum amount of jitter (in pixels)
            """
            num_projections = self.sinogram.shape[1]
            jittered_sinogram = np.zeros_like(self.sinogram)

            for i in range(num_projections):
                jitter = np.random.randint(-max_jitter, max_jitter)
                jittered_projection = np.roll(self.sinogram[0, i, :, :], jitter, axis=1)
                jittered_projection = np.roll(jittered_projection, jitter, axis=0)

                jittered_sinogram[0, i, :, :] = jittered_projection

            self.sinogram = jittered_sinogram

        def add_gantry_motion_blur(self, blur_length, angle_range):
            """
            Apply gantry motion blur to the sinogram.

            :param blur_length: Length of the blur kernel
            :param angle_range: Range of angles (in degrees) for the gantry rotation during acquisition
            """
            num_projections, height, width = self.sinogram.shape[1:]
            
            # Create a curved blur kernel
            kernel_size = blur_length * 2 + 1
            y, x = np.ogrid[-blur_length:blur_length+1, -blur_length:blur_length+1]
            mask = x**2 + y**2 <= blur_length**2
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[mask] = 1
            kernel /= kernel.sum()  # Normalize the kernel

            blurred_sinogram = np.zeros_like(self.sinogram)

            for i in range(num_projections):
                projection = self.sinogram[0, i, :, :]
                
                # Calculate the rotation angle for this projection
                rotation_angle = (i / num_projections) * angle_range
                
                # Rotate the kernel
                rotated_kernel = rotate(kernel, rotation_angle, reshape=False)
                
                # Apply the rotated kernel
                blurred_projection = convolve2d(projection, rotated_kernel, mode='same', boundary='wrap')
                
                blurred_sinogram[0, i, :, :] = blurred_projection

            self.sinogram = blurred_sinogram

        def add_high_frequency_noise(self, noise_level, high_freq_strength):
            """
            Add high-frequency noise to the sinogram.

            :param noise_level: Standard deviation of the Gaussian noise
            :param high_freq_strength: Strength of the high-frequency component
            """
            num_projections = self.sinogram.shape[1]
            noisy_sinogram = np.zeros_like(self.sinogram)

            for i in range(num_projections):
                noise = np.random.normal(0, noise_level, size=self.sinogram.shape[2:])
                low_freq_noise = gaussian_filter(noise, sigma=high_freq_strength)
                high_freq_noise = noise - low_freq_noise

                noisy_sinogram[0, i, :, :] = self.sinogram[0, i, :, :] + high_freq_noise

            self.sinogram = noisy_sinogram

        
        def add_poisson_noise(self, scale_factor):
            """
            Add Poisson noise to the sinogram.

            :param scale_factor: Scale factor to control the intensity of the noise
            """
            noisy_sinogram = np.zeros_like(self.sinogram)
            for i in range(self.sinogram.shape[1]):
                noisy_sinogram[0, i, :, :] = np.random.poisson(self.sinogram[0, i, :, :] * scale_factor) / scale_factor
            self.sinogram = noisy_sinogram
        
        def add_aliasing_artifacts(self, undersample_factor):
            num_projections = self.sinogram.shape[1]
            self.sinogram = self.sinogram[:, ::undersample_factor, :, :]

        def add_metal_artifacts(self, metal_positions, metal_intensity):
            """
            Add metal artifacts to the sinogram.

            :param sinogram: The original sinogram data
            :param metal_positions: List of tuples indicating the positions of metal objects in the sinogram
            :param metal_intensity: The intensity of the metal artifact
            :return: The sinogram with metal artifacts
            """
            for pos in metal_positions:
                self.sinogram[:, :, pos[0], pos[1]] += metal_intensity

        def add_gaussian_noise(self, mean, std):
            """
            Add Gaussian noise to the sinogram.

            :param mean: Mean of the Gaussian noise
            :param std: Standard deviation of the Gaussian noise
            """
            gaussian_noise = np.random.normal(mean, std, self.sinogram.shape)
            self.sinogram += gaussian_noise
            self.sinogram = self.sinogram.float()

        def add_ring_artifacts(self, num_rings):
            """
            Add ring artifacts to the sinogram.

            :param num_rings: Number of rings to add
            """

            num_projections = self.sinogram.shape[1]
            height = self.sinogram.shape[2]
            width = self.sinogram.shape[3]

            for _ in range(num_rings):
                # Randomly select detector index
                broken_detector_index = np.random.randint(0, width)

                # Randomly select the range of heights for the artifact
                start_height = np.random.randint(0, height - 1)
                end_height = np.random.randint(start_height + 1, height)

                # Add the artifact to the specified region of the sinogram
                self.sinogram[:, :, start_height:end_height, broken_detector_index] = 0

        def add_scatter_noise(self, scatter_fraction, energy_MeV=0.14):
            """
            Add realistic scatter noise to the CT sinogram.

            :param scatter_fraction: Fraction of total intensity that becomes scatter (0-1)
            :param energy_MeV: X-ray energy in MeV (affects scatter distribution)
            """
            # Calculate total intensity
            total_intensity = self.sinogram.sum()

            # Calculate scatter intensity
            scatter_intensity = scatter_fraction * total_intensity

            # Generate object-dependent scatter distribution
            scatter_base = self._generate_object_dependent_scatter()

            # Apply energy-dependent scatter kernel
            scatter_noise = self._apply_scatter_kernel(scatter_base, energy_MeV)

            # Normalize scatter noise to desired intensity
            scatter_noise = scatter_noise / scatter_noise.sum() * scatter_intensity

            # Add scatter to original sinogram
            self.sinogram += scatter_noise

        def _generate_object_dependent_scatter(self):
            """Generate initial scatter distribution based on object density."""
            # Use softmax to create a probability distribution based on object density
            scatter_prob = torch.softmax(self.sinogram.flatten(), dim=0).reshape(self.sinogram.shape)
            
            # Generate initial scatter based on this probability
            scatter_base = torch.poisson(scatter_prob * 1000)  # Multiplier for more pronounced effect
            return scatter_base

        def _apply_scatter_kernel(self, scatter_base, energy_MeV):
            """Apply an energy-dependent scatter kernel."""
            # Convert to NumPy for convolution
            scatter_np = scatter_base.numpy()

            # Create an anisotropic kernel favoring forward scatter
            kernel = self._create_anisotropic_kernel(scatter_np.shape, energy_MeV)

            # Apply the kernel using FFT convolution
            scatter_noise = fftconvolve(scatter_np, kernel, mode='same')

            return torch.tensor(scatter_noise, dtype=torch.float32)

        def _create_anisotropic_kernel(self, shape, energy_MeV):
            """Create an anisotropic scatter kernel based on energy and sinogram shape."""
            # Ensure we're working with the last two dimensions for 2D convolution
            if len(shape) > 2:
                rows, cols = shape[-2], shape[-1]
            else:
                rows, cols = shape

            y, x = np.ogrid[-rows//2:rows//2, -cols//2:cols//2]
            
            # Adjust these parameters based on your specific CT geometry and energy range
            forward_sigma = cols / 8 * (energy_MeV / 0.1)**0.5  # Increases with energy
            lateral_sigma = rows / 16

            # Create an elliptical Gaussian kernel
            kernel = np.exp(-(x**2 / (2 * forward_sigma**2) + y**2 / (2 * lateral_sigma**2)))

            # Normalize the kernel
            kernel /= kernel.sum()
            
            # If the sinogram has more than 2 dimensions, expand the kernel
            if len(shape) > 2:
                for _ in range(len(shape) - 2):
                    kernel = np.expand_dims(kernel, axis=0)
            
            return kernel
        def add_beam_hardening(self, material):
            print("adding beam hardening")
            energy_bins = np.linspace(1, 120, 100)

            # Simulate polychromatic spectrum
            spectrum = simulate_polychromatic_spectrum(energy_bins)

            # Apply beam hardening
            hardened_projections = apply_beam_hardening4D(self.sinogram[0], energy_bins, spectrum, material)

            hardened_projections = torch.tensor(
                np.expand_dims(hardened_projections, axis=0).copy(), dtype=torch.float32
            )
            self.sinogram = hardened_projections

    def apply_noise(noise_simulator, noise_type):
        if noise_type == "add_gantry_motion_blur":
            noise_simulator.add_gantry_motion_blur(blur_length=5, angle_range=0.5)
        elif noise_type == "add_poisson_noise":
            noise_simulator.add_poisson_noise(scale_factor=10)
        elif noise_type == "add_ring_artifacts":
            noise_simulator.add_ring_artifacts(num_rings=5)
        elif noise_type == "add_scatter_noise":
            noise_simulator.add_scatter_noise(0.5)
        elif noise_type == "add_beam_hardening":
            noise_simulator.add_beam_hardening(material="bone")
        elif noise_type == "add_detector_jitter":
            noise_simulator.add_detector_jitter(max_jitter=5)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def save_results(results, input_file, output_path, noise_type, compression="gzip"):
        base_name = os.path.basename(input_file)
        name_without_extension = os.path.splitext(base_name)[0]
        
        output_file = os.path.join(output_path, f"{name_without_extension}_{noise_type}_processed.h5")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with h5py.File(output_file, "w") as h5f:
            for i, (sinogram, noisy_sinogram, volume, noisy_volume) in enumerate(results):
                h5f.create_dataset(f"sinogram_{i:03d}", data=sinogram.numpy(), compression=compression)
                h5f.create_dataset(f"noisy_sinogram_{i:03d}", data=noisy_sinogram, compression=compression)
                h5f.create_dataset(f"volume_{i:03d}", data=volume.numpy(), compression=compression)
                h5f.create_dataset(f"noisy_volume_{i:03d}", data=noisy_volume.numpy(), compression=compression)
        print(f"Processed data for {noise_type} saved to {output_file}")

    def process_dataset(input_file, output_path, detector_shape, detector_spacing, num_projections, angular_range, sdd, sid, noise_types):
        with h5py.File(input_file, 'r') as h5f:
            num_volumes = sum(1 for key in h5f.keys() if key.startswith('array_'))
            print(f"Number of volumes in the dataset: {num_volumes}")

            if num_volumes == 0:
                raise ValueError("No datasets found with 'array_' prefix in the input file.")

            # Read the first volume to get the shape
            first_volume = h5f['array_000'][()]
            volume_shape = first_volume.shape
            print(f"Volume shape: {volume_shape}")

            # Assuming cubic voxels, calculate volume spacing
            volume_spacing = (1, 1, 1)  # You might want to adjust this if you have specific spacing information

            # Initialize geometry
            geometry = Geometry()
            geometry.init_from_parameters(
                volume_shape=volume_shape,
                volume_spacing=volume_spacing,
                detector_shape=detector_shape,
                detector_spacing=detector_spacing,
                number_of_projections=num_projections,
                angular_range=angular_range,
                trajectory=circular_trajectory_3d,
                source_isocenter_distance=sid,
                source_detector_distance=sdd,
            )

            reco_filter = torch.tensor(
                filters.ram_lak_3D(
                    geometry.detector_shape,
                    geometry.detector_spacing,
                    geometry.number_of_projections,
                ),
                dtype=torch.float32,
            ).cuda()

            for noise_type in noise_types:
                results = []
                for index in range(num_volumes):
                    volume = h5f[f'array_{index:03d}'][()]
                    volume_tensor = torch.tensor(
                        np.expand_dims(volume, axis=0), dtype=torch.float32
                    ).cuda()
                    sinogram = ConeProjection3D(hardware_interp=True).forward(
                        volume_tensor, **geometry
                    )
                    sinogram_tensor = sinogram.detach().cpu()

                    noise_simulator = SinogramNoiseSimulator(sinogram_tensor)
                    
                    # Apply the current noise type
                    apply_noise(noise_simulator, noise_type)

                    noisy_sinogram = noise_simulator.sinogram
                    noisy_sinogram = noisy_sinogram * weights.cosine_weights_3d(geometry)

                    x = torch.fft.fft(torch.Tensor(noisy_sinogram).cuda(), dim=-1)
                    x = torch.multiply(x, reco_filter)
                    x = torch.fft.ifft(x, dim=-1).real

                    noisy_reco = ConeBackProjection3D(hardware_interp=True).forward(
                        x.contiguous(), **geometry
                    )

                    noisy_reco = F.relu(noisy_reco)
                    noisy_reco = noisy_reco.cpu()

                    results.append((sinogram_tensor, noisy_sinogram, volume_tensor.cpu(), noisy_reco))
                    print(f"Processed volume {index + 1}/{num_volumes} with {noise_type}")

                save_results(results, input_file, output_path, noise_type)

    def main():
        # Define parameters
        detector_row = 800
        detector_col = 800
        detector_spacer = 1
        num_projections = 400
        angular_range = 2 * np.pi
        sdd = 3000  # Source-Detector distance
        sid = 2400  # Source-Isocenter distance
        output_path = r"D:\datasets\stl2\output_h5\noisy_data"
        input_file = r"D:\datasets\stl2\output_h5\batch_003.h5"
        # input_file = r"C:\Users\sun\OneDrive - Fraunhofer\PhD\known_operator\2D-2-3D\pancreas_ct_data.h5"

        os.makedirs(output_path, exist_ok=True)

        detector_shape = (detector_row, detector_col)
        detector_spacing = (detector_spacer, detector_spacer)

        noise_types = [
            "add_gantry_motion_blur",
            # "add_poisson_noise",
            # "add_ring_artifacts",
            # # "add_scatter_noise",
            # "add_beam_hardening",
            "add_detector_jitter"
        ]

        process_dataset(input_file, output_path, detector_shape, detector_spacing, num_projections, angular_range, sdd, sid, noise_types)

    if __name__ == "__main__":
        main()
    ```