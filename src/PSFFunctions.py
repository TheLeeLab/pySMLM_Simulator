# -*- coding: utf-8 -*-
"""
This class contains functions that collect analysis routines for RASP
jsb92, 2024/03/04
"""
import os
import numpy as np
import sys
from skimage.filters import gaussian

module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)

import pathos
from pathos.pools import ThreadPool as Pool
cpu_number = int(pathos.helpers.cpu_count()*0.75)

class PSF_Functions():
    def __init__(self):
        self = self
        return
    
    def diffraction_limit(self, wavelength, NA):
        """
        calculates diffraction limit from Abbe criterion
        
        Args:
        - wavelength (float). wavelength of light being imaged
        - NA numerical aperture of microscope
    
        Returns:
        - diffraction_limit (float). d of psf
        """
        return np.divide(wavelength, np.multiply(2., NA))

    def gaussian2d_PSF(self, x, y, sigma_psf, x0, y0, n_photons=4000, bitdepth=np.float32):
        """
        simulates a 2d gaussian psf of width sigma_psf on a grid with coordinates
        x and y, with the origin x0 and y0
        
        Args:
        - x (1d array). x coordinate locations, in same unit as sigma_psf (typically micron)
        - x (1d array). y coordinate locations, in same unit as sigma_psf (typically micron)
        - sigma_psf (float). width of 2d gaussian. 
        - x0 (float or 1d array). origin position of 2d gaussian in x, in same unit as sigma_psf
        - y0 (float or 1d array). origin position of 2d gaussian in y, in same unit as sigma_psf
        - n_photons (int). n_photons per localisation. Given to poisson rng
        - bitdepth (type). bit depth. default uint16        
        
        Returns:
        - PSF_g2d (2D array). 2d probability density function of PSF
        """
        X, Y = np.meshgrid(x, y)
        nfactor = np.multiply(sigma_psf, np.sqrt(2.*np.pi))
        if isinstance(x0, float):
            PSF_g2d = np.asarray(np.multiply(np.divide(np.random.poisson(n_photons), nfactor), np.exp(np.subtract(-np.divide(np.square(X-x0), np.multiply(2., np.square(sigma_psf))), 
                                         np.divide(np.square(Y-y0), np.multiply(2., np.square(sigma_psf)))))), dtype=bitdepth)
        else:
            PSF_g2d = np.zeros_like(X, dtype=bitdepth)
            photon_numbers = np.random.poisson(n_photons, size=len(x0))
            for i in np.arange(len(x0)):
                PSF_g2d = PSF_g2d + np.asarray(np.multiply(np.divide(photon_numbers[i], nfactor), np.exp(np.subtract(-np.divide(np.square(X-x0[i]), np.multiply(2., np.square(sigma_psf))), 
                                             np.divide(np.square(Y-y0[i]), np.multiply(2., np.square(sigma_psf)))))), dtype=bitdepth)
        return PSF_g2d
    
    def gaussian2d_PSF_pixel(self, image_size, sigma_psf, x0, y0, n_photons=4000, bitdepth=np.float32):
        """
        simulates a 2d gaussian psf of width sigma_psf on a grid with coordinates
        x and y, with the origin x0 and y0
        
        Args:
        - image_size (tuple). tuple of how big the image is in pixels
        - sigma_psf (float). width of 2d gaussian in pixels 
        - x0 (float or 1d array). origin position of 2d gaussian in x, in pixels
        - y0 (float or 1d array). origin position of 2d gaussian in y, in pixels
        - n_photons (int). n_photons per localisation. Given to poisson rng
        - bitdepth (type). bit depth. default uint16        
        
        Returns:
        - PSF_g2d (2D array). 2d probability density function of PSF
        """
        PSF_g2d = np.zeros(image_size, dtype=bitdepth)

        if isinstance(x0, float):
            PSF_g2d[x0, y0] = n_photons
            PSF_g2d = np.asarray(gaussian(PSF_g2d, sigma=sigma_psf), dtype=bitdepth)
        else:
            photon_numbers = np.random.poisson(n_photons, size=len(x0))
            PSF_g2d[x0, y0] = photon_numbers
            PSF_g2d = np.asarray(gaussian(PSF_g2d, sigma=sigma_psf), dtype=bitdepth)
        return PSF_g2d
    
    def generate_noisy_image_matrix(self, image_size, lambda_sensor, mu_sensor, sigma_sensor, bitdepth=np.float32):
        """
        simulates a noisy image matrix, using noise formulation of Ober et al,
        Biophys J., 2004
        
        Args:
        - image_size (tuple). tuple of how big the image is in pixels
        - lambda_sensor (float). mean of possion random variable for background noise
        - mu_sensor (float). mean of gaussian for camera read noise
        - sigma_sensor (float). sigma of gaussian for camera read noise
        - bitdepth (type). bit depth. default uint16
    
        Returns:
        - image_matrix (ND array). ND image matrix with noise added to simulate
        detector noise
        """
        image_matrix = np.add(np.asarray(np.random.poisson(lambda_sensor, size=(image_size)), dtype=bitdepth),
                    np.asarray(np.random.normal(loc=mu_sensor, scale=sigma_sensor, size=(image_size)), dtype=bitdepth))
        return image_matrix
    
    def generate_superres_stack(self, image_size, labelled_pixels, n_photons=4000, n_frames=100, 
        labelling_density=0.2, pixel_size=0.11, imaging_wavelength=0.520, NA=1.49,
        lambda_sensor=100, mu_sensor=100, sigma_sensor=10, bitdepth=np.float32):
        """
        simulates a super-resolution image stack based on an specified labelled
        pixels in an image
        
        Args:
        - image_size (tuple). tuple of how big the image is in pixels
        - labelled_pixels (1d array). pixel indices of where labels are
        - n_photons (int). number of photons per localisation
        - n_frames (int). number of frames to make up the super-res trace
        - labelling_density (float). how many of the pixels will be labelled
            across the whole imaging simulation
        - pixel_size (float). Default 0.11 micron, how large pixel sizes are
        - imaging_wavelength (float). Default is 0.52 microns. Imaging wavelength
        - NA (float). Default 1.49 micron, defines how large your PSF will be
        - lambda_sensor (float). mean of poisson rnv
        - mu_semsor (float) mean of gaussian for camera read noise
        - sigma_sensor (float) sigma of gaussian read noise
        - bitdepth (type). bit depth. default float32

        Returns:
        - superres_image_matrix (ND array). ND image matrix with noise and PSFs added.
        - superres_cumsum_matrix (ND array). ND image matrix of cumulative localisations.
        - dl_image_matrix (2D array). Equivalent diffraction-limited image.
        - supreres_image (2D array). Final superres image.
        """
        stack_size = (image_size[0], image_size[1], n_frames)
        sigma_psf = self.diffraction_limit(imaging_wavelength, NA)
        labelling_number = int(labelling_density*len(labelled_pixels)) # get number of labelled pixels
        pixel_subset = np.random.choice(labelled_pixels, labelling_number) # get specifically labelled pixels
        superres_cumsum_matrix = np.zeros(stack_size)
        superres_image_matrix = self.generate_noisy_image_matrix(stack_size,
                                    lambda_sensor, mu_sensor, sigma_sensor, bitdepth)

        def simulate_frames(frame):
            singleframe_subset = np.random.choice(pixel_subset, int(labelling_number/n_frames))
            x0, y0 = np.unravel_index(singleframe_subset, image_size, order='F')
            superres_cumsum_matrix[x0, y0, frame:] = n_photons
            superres_image_matrix[:, :, frame] += self.gaussian2d_PSF_pixel(image_size, 
                                                    sigma_psf/pixel_size, x0, y0, n_photons, bitdepth)
        
        pool = Pool(nodes=cpu_number); pool.restart()
        pool.map(simulate_frames, np.arange(n_frames))
        pool.close(); pool.terminate()
        
        x0d, y0d = np.unravel_index(pixel_subset, image_size, order='F')
        dl_image_matrix = self.generate_noisy_image_matrix(image_size,
                                    lambda_sensor, mu_sensor, sigma_sensor, np.float64)
        dl_image_matrix += self.gaussian2d_PSF_pixel(image_size, sigma_psf/pixel_size, x0d, y0d, n_photons, bitdepth)
        superres_image = np.zeros_like(dl_image_matrix, dtype=bitdepth)
        superres_image[x0d, y0d] = n_photons
        return superres_image_matrix, superres_cumsum_matrix, dl_image_matrix, superres_image