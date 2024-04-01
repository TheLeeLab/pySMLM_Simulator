# -*- coding: utf-8 -*-
"""
This class contains functions that simulate tiffs and gifs
jsb92, 2024/03/04
"""
import os
import numpy as np
import sys

module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)
import IOFunctions
IO = IOFunctions.IO_Functions()
import PSFFunctions
PSF_F = PSFFunctions.PSF_Functions()


class Simulation_Routines():
    def __init__(self):
        self = self
        return
    
    def generate_tiffs_and_gifs(self, example_image_path, n_photons=1000, n_frames=1000, 
        labelling_density=2, pixel_size=0.11, imaging_wavelength=0.520, NA=1.49,
        lambda_sensor=100, mu_sensor=100, sigma_sensor=10, invert=False):
        """
        generates tiffs and gifs of "super-resolved" stacks and images from an
        intial input image (ideally a black-and-white image where values above 0
                            are what you "label" in the super-res "xpt")
        
        Args:
        - example_image_path (abspath). Path that points to image to super-resolve
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
        - Invert (boolean). If True, goes from white to black when plot
        """
        from datetime import datetime
        d = datetime.today().strftime('%Y%m%d')

        image_path_orig = os.path.split(example_image_path)[0]
        image_paths = os.path.split(image_path_orig)
        image_path = os.path.join(image_paths[0], d+'_'+image_paths[1])
        raw_filename = os.path.split(example_image_path)[-1].split('.')[0]
        simulation_directory = image_path+'_simulation'
        IO.make_directory(simulation_directory)
        simulation_p_directory = image_path+'_simulationparameters'
        
        to_save = {'n_photons_per_loc': n_photons, 'n_frames': n_frames, 'labelling_density': labelling_density,
                   'pixel_size': pixel_size, 'imaging_wavelength':
                       imaging_wavelength, 'NA': NA, 'lambda_sensor': lambda_sensor,
                       'mu_sensor': mu_sensor, 
                       'sigma_sensor': sigma_sensor}
            
        IO.save_simulation_params(simulation_p_directory, 
                to_save)
        
        image = IO.read_png(example_image_path)
        image[image < 0.5] = 0
        x0, y0 = np.nonzero(image)
        indices = np.ravel_multi_index([x0, y0], image.shape, order='F')
        
        superres_image_stack, superres_cumsum_stack, dl_image, superres_image = PSF_F.generate_superres_stack(image.shape, indices, 
                                    n_photons=n_photons, n_frames=n_frames, 
                                    labelling_density=labelling_density, pixel_size=pixel_size, 
                                    imaging_wavelength=imaging_wavelength, NA=NA,
                                    lambda_sensor=lambda_sensor, mu_sensor=mu_sensor, 
                                    sigma_sensor=sigma_sensor)
        
        if invert==True: # if true, invert all of these values
            max_value = np.iinfo(np.uint16).max
            superres_image_stack = ((superres_image_stack - np.min(superres_image_stack)) / (np.max(superres_image_stack) - np.min(superres_image_stack)) * max_value)
            superres_image_stack = -superres_image_stack + max_value
            superres_cumsum_stack = ((superres_cumsum_stack - np.min(superres_cumsum_stack)) / (np.max(superres_cumsum_stack) - np.min(superres_cumsum_stack)) * max_value)
            superres_cumsum_stack = -superres_cumsum_stack + max_value
            dl_image = ((dl_image - np.min(dl_image)) / (np.max(dl_image) - np.min(dl_image)) * max_value)
            dl_image = -dl_image + max_value
            superres_image = ((superres_image - np.min(superres_image)) / (np.max(superres_image) - np.min(superres_image)) * max_value)
            superres_image = -superres_image + max_value
            
        stack_superres_cumsum = np.vstack([superres_image_stack, superres_cumsum_stack])
        del superres_cumsum_stack
        
        gif_filepath = os.path.join(simulation_directory, raw_filename+'_superres.gif')
        tiff_filepath = os.path.join(simulation_directory, raw_filename+'_superres.tiff')
        tiff_sr_filepath = os.path.join(simulation_directory, raw_filename+'_superres_singleimage.tiff')
        tiff_dl_filepath = os.path.join(simulation_directory, raw_filename+'_diffractionlimited.tiff')
        
        IO.write_tiff(superres_image_stack, tiff_filepath)
        del superres_image_stack
        IO.write_tiff(superres_image, tiff_sr_filepath)
        del superres_image
        IO.write_tiff(dl_image, tiff_dl_filepath)
        del dl_image
        IO.write_gif(stack_superres_cumsum, gif_filepath, fps=500, loop=0)
        del stack_superres_cumsum
        return
