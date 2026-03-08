#!/usr/bin/env python3
"""
FITS Peak Finder and Cutout Generator - Batch Processing Mode

This script processes all FITS files in a directory to find peaks and create 64x64 cutouts.
Saves cutouts to pos_and_neg/negatives/ and JPG visualizations to training_peaks/jpg/

Requirements:
- astropy
- numpy
- matplotlib
- scipy
- photutils (for peak detection)
- pillow (for exact PNG dimensions)

Install with: pip install astropy numpy matplotlib scipy photutils pillow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.ndimage import maximum_filter, center_of_mass
from scipy.ndimage import label, generate_binary_structure
from scipy.optimize import curve_fit
from PIL import Image
import warnings

# Suppress astropy warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function for fitting peaks.
    
    Parameters:
    -----------
    xy : tuple
        Meshgrid of x, y coordinates
    amplitude : float
        Peak amplitude
    x0, y0 : float
        Center coordinates
    sigma_x, sigma_y : float
        Standard deviations in x and y
    theta : float
        Rotation angle
    offset : float
        Background offset
    
    Returns:
    --------
    z : numpy.ndarray
        2D Gaussian values
    """
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    z = offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return z.ravel()

def estimate_background_noise(data, sigma_clip=3, max_iterations=5):
    """
    Estimate background noise level using iterative sigma clipping.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array of image data
    sigma_clip : float
        Sigma threshold for clipping outliers
    max_iterations : int
        Maximum number of sigma clipping iterations
    
    Returns:
    --------
    background_mean : float
        Mean background level
    background_std : float
        Standard deviation of background noise
    """
    # Start with all finite values
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return 0.0, 0.0
    
    working_data = data[valid_mask].copy()
    
    # Iterative sigma clipping to remove outliers (stars, cosmic rays, etc.)
    for iteration in range(max_iterations):
        mean_val = np.mean(working_data)
        std_val = np.std(working_data)
        
        # Create mask for values within sigma_clip * std of mean
        clip_mask = np.abs(working_data - mean_val) < sigma_clip * std_val
        
        if np.sum(clip_mask) == len(working_data):
            # No more outliers to clip
            break
            
        working_data = working_data[clip_mask]
        
        if len(working_data) == 0:
            # Shouldn't happen, but safety check
            return 0.0, 0.0
    
    background_mean = np.mean(working_data)
    background_std = np.std(working_data)
    
    return background_mean, background_std

def refine_peak_center(data, initial_y, initial_x, search_radius=5, method='centroid'):
    """
    Refine peak center using sub-pixel accuracy methods.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D image data
    initial_y, initial_x : int
        Initial peak coordinates
    search_radius : int
        Radius around initial position to search
    method : str
        Method to use: 'centroid', 'gaussian', or 'maximum'
    
    Returns:
    --------
    refined_y, refined_x : float
        Refined peak coordinates (can be sub-pixel)
    quality : float
        Quality metric (0-1, higher is better)
    """
    # Extract region around peak
    y_min = max(0, initial_y - search_radius)
    y_max = min(data.shape[0], initial_y + search_radius + 1)
    x_min = max(0, initial_x - search_radius)
    x_max = min(data.shape[1], initial_x + search_radius + 1)
    
    region = data[y_min:y_max, x_min:x_max]
    
    if region.size == 0:
        return float(initial_y), float(initial_x), 0.0
    
    # Subtract local background
    bg_level = np.percentile(region, 10)  # Use 10th percentile as local background
    region_bg_sub = region - bg_level
    region_bg_sub = np.maximum(region_bg_sub, 0)  # Ensure non-negative
    
    if method == 'centroid':
        # Center of mass approach
        if np.sum(region_bg_sub) > 0:
            com_y, com_x = center_of_mass(region_bg_sub)
            refined_y = y_min + com_y
            refined_x = x_min + com_x
            
            # Quality based on peak sharpness
            max_val = np.max(region_bg_sub)
            mean_val = np.mean(region_bg_sub)
            quality = (max_val - mean_val) / (max_val + 1e-10) if max_val > 0 else 0.0
        else:
            refined_y, refined_x = float(initial_y), float(initial_x)
            quality = 0.0
            
    elif method == 'gaussian':
        # Gaussian fitting approach
        try:
            # Create coordinate meshgrids
            y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]
            
            # Initial guess for Gaussian parameters
            max_idx = np.unravel_index(np.argmax(region_bg_sub), region_bg_sub.shape)
            amplitude_guess = np.max(region_bg_sub)
            y0_guess, x0_guess = max_idx
            sigma_guess = 2.0
            
            # Flatten data for fitting
            coords = (x_coords, y_coords)
            data_flat = region_bg_sub.ravel()
            
            # Fit Gaussian
            initial_guess = [amplitude_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, 0, 0]
            bounds = ([0, -search_radius, -search_radius, 0.5, 0.5, -np.pi, -np.inf],
                     [np.inf, 2*search_radius, 2*search_radius, search_radius, search_radius, np.pi, np.inf])
            
            popt, pcov = curve_fit(gaussian_2d, coords, data_flat, 
                                 p0=initial_guess, bounds=bounds, maxfev=1000)
            
            # Extract fitted center
            fitted_x0, fitted_y0 = popt[1], popt[2]
            refined_y = y_min + fitted_y0
            refined_x = x_min + fitted_x0
            
            # Quality based on fit quality
            residuals = data_flat - gaussian_2d(coords, *popt)
            r_squared = 1 - (np.sum(residuals**2) / np.sum((data_flat - np.mean(data_flat))**2))
            quality = max(0.0, r_squared)
            
        except Exception:
            # Fall back to centroid method if Gaussian fitting fails
            return refine_peak_center(data, initial_y, initial_x, search_radius, 'centroid')
    
    else:  # method == 'maximum'
        # Simple maximum finding with sub-pixel interpolation
        max_idx = np.unravel_index(np.argmax(region_bg_sub), region_bg_sub.shape)
        refined_y = y_min + max_idx[0]
        refined_x = x_min + max_idx[1]
        
        # Try parabolic interpolation for sub-pixel accuracy
        try:
            if (1 <= max_idx[0] < region_bg_sub.shape[0]-1 and 
                1 <= max_idx[1] < region_bg_sub.shape[1]-1):
                
                # Parabolic fit in y direction
                y_vals = region_bg_sub[max_idx[0]-1:max_idx[0]+2, max_idx[1]]
                y_offset = 0.5 * (y_vals[0] - y_vals[2]) / (y_vals[0] - 2*y_vals[1] + y_vals[2])
                if abs(y_offset) < 1:
                    refined_y += y_offset
                
                # Parabolic fit in x direction
                x_vals = region_bg_sub[max_idx[0], max_idx[1]-1:max_idx[1]+2]
                x_offset = 0.5 * (x_vals[0] - x_vals[2]) / (x_vals[0] - 2*x_vals[1] + x_vals[2])
                if abs(x_offset) < 1:
                    refined_x += x_offset
        except:
            pass  # Use integer coordinates if interpolation fails
        
        # Quality based on local contrast
        max_val = np.max(region_bg_sub)
        local_mean = np.mean(region_bg_sub)
        quality = (max_val - local_mean) / (max_val + 1e-10) if max_val > 0 else 0.0
    
    # Ensure coordinates are within image bounds
    refined_y = max(0, min(data.shape[0] - 1, refined_y))
    refined_x = max(0, min(data.shape[1] - 1, refined_x))
    
    return refined_y, refined_x, quality

def find_peaks_flexible(data, bg_mean, bg_std, min_distance=10, 
                       sigma_threshold=3.0, centering_method='centroid', 
                       centering_radius=5, edge_buffer=16, 
                       max_peaks=None, allow_edge_peaks=False):
    """
    Find peaks with more flexible filtering options to reduce peak loss.
    Uses adaptive distance filtering based on peak brightness and quality.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D normalized image data
    bg_mean : float
        Background mean level
    bg_std : float
        Background standard deviation
    min_distance : int
        Minimum distance between peaks in pixels
    sigma_threshold : float
        Sigma threshold above background for peak detection
    centering_method : str
        Method for peak centering: 'centroid', 'gaussian', or 'maximum'
    centering_radius : int
        Radius for centering refinement
    edge_buffer : int
        Minimum distance from image edges for valid peaks
    max_peaks : int or None
        Maximum number of peaks to return (None = unlimited)
    allow_edge_peaks : bool
        If True, allows peaks closer to edges
    
    Returns:
    --------
    peaks : list of tuples
        List of (y, x) coordinates of detected peaks
    peak_info : dict
        Dictionary with peak detection information
    """
    threshold_abs = bg_mean + sigma_threshold * bg_std
    
    # Adjust edge buffer if allowing edge peaks
    if allow_edge_peaks:
        edge_buffer = max(edge_buffer, 32)  # Minimum for safe cutout creation
    
    peak_info = {
        'bg_mean': bg_mean,
        'bg_std': bg_std,
        'threshold': threshold_abs,
        'sigma_threshold': sigma_threshold,
        'centering_method': centering_method,
        'edge_buffer': edge_buffer,
        'allow_edge_peaks': allow_edge_peaks,
        'filtering_stats': {}
    }
    
    print(f"  Using flexible edge buffer: {edge_buffer} pixels")
    print(f"  Allow edge peaks: {allow_edge_peaks}")
    
    # Use smaller, more sensitive local maxima detection
    # Reduce footprint to catch more peaks
    footprint_size = max(3, min(7, min_distance // 3))  # Much smaller footprint
    neighborhood = np.ones((footprint_size, footprint_size))
    local_maxima = maximum_filter(data, footprint=neighborhood) == data
    
    # Apply threshold
    peaks_mask = local_maxima & (data > threshold_abs)
    initial_coords = np.column_stack(np.where(peaks_mask))
    
    peak_info['filtering_stats']['initial_candidates'] = len(initial_coords)
    
    if len(initial_coords) == 0:
        return [], peak_info
    
    print(f"  Found {len(initial_coords)} initial peak candidates")
    
    # More flexible edge filtering
    image_height, image_width = data.shape
    edge_filtered_coords = []
    
    for coord in initial_coords:
        y, x = coord
        if (edge_buffer <= y < image_height - edge_buffer and 
            edge_buffer <= x < image_width - edge_buffer):
            edge_filtered_coords.append(coord)
    
    peak_info['filtering_stats']['after_edge_filter'] = len(edge_filtered_coords)
    
    if not edge_filtered_coords:
        print(f"  No peaks found away from edges (buffer={edge_buffer} pixels)")
        return [], peak_info
    
    print(f"  {len(edge_filtered_coords)} peaks remain after edge filtering")
    
    # Refine peak centers with quality assessment
    refined_peaks = []
    peak_qualities = []
    peak_intensities = []
    refinement_failures = 0
    
    for coord in edge_filtered_coords:
        y, x = coord
        refined_y, refined_x, quality = refine_peak_center(
            data, y, x, centering_radius, centering_method
        )
        
        # More lenient bounds checking for refined coordinates
        safe_margin = edge_buffer if not allow_edge_peaks else 32
        if (safe_margin <= refined_y < image_height - safe_margin and 
            safe_margin <= refined_x < image_width - safe_margin):
            
            # Interpolated intensity at refined position
            y_floor, x_floor = int(np.floor(refined_y)), int(np.floor(refined_x))
            y_ceil = min(image_height - 1, y_floor + 1)
            x_ceil = min(image_width - 1, x_floor + 1)
            
            dy, dx = refined_y - y_floor, refined_x - x_floor
            intensity = (data[y_floor, x_floor] * (1-dy) * (1-dx) +
                        data[y_ceil, x_floor] * dy * (1-dx) +
                        data[y_floor, x_ceil] * (1-dy) * dx +
                        data[y_ceil, x_ceil] * dy * dx)
            
            refined_peaks.append((refined_y, refined_x))
            peak_qualities.append(quality)
            peak_intensities.append(intensity)
        else:
            refinement_failures += 1
    
    peak_info['filtering_stats']['after_refinement'] = len(refined_peaks)
    peak_info['filtering_stats']['refinement_failures'] = refinement_failures
    
    if not refined_peaks:
        print(f"  No peaks remain after refinement ({refinement_failures} failed)")
        return [], peak_info
    
    # Sort by intensity (bright peaks first) rather than combined score
    # This is more predictable and preserves bright peaks
    sorted_indices = np.argsort(peak_intensities)[::-1]
    refined_peaks = [refined_peaks[i] for i in sorted_indices]
    peak_qualities = [peak_qualities[i] for i in sorted_indices]
    peak_intensities = [peak_intensities[i] for i in sorted_indices]
    
    # More intelligent minimum distance filtering
    # Use adaptive distance based on peak brightness/quality
    filtered_peaks = []
    filtered_qualities = []
    filtered_intensities = []
    distance_rejected = 0
    
    for i, peak in enumerate(refined_peaks):
        if len(filtered_peaks) == 0:
            # Always accept the first (brightest) peak
            filtered_peaks.append(peak)
            filtered_qualities.append(peak_qualities[i])
            filtered_intensities.append(peak_intensities[i])
        else:
            # Check distance to existing peaks
            distances = [np.sqrt((peak[0] - p[0])**2 + (peak[1] - p[1])**2) 
                        for p in filtered_peaks]
            min_dist = min(distances)
            
            # Adaptive minimum distance: closer peaks allowed if they're much brighter
            # or if both peaks have high quality
            current_intensity = peak_intensities[i]
            current_quality = peak_qualities[i]
            
            # Find the nearest existing peak's properties
            nearest_idx = np.argmin(distances)
            nearest_intensity = filtered_intensities[nearest_idx]
            nearest_quality = filtered_qualities[nearest_idx]
            
            # Dynamic distance threshold
            base_distance = min_distance
            
            # Allow closer peaks if current peak is much brighter
            intensity_ratio = current_intensity / nearest_intensity
            if intensity_ratio > 2.0:  # Much brighter
                effective_min_distance = base_distance * 0.7
            elif intensity_ratio > 1.5:  # Moderately brighter
                effective_min_distance = base_distance * 0.85
            else:
                effective_min_distance = base_distance
            
            # Also consider quality - high quality peaks can be closer
            if current_quality > 0.8 and nearest_quality > 0.8:
                effective_min_distance *= 0.8
            
            if min_dist >= effective_min_distance:
                filtered_peaks.append(peak)
                filtered_qualities.append(peak_qualities[i])
                filtered_intensities.append(peak_intensities[i])
            else:
                distance_rejected += 1
        
        # Apply max_peaks limit if specified
        if max_peaks and len(filtered_peaks) >= max_peaks:
            break
    
    peak_info['filtering_stats']['after_distance_filter'] = len(filtered_peaks)
    peak_info['filtering_stats']['distance_rejected'] = distance_rejected
    peak_info['filtering_stats']['final_count'] = len(filtered_peaks)
    
    # Add detailed peak information
    if filtered_peaks:
        peak_info['peak_intensities'] = filtered_intensities
        peak_info['peak_qualities'] = filtered_qualities
        peak_info['peak_sigmas'] = [(intensity - bg_mean) / bg_std 
                                   for intensity in filtered_intensities]
        
        # Enhanced statistics
        avg_quality = np.mean(filtered_qualities)
        min_quality = np.min(filtered_qualities)
        max_sigma = max(peak_info['peak_sigmas'])
        min_sigma = min(peak_info['peak_sigmas'])
        
        peak_info['avg_centering_quality'] = avg_quality
        peak_info['min_centering_quality'] = min_quality
        peak_info['sigma_range'] = (min_sigma, max_sigma)
        
        # Print detailed filtering statistics
        stats = peak_info['filtering_stats']
        print(f"  Peak filtering summary:")
        print(f"    Initial candidates: {stats['initial_candidates']}")
        print(f"    After edge filter: {stats['after_edge_filter']}")
        print(f"    After refinement: {stats['after_refinement']} ({stats['refinement_failures']} failed)")
        print(f"    After distance filter: {stats['after_distance_filter']} ({stats['distance_rejected']} too close)")
        print(f"    Final peaks: {stats['final_count']}")
        print(f"  Peak quality: avg={avg_quality:.3f}, min={min_quality:.3f}")
        print(f"  Sigma range: {min_sigma:.1f} to {max_sigma:.1f}")
    
    return filtered_peaks, peak_info

def create_cutout(data, center_y, center_x, cutout_size=64):
    """
    Create a square cutout from 2D data centered at given coordinates.
    Handles sub-pixel centers by proper interpolation with safe bounds checking.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array of image data
    center_y, center_x : float
        Center coordinates for the cutout (can be sub-pixel)
    cutout_size : int
        Size of the square cutout
    
    Returns:
    --------
    cutout : numpy.ndarray
        Cutout array, padded with zeros if necessary
    """
    half_size = cutout_size // 2
    
    # Calculate bounds
    y_start = center_y - half_size
    x_start = center_x - half_size
    
    # Create coordinate arrays for the cutout
    y_coords = np.arange(cutout_size) + y_start
    x_coords = np.arange(cutout_size) + x_start
    
    # Create output cutout
    cutout = np.zeros((cutout_size, cutout_size), dtype=data.dtype)
    
    # Get data dimensions for safe bounds checking
    max_y, max_x = data.shape[0] - 1, data.shape[1] - 1
    
    # For each pixel in the cutout, interpolate from the original data
    for i in range(cutout_size):
        for j in range(cutout_size):
            y_pos = y_coords[i]
            x_pos = x_coords[j]
            
            if (0 <= y_pos <= max_y - 1 and 0 <= x_pos <= max_x - 1):
                # Bilinear interpolation
                y_floor = int(np.floor(y_pos))
                x_floor = int(np.floor(x_pos))
                y_ceil = min(max_y, y_floor + 1)
                x_ceil = min(max_x, x_floor + 1)
                
                y_floor = max(0, min(max_y, y_floor))
                x_floor = max(0, min(max_x, x_floor))
                
                dy = y_pos - y_floor
                dx = x_pos - x_floor
                
                try:
                    cutout[i, j] = (data[y_floor, x_floor] * (1-dy) * (1-dx) +
                                   data[y_ceil, x_floor] * dy * (1-dx) +
                                   data[y_floor, x_ceil] * (1-dy) * dx +
                                   data[y_ceil, x_ceil] * dy * dx)
                except IndexError:
                    safe_y = max(0, min(max_y, int(round(y_pos))))
                    safe_x = max(0, min(max_x, int(round(x_pos))))
                    cutout[i, j] = data[safe_y, safe_x]
                    
            elif (0 <= y_pos <= max_y and 0 <= x_pos <= max_x):
                safe_y = max(0, min(max_y, int(round(y_pos))))
                safe_x = max(0, min(max_x, int(round(x_pos))))
                cutout[i, j] = data[safe_y, safe_x]
    
    return cutout

def normalize_with_zscale(data):
    """
    Normalize data using astropy's ZScale algorithm.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    
    Returns:
    --------
    normalized_data : numpy.ndarray
        Normalized data between 0 and 1
    """
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data[valid_mask])
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    normalized[~valid_mask] = 0
    
    return normalized

def save_cutout_fits(cutout, filepath, header=None):
    """
    Save cutout as FITS file.
    
    Parameters:
    -----------
    cutout : numpy.ndarray
        Cutout data
    filepath : str or Path
        Output file path
    header : astropy.io.fits.Header or None
        FITS header to include
    """
    hdu = fits.PrimaryHDU(data=cutout, header=header)
    hdu.writeto(filepath, overwrite=True)

def save_cutout_jpg(cutout, filepath):
    """
    Save cutout as JPG file.
    
    Parameters:
    -----------
    cutout : numpy.ndarray
        Normalized cutout data (0-1 range)
    filepath : str or Path
        Output file path
    """
    cutout_8bit = (cutout * 255).astype(np.uint8)
    img = Image.fromarray(cutout_8bit, mode='L')
    img.save(filepath, 'JPEG', quality=95)

def process_single_diff_file(fits_path, output_base_dir, cutout_size=64, 
                            sigma_threshold=3.0, centering_method='centroid',
                            edge_buffer=32, max_peaks=None, allow_edge_peaks=True):
    """
    Process a single diff FITS file to find peaks and create cutouts.
    Saves to pos_and_neg/negatives/ and training_peaks/jpg/
    
    Parameters:
    -----------
    fits_path : str or Path
        Path to input diff FITS file
    output_base_dir : str or Path
        Base output directory
    cutout_size : int
        Size of cutouts to create (default: 64)
    sigma_threshold : float
        Sigma threshold for peak detection (default: 3.0)
    centering_method : str
        Method for peak centering: 'centroid', 'gaussian', or 'maximum'
    edge_buffer : int
        Minimum distance from edges (default: 32)
    max_peaks : int or None
        Maximum number of peaks to return
    allow_edge_peaks : bool
        If True, allows peaks closer to edges
    """
    fits_path = Path(fits_path)
    output_base_dir = Path(output_base_dir)
    
    # Get file name
    file_name = fits_path.stem
    
    # Create output directories
    negatives_dir = output_base_dir / 'pos_and_neg' / 'negatives'
    jpg_dir = output_base_dir / 'training_peaks' / 'jpg'
    negatives_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {fits_path.name}")
        print(f"{'='*60}")
        
        # Read FITS file with robust HDU handling
        with fits.open(fits_path) as hdul:
            print(f"FITS file has {len(hdul)} HDU(s)")
            
            # Find the first HDU with valid data
            data = None
            header = None
            for i, hdu in enumerate(hdul):
                if hdu.data is not None:
                    data = hdu.data
                    header = hdu.header.copy()
                    print(f"  Using HDU {i} ({hdu.__class__.__name__})")
                    break
            
            if data is None:
                print(f"✗ Error: No valid data found in any HDU")
                return 0
        
        # Handle different data shapes
        if data.ndim > 2:
            print(f"  Original shape: {data.shape} (multi-dimensional)")
            data = data[0] if data.ndim == 3 else data.squeeze()
            print(f"  Squeezed to: {data.shape}")
        
        print(f"Image shape: {data.shape}")
        
        # Apply zscale normalization
        normalized_data = normalize_with_zscale(data)
        
        # Compute background noise from the same file
        print("\nComputing background noise statistics...")
        bg_mean, bg_std = estimate_background_noise(normalized_data)
        print(f"  Background: mean={bg_mean:.6f}, std={bg_std:.6f}")
        print(f"  {sigma_threshold}-sigma threshold: {bg_mean + sigma_threshold * bg_std:.6f}")
        
        # Find peaks
        print("\nFinding peaks...")
        peaks, peak_info = find_peaks_flexible(
            normalized_data, 
            bg_mean,
            bg_std,
            min_distance=max(10, cutout_size//4),
            sigma_threshold=sigma_threshold,
            centering_method=centering_method,
            edge_buffer=edge_buffer,
            max_peaks=max_peaks,
            allow_edge_peaks=allow_edge_peaks
        )
        
        if not peaks:
            print(f"\n✗ No peaks found above {sigma_threshold}-sigma threshold")
            return 0
        
        print(f"\n✓ Successfully detected {len(peaks)} peaks")
        
        # Create cutouts for each peak
        print(f"\nCreating {cutout_size}x{cutout_size} cutouts...")
        saved_count = 0
        
        for i, (peak_y, peak_x) in enumerate(peaks):
            cutout = create_cutout(normalized_data, peak_y, peak_x, cutout_size)
            
            peak_sigma = peak_info['peak_sigmas'][i] if 'peak_sigmas' in peak_info else 0
            
            # Create filename
            cutout_name = f"{file_name}_peak_{i:03d}_y{int(round(peak_y))}_x{int(round(peak_x))}_sigma{peak_sigma:.1f}"
            
            # Save as FITS to negatives folder
            fits_path_out = negatives_dir / f"{cutout_name}.fits"
            save_cutout_fits(cutout, fits_path_out, header)
            
            # Save as JPG to training_peaks/jpg folder
            jpg_path = jpg_dir / f"{cutout_name}.jpg"
            save_cutout_jpg(cutout, jpg_path)
            
            saved_count += 1
        
        print(f"\n✓ Saved {saved_count} cutouts")
        print(f"  FITS output: {negatives_dir}")
        print(f"  JPG output: {jpg_dir}")
        
        return saved_count
        
    except Exception as e:
        print(f"\n✗ Error processing {fits_path.name}: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 0

def main():
    """
    Main function - process all FITS files in test_diff_files directory.
    """
    # ==== CONFIGURATION ====
    # Input directory containing all FITS diff files
    input_dir = Path("test_diff_files")
    
    # Output base directory
    output_base_dir = Path("./")
    
    # Detection parameters
    sigma_threshold = 3        # Lower = more sensitive
    edge_buffer = 32             # Pixels from edge to exclude (minimum for 64x64 cutouts)
    allow_edge_peaks = True      # Allow peaks near edges
    max_peaks = 5000            # Limit number of peaks per file
    centering_method = 'centroid'  # 'centroid', 'gaussian', or 'maximum'
    cutout_size = 64            # Size of cutout boxes (DO NOT CHANGE)
    
    # ==== END CONFIGURATION ====
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("\nPlease create the directory and add FITS files to it")
        return
    
    # Find all FITS files
    fits_files = list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fit"))
    
    if not fits_files:
        print(f"Error: No FITS files found in {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"FITS Peak Finder - Batch Processing Mode")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Found {len(fits_files)} FITS files")
    print(f"  Output base directory: {output_base_dir}")
    print(f"  Sigma threshold: {sigma_threshold}")
    print(f"  Edge buffer: {edge_buffer} pixels")
    print(f"  Allow edge peaks: {allow_edge_peaks}")
    print(f"  Max peaks per file: {max_peaks}")
    print(f"  Centering method: {centering_method}")
    print(f"  Cutout size: {cutout_size}x{cutout_size}")
    print(f"\nOutput locations:")
    print(f"  FITS cutouts → pos_and_neg/negatives/")
    print(f"  JPG cutouts → training_peaks/jpg/")
    
    # Process all files
    total_peaks = 0
    successful_files = 0
    
    for idx, fits_file in enumerate(fits_files, 1):
        print(f"\n{'='*60}")
        print(f"File {idx}/{len(fits_files)}")
        print(f"{'='*60}")
        
        num_peaks = process_single_diff_file(
            fits_file, 
            output_base_dir,
            cutout_size=cutout_size,
            sigma_threshold=sigma_threshold,
            edge_buffer=edge_buffer,
            max_peaks=max_peaks,
            allow_edge_peaks=allow_edge_peaks,
            centering_method=centering_method
        )
        
        if num_peaks > 0:
            total_peaks += num_peaks
            successful_files += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Total files processed: {len(fits_files)}")
    print(f"  Successful files: {successful_files}")
    print(f"  Failed files: {len(fits_files) - successful_files}")
    print(f"  Total peaks found: {total_peaks}")
    print(f"  Average peaks per file: {total_peaks / successful_files if successful_files > 0 else 0:.1f}")
    print(f"\nOutputs saved to:")
    print(f"  - {output_base_dir / 'pos_and_neg' / 'negatives'} (FITS)")
    print(f"  - {output_base_dir / 'training_peaks' / 'jpg'} (JPG)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()