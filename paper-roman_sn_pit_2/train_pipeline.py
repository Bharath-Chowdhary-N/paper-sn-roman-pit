#!/usr/bin/env python3
"""
Complete Training Pipeline for Transient Detection

This pipeline orchestrates the following steps:
1. PSF injection into diff files
2. Find non-peak regions for negative samples
3. Find peaks in training diff files for additional negative samples
4. Copy and organize all cutouts into pos_and_neg structure
5. Train the ensemble models

Usage:
    python pipeline.py

Requirements:
- All individual scripts must be in the same directory
- Required directories and files must exist (or will be created)
"""

import subprocess
import sys
import shutil
from pathlib import Path
import time
import os

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"{message}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_step(step_num, total_steps, message):
    """Print a formatted step message"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}/{total_steps}] {message}{Colors.ENDC}")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def run_script(script_name, description):
    """
    Run a Python script and handle errors.
    
    Parameters:
    -----------
    script_name : str
        Name of the script to run
    description : str
        Description of what the script does
    
    Returns:
    --------
    success : bool
        True if script ran successfully, False otherwise
    """
    script_path = Path(script_name)
    
    if not script_path.exists():
        print_error(f"Script not found: {script_name}")
        return False
    
    print(f"\n{Colors.OKBLUE}Running: {script_name}{Colors.ENDC}")
    print(f"Description: {description}\n")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print_success(f"Completed {script_name} in {elapsed_time:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print_error(f"Failed to run {script_name} (after {elapsed_time:.1f} seconds)")
        print_error(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print_error(f"Unexpected error running {script_name} (after {elapsed_time:.1f} seconds)")
        print_error(f"Error: {str(e)}")
        return False
    
def copy_files_recursive(source_dir, dest_dir, file_pattern="*.fits", description=""):
    """
    Recursively copy files from source to destination directory, including subfolders.
    
    Parameters:
    -----------
    source_dir : Path or str
        Source directory
    dest_dir : Path or str
        Destination directory
    file_pattern : str
        Pattern to match files (default: "*.fits")
    description : str
        Description for logging
    
    Returns:
    --------
    count : int
        Number of files copied
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files recursively using rglob instead of glob
    files_to_copy = list(source_dir.rglob(file_pattern))
    
    if not files_to_copy:
        print_warning(f"No files found matching {file_pattern} in {source_dir} (searched recursively)")
        return 0
    
    print(f"\n{Colors.OKBLUE}Copying {len(files_to_copy)} files (recursive):{Colors.ENDC}")
    print(f"  From: {source_dir}")
    print(f"  To:   {dest_dir}")
    if description:
        print(f"  Purpose: {description}")
    
    copied_count = 0
    for file_path in files_to_copy:
        try:
            dest_path = dest_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            copied_count += 1
            
            # Print progress every 100 files
            if copied_count % 100 == 0:
                print(f"  Copied {copied_count}/{len(files_to_copy)} files...")
                
        except Exception as e:
            print_warning(f"Failed to copy {file_path.name}: {str(e)}")
    
    print_success(f"Copied {copied_count} files to {dest_dir}")
    return copied_count

def copy_files(source_dir, dest_dir, file_pattern="*.fits", description=""):
    """
    Copy files from source to destination directory.
    
    Parameters:
    -----------
    source_dir : Path or str
        Source directory
    dest_dir : Path or str
        Destination directory
    file_pattern : str
        Pattern to match files (default: "*.fits")
    description : str
        Description for logging
    
    Returns:
    --------
    count : int
        Number of files copied
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files to copy
    files_to_copy = list(source_dir.glob(file_pattern))
    
    if not files_to_copy:
        print_warning(f"No files found matching {file_pattern} in {source_dir}")
        return 0
    
    print(f"\n{Colors.OKBLUE}Copying {len(files_to_copy)} files:{Colors.ENDC}")
    print(f"  From: {source_dir}")
    print(f"  To:   {dest_dir}")
    if description:
        print(f"  Purpose: {description}")
    
    copied_count = 0
    for file_path in files_to_copy:
        try:
            dest_path = dest_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            copied_count += 1
            
            # Print progress every 100 files
            if copied_count % 100 == 0:
                print(f"  Copied {copied_count}/{len(files_to_copy)} files...")
                
        except Exception as e:
            print_warning(f"Failed to copy {file_path.name}: {str(e)}")
    
    print_success(f"Copied {copied_count} files to {dest_dir}")
    return copied_count

def check_directory_exists(directory, create=False):
    """
    Check if a directory exists, optionally create it.
    
    Parameters:
    -----------
    directory : Path or str
        Directory to check
    create : bool
        If True, create directory if it doesn't exist
    
    Returns:
    --------
    exists : bool
        True if directory exists (or was created), False otherwise
    """
    directory = Path(directory)
    
    if directory.exists():
        return True
    elif create:
        directory.mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {directory}")
        return True
    else:
        print_warning(f"Directory does not exist: {directory}")
        return False

def count_files(directory, pattern="*.fits"):
    """Count files matching pattern in directory"""
    directory = Path(directory)
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))

def main():
    """Main pipeline execution"""
    
    pipeline_start_time = time.time()
    
    print_header("TRANSIENT DETECTION TRAINING PIPELINE")
    print(f"{Colors.BOLD}This pipeline will:{Colors.ENDC}")
    print("  1. Inject PSFs into diff files (create positives)")
    print("  2. Find non-peak regions (create negatives)")
    print("  3. Find peaks in training diff files (additional negatives)")
    print("  4. Organize all cutouts into pos_and_neg structure")
    print("  5. Train ensemble models (6 families x 4 members = 24 total)")
    
    # Configuration
    TOTAL_STEPS = 8
    
    # Script names
    PSF_INJECTION_SCRIPT = "psf_injection_script.py"
    FIND_NON_PEAKS_SCRIPT = "find_non_peaks_64.py"
    FIND_PEAKS_TRAINING_SCRIPT = "find_peaks_above_k_sigma_training.py"
    TRAINING_SCRIPT = "training_script.py"
    
    # Directory paths
    OUTPUT_PSF_ADDED = Path("output_psf_added")
    POSITIVES_SOURCE = OUTPUT_PSF_ADDED / "output_psf_added" / "positives" / "fits"
    NON_PEAKS_SOURCE = Path("non_peaks") / "fits"
    TRAINING_PEAKS_SOURCE = Path("pos_and_neg") / "negatives"  # Training peaks script saves here
    POS_AND_NEG = Path("pos_and_neg")
    POSITIVES_DEST = POS_AND_NEG / "positives"
    NEGATIVES_DEST = POS_AND_NEG / "negatives"
    
    # Check if required input directories exist
    print_step(0, TOTAL_STEPS, "Checking prerequisites")
    
    required_dirs = [
        ("test_diff_files", "Input diff files for processing"),
        ("PSF", "PSF array file location"),
    ]
    
    missing_dirs = []
    for dir_path, description in required_dirs:
        if not Path(dir_path).exists():
            print_error(f"Required directory missing: {dir_path} ({description})")
            missing_dirs.append(dir_path)
        else:
            print_success(f"Found: {dir_path}")
    
    if missing_dirs:
        print_error("Please create the missing directories and add required files")
        return
    
    # Check if PSF file exists
    psf_file = Path("PSF/psf_array.npy")
    if not psf_file.exists():
        print_error(f"PSF file not found: {psf_file}")
        print_error("Please ensure psf_array.npy exists in the PSF directory")
        return
    else:
        print_success(f"Found PSF file: {psf_file}")
    
    # Track success of each step
    steps_status = []
    
    # =========================================================================
    # STEP 1: Run PSF Injection Script
    # =========================================================================
    print_step(1, TOTAL_STEPS, "PSF Injection - Creating Positive Samples")
    
    success = run_script(
        PSF_INJECTION_SCRIPT,
        "Injects PSFs into diff files to create positive training samples"
    )
    steps_status.append(("PSF Injection", success))
    
    if not success:
        print_error("PSF injection failed. Cannot continue pipeline.")
        return
    
    # Verify output
    positive_count = count_files(POSITIVES_SOURCE)
    print(f"  Generated {positive_count} positive samples")
    
    # =========================================================================
    # STEP 2: Run Find Non-Peaks Script
    # =========================================================================
    print_step(2, TOTAL_STEPS, "Finding Non-Peak Regions - Creating Negative Samples")
    
    success = run_script(
        FIND_NON_PEAKS_SCRIPT,
        "Finds uniform non-peak regions for negative training samples"
    )
    steps_status.append(("Find Non-Peaks", success))
    
    if not success:
        print_warning("Non-peak finding failed. Continuing with partial data...")
    
    # Verify output - count recursively since files are in subfolders
    if NON_PEAKS_SOURCE.exists():
        non_peak_files = list(NON_PEAKS_SOURCE.rglob("*.fits"))
        non_peak_count = len(non_peak_files)
        print(f"  Generated {non_peak_count} non-peak samples in subfolders")
    else:
        non_peak_count = 0
        print_warning(f"Non-peaks directory not found: {NON_PEAKS_SOURCE}")
    
    # =========================================================================
    # STEP 3: Run Find Peaks Training Script
    # =========================================================================
    print_step(3, TOTAL_STEPS, "Finding Peaks in Training Data - Additional Negatives")
    
    success = run_script(
        FIND_PEAKS_TRAINING_SCRIPT,
        "Finds real peaks in diff files for additional negative samples"
    )
    steps_status.append(("Find Peaks Training", success))
    
    if not success:
        print_warning("Peak finding in training data failed. Continuing...")
    
    # Check what was generated
    training_peaks_jpg = Path("training_peaks") / "jpg"
    if training_peaks_jpg.exists():
        training_peaks_count = len(list(training_peaks_jpg.rglob("*.jpg")))
        print(f"  Generated {training_peaks_count} peak sample visualizations")
    else:
        training_peaks_count = 0
    
    # Count FITS files that went directly to negatives
    if TRAINING_PEAKS_SOURCE.exists():
        training_peaks_fits = len(list(TRAINING_PEAKS_SOURCE.glob("*.fits")))
        print(f"  Generated {training_peaks_fits} peak FITS samples")
    
    # =========================================================================
    # STEP 4: Organize Positive Samples
    # =========================================================================
    print_step(4, TOTAL_STEPS, "Organizing Positive Samples")
    
    # Create destination directory
    check_directory_exists(POSITIVES_DEST, create=True)
    
    # Copy positives from output_psf_added
    positive_copied = copy_files(
        POSITIVES_SOURCE,
        POSITIVES_DEST,
        "*.fits",
        "PSF-injected positive samples"
    )
    steps_status.append(("Copy Positives", positive_copied > 0))
    
    if positive_copied == 0:
        print_error("No positive samples were copied!")
        print_error("Cannot continue without positive samples")
        return
    
    # =========================================================================
    # STEP 5: Organize Negative Samples
    # =========================================================================
    print_step(5, TOTAL_STEPS, "Organizing Negative Samples")
    
    # Create destination directory
    check_directory_exists(NEGATIVES_DEST, create=True)
    
    # Copy non-peaks recursively from subfolders
    negatives_copied = 0
    
    # From non_peaks/fits/<subfolders>/*.fits
    if NON_PEAKS_SOURCE.exists():
        print(f"\n{Colors.OKBLUE}Searching for non-peak files in subfolders...{Colors.ENDC}")
        count = copy_files_recursive(
            NON_PEAKS_SOURCE,
            NEGATIVES_DEST,
            "*.fits",
            "Non-peak negative samples from all subfolders"
        )
        negatives_copied += count
    else:
        print_warning(f"Non-peaks source directory not found: {NON_PEAKS_SOURCE}")
    
    # Note: The find_peaks_training script already saves directly to negatives folder
    # Check total files in destination
    existing_negatives = count_files(NEGATIVES_DEST)
    print(f"\n{Colors.BOLD}Total negative samples in destination: {existing_negatives}{Colors.ENDC}")
    
    steps_status.append(("Organize Negatives", existing_negatives > 0))
    
    if existing_negatives == 0:
        print_error("No negative samples available!")
        print_error("Cannot continue without negative samples")
        return
    
    # =========================================================================
    # STEP 6: Verify Dataset Balance
    # =========================================================================
    print_step(6, TOTAL_STEPS, "Verifying Dataset")
    
    final_positive_count = count_files(POSITIVES_DEST)
    final_negative_count = count_files(NEGATIVES_DEST)
    
    print(f"\n{Colors.BOLD}Dataset Summary:{Colors.ENDC}")
    print(f"  Positives: {final_positive_count} samples")
    print(f"  Negatives: {final_negative_count} samples")
    print(f"  Total:     {final_positive_count + final_negative_count} samples")
    
    if final_positive_count > 0 and final_negative_count > 0:
        ratio = final_positive_count / final_negative_count
        print(f"  Ratio (pos/neg): {ratio:.2f}")
        
        if 0.5 <= ratio <= 2.0:
            print_success("Dataset is reasonably balanced")
        elif ratio < 0.5:
            print_warning(f"More negatives than positives (ratio: {ratio:.2f})")
            print_warning("This is actually good for training - helps reduce false positives")
        else:
            print_warning(f"More positives than negatives (ratio: {ratio:.2f})")
            print_warning("Consider generating more negative samples")
    
    steps_status.append(("Dataset Verification", True))
    
    # =========================================================================
    # STEP 7: Clean Up Intermediate Files (Optional)
    # =========================================================================
    print_step(7, TOTAL_STEPS, "Cleaning Up Intermediate Files")
    
    print("\nIntermediate directories that can be cleaned up:")
    print(f"  - {OUTPUT_PSF_ADDED} (after copying positives)")
    print(f"  - {Path('non_peaks')} (after copying negatives)")
    print(f"  - {Path('training_peaks')} (JPG visualizations)")
    
    # Ask user if they want to clean up (optional)
    # For automation, we'll skip this and keep files
    print_warning("Keeping intermediate files for verification")
    print("  (You can manually delete them later if needed)")
    print("\nTo clean up manually, run:")
    print(f"  rm -rf {OUTPUT_PSF_ADDED}")
    print(f"  rm -rf non_peaks")
    print(f"  rm -rf training_peaks")
    
    steps_status.append(("Cleanup", True))
    
    # =========================================================================
    # STEP 8: Run Training Script
    # =========================================================================
    print_step(8, TOTAL_STEPS, "Training Ensemble Models")
    
    print(f"\n{Colors.BOLD}Starting stable model training...{Colors.ENDC}")
    print("  Families: DenseNet169, ResNeXt50, RegNetY016,")
    print("            EfficientNetB0, ConvNeXtTiny, DeiTTiny")
    print("  Members : 4 per family  (24 models total)")
    print("  Norm    : ZScale then min-max [0,1]  |  No augmentation  |  DeiT pretrained=True")
    print("\nExpected outputs (one per member per family):")
    for fam in ['DenseNet169', 'ResNeXt50', 'RegNetY016',
                'EfficientNetB0', 'ConvNeXtTiny', 'DeiTTiny']:
        print(f"  - {fam}_Ensemble_Model{{1-4}}_best.pth")
    print("  - *_progress.png (training plots)")
    
    success = run_script(
        TRAINING_SCRIPT,
        "Trains stable ensemble: 6 families x 4 members, ZScale->MinMax norm, no augmentation"
    )
    steps_status.append(("Model Training", success))
    
    if not success:
        print_error("Training failed!")
    else:
        print_success("Training completed successfully!")
        
        model_files = list(Path(".").glob("*_Ensemble_Model*_best.pth"))
        print(f"  Generated {len(model_files)} model checkpoint files")
    
    # =========================================================================
    # Pipeline Summary
    # =========================================================================
    pipeline_elapsed_time = time.time() - pipeline_start_time
    
    print_header("PIPELINE EXECUTION SUMMARY")
    
    print(f"\n{Colors.BOLD}Step Results:{Colors.ENDC}")
    for i, (step_name, step_success) in enumerate(steps_status, 1):
        status_symbol = "✓" if step_success else "✗"
        status_color = Colors.OKGREEN if step_success else Colors.FAIL
        print(f"  {status_color}{status_symbol} Step {i}: {step_name}{Colors.ENDC}")
    
    # Overall status
    all_critical_steps_passed = all([
        steps_status[0][1],  # PSF Injection
        steps_status[3][1],  # Copy Positives
        steps_status[4][1],  # Organize Negatives
    ])
    
    print(f"\n{Colors.BOLD}Overall Status:{Colors.ENDC}")
    if all_critical_steps_passed:
        print_success(f"Pipeline completed successfully in {pipeline_elapsed_time/60:.1f} minutes")
        
        if steps_status[-1][1]:  # Training succeeded
            print(f"\n{Colors.BOLD}✓ All steps completed successfully!{Colors.ENDC}")
            print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
            print("  1. Check training progress plots (*_progress.png)")
            print("  2. Review model checkpoints (*_best.pth)")
            print("  3. Evaluate models on test data")
            print("  4. Use trained models for inference on new data")
        else:
            print_warning("Pipeline completed but training failed")
            print("  Please check the training logs above for errors")
    else:
        print_error(f"Pipeline completed with errors after {pipeline_elapsed_time/60:.1f} minutes")
        print("\nPlease review the errors above and re-run the pipeline")
    
    print(f"\n{Colors.BOLD}Output Locations:{Colors.ENDC}")
    print(f"  Training data:")
    print(f"    - Positives: {POSITIVES_DEST} ({final_positive_count} files)")
    print(f"    - Negatives: {NEGATIVES_DEST} ({final_negative_count} files)")
    print(f"  Model files: *_Ensemble_Model*_best.pth  (6 families x 4 members)")
    print(f"  Progress plots: *_progress.png")
    print(f"  Split data: pwd/split_folders/ (train/val/test)")
    
    print(f"\n{Colors.BOLD}Dataset Statistics:{Colors.ENDC}")
    print(f"  Total samples: {final_positive_count + final_negative_count}")
    print(f"  Positive samples: {final_positive_count}")
    print(f"  Negative samples: {final_negative_count}")
    if final_positive_count > 0 and final_negative_count > 0:
        print(f"  Class ratio: {ratio:.2f}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Unexpected error in pipeline:{Colors.ENDC}")
        print(f"{Colors.FAIL}{str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
