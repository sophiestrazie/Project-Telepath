# tests/test_data_loading.py

"""
This test file validates basic data loading functionality for multimodal neuroscience data.
It performs the following checks:

What this tests:
1. File Existence: Verifies required files are present in the data directory
   - Audio file (.wav)
   - Transcript file (.tsv)
   - fMRI data (.nii.gz)
2. File Integrity: Confirms files can be opened and are not empty/corrupt
3. GZIP Validation: Checks that compressed fMRI data is properly formatted

What this does NOT test:
1. Content Accuracy: Does not validate the actual data values or formats
2. Data Completeness: Does not verify all expected data points exist
3. Modality Synchronization: Does not check alignment between audio/transcript/fMRI
4. Advanced NIfTI Features: Does not validate fMRI header information or dimensions
5. Video Data: Video file testing is currently commented out
6. Performance: Does not test loading speed or memory usage

Note: The gzip test currently fails due to suspected file corruption - this needs
investigation to determine if the test is too strict or the file is genuinely invalid.
"""


from pathlib import Path
import gzip
import pytest

def test_files_exist():
    """Test that required files exist in the sample data directory"""
    sample_dir = Path("data")
    required_files = [
        #"sample_video.mp4",
        "audio.wav",
        "transcript.tsv",
        "sub-01_ses-001_task-s01e02a_space-T1w_boldref.nii.gz"
    ]
    
    for file in required_files:
        assert (sample_dir / file).exists(), f"Missing file: {file}"

def test_files_open():
    """Test that files can be opened (no corruption)"""
    sample_dir = Path("data")
    
    # Video
    #with open(sample_dir / "sample_video.mp4", "rb") as f:
        #assert f.read(10), "Video file empty/corrupt"
    
    # Audio
    with open(sample_dir / "audio.wav", "rb") as f:
        assert f.read(10), "Audio file empty/corrupt"
    
    # Transcript
    with open(sample_dir / "transcript.tsv", "r") as f:
        assert f.readline(), "Transcript file empty"

# NOTE: Corrupted .gz file
def test_gz_file_exists_and_valid():
    """Test that a .gz file exists and can be decompressed"""
    gz_path = Path("data/sub-01_ses-001_task-s01e02a_space-T1w_boldref.nii.gz")  # Update path
    
    # 1. Check file exists
    assert gz_path.exists(), f"GZIP file missing: {gz_path}"
    
    # 2. Verify it's a valid gzip file
    try:
        with gzip.open(gz_path, 'rb') as f:
            _ = f.read(1)  # Attempt to read 1 byte
    except (gzip.BadGzipFile, OSError) as e:
        pytest.fail(f"Corrupted/invalid .gz file: {e}")