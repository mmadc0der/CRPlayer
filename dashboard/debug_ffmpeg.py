"""
Debug script to test FFmpeg frame extraction from MKV files.
"""

import subprocess
import os
import tempfile
import base64


def test_ffmpeg_extraction(mkv_file_path: str):
    """Test FFmpeg extraction on a real MKV file."""
    print(f"Testing FFmpeg extraction on: {mkv_file_path}")
    print(f"File size: {os.path.getsize(mkv_file_path)} bytes")
    
    # Test 1: Get video info
    info_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', mkv_file_path]
    try:
        result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("FFprobe output:")
            print(result.stdout[:500])
        else:
            print(f"FFprobe failed: {result.stderr}")
    except Exception as e:
        print(f"FFprobe error: {e}")
    
    print("\n" + "="*50)
    
    # Test 2: Extract a few frames
    extract_cmd = [
        'ffmpeg',
        '-y',
        '-i', mkv_file_path,
        '-vf', 'scale=270:-1',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-q:v', '5',
        '-frames:v', '3',  # Only extract 3 frames
        '-an',
        '-loglevel', 'info',
        'pipe:1'
    ]
    
    print(f"FFmpeg command: {' '.join(extract_cmd)}")
    
    try:
        process = subprocess.Popen(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        # Read output
        stdout_data, stderr_data = process.communicate(timeout=15)
        
        print(f"FFmpeg return code: {process.returncode}")
        print(f"Stderr output:\n{stderr_data.decode('utf-8', errors='ignore')}")
        print(f"Stdout data length: {len(stdout_data)} bytes")
        
        if len(stdout_data) > 0:
            # Look for JPEG markers
            jpeg_count = stdout_data.count(b'\xff\xd8')
            print(f"Found {jpeg_count} JPEG start markers")
            
            # Try to extract first frame
            start_idx = stdout_data.find(b'\xff\xd8')
            if start_idx != -1:
                end_idx = stdout_data.find(b'\xff\xd9', start_idx + 2)
                if end_idx != -1:
                    frame_data = stdout_data[start_idx:end_idx + 2]
                    print(f"First frame: {len(frame_data)} bytes")
                    
                    # Save as test file
                    with open('test_frame.jpg', 'wb') as f:
                        f.write(frame_data)
                    print("Saved test frame as test_frame.jpg")
                    
                    # Convert to base64 for web
                    b64_data = base64.b64encode(frame_data).decode('utf-8')
                    print(f"Base64 length: {len(b64_data)} characters")
                    return True
        
        return False
        
    except subprocess.TimeoutExpired:
        print("FFmpeg process timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"FFmpeg extraction error: {e}")
        return False


def find_recent_mkv_file():
    """Find a recent MKV file from temp directory."""
    temp_files = []
    
    # Check common temp locations
    temp_dirs = ['/tmp', tempfile.gettempdir()]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                if file.endswith('.mkv'):
                    full_path = os.path.join(temp_dir, file)
                    if os.path.getsize(full_path) > 1000:  # At least 1KB
                        temp_files.append((full_path, os.path.getmtime(full_path)))
    
    if temp_files:
        # Sort by modification time, newest first
        temp_files.sort(key=lambda x: x[1], reverse=True)
        return temp_files[0][0]
    
    return None


if __name__ == "__main__":
    # Look for recent MKV file
    mkv_file = find_recent_mkv_file()
    
    if mkv_file:
        print(f"Found MKV file: {mkv_file}")
        success = test_ffmpeg_extraction(mkv_file)
        if success:
            print("\n✅ FFmpeg extraction successful!")
        else:
            print("\n❌ FFmpeg extraction failed!")
    else:
        print("No MKV files found in temp directories")
        print("Please run your pipeline first to generate MKV data")
