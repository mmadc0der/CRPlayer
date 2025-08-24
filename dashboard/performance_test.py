"""
Performance comparison test between FFmpeg and FastVideoProcessor approaches.
"""

import time
import tempfile
import os
import subprocess
from fast_video_processor import FastVideoProcessor


def create_test_mkv(duration_seconds: int = 5) -> str:
    """Create a test MKV file for performance testing."""
    temp_file = tempfile.mktemp(suffix='.mkv')
    
    # Create test video using FFmpeg
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'testsrc=duration={duration_seconds}:size=1080x1920:rate=30',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        temp_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_file
    except subprocess.CalledProcessError as e:
        print(f"Failed to create test video: {e}")
        return None


def test_ffmpeg_approach(mkv_data: bytes) -> dict:
    """Test the old FFmpeg approach."""
    start_time = time.time()
    frames_extracted = 0
    
    try:
        # Simulate FFmpeg process
        cmd = [
            'ffmpeg',
            '-f', 'matroska',
            '-i', 'pipe:0',
            '-vf', 'scale=400:711',
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-r', '5',
            'pipe:1'
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        # Send data and count frames
        process.stdin.write(mkv_data)
        process.stdin.close()
        
        # Read output frames
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            if b'\x89PNG' in chunk:
                frames_extracted += 1
                
        process.wait()
        
    except Exception as e:
        print(f"FFmpeg test error: {e}")
    
    elapsed = time.time() - start_time
    return {
        'method': 'FFmpeg',
        'elapsed_time': elapsed,
        'frames_extracted': frames_extracted,
        'fps': frames_extracted / elapsed if elapsed > 0 else 0
    }


def test_fast_processor_approach(mkv_data: bytes) -> dict:
    """Test the new FastVideoProcessor approach."""
    start_time = time.time()
    frames_extracted = 0
    
    try:
        processor = FastVideoProcessor(max_fps=10, target_width=400)
        processor.start()
        
        # Feed data to processor
        processor.add_mkv_data(mkv_data)
        
        # Extract frames for 3 seconds
        test_duration = 3.0
        test_start = time.time()
        
        while time.time() - test_start < test_duration:
            frame = processor.get_frame(timeout=0.1)
            if frame:
                frames_extracted += 1
            time.sleep(0.01)
        
        processor.stop()
        
    except Exception as e:
        print(f"FastProcessor test error: {e}")
    
    elapsed = time.time() - start_time
    return {
        'method': 'FastVideoProcessor',
        'elapsed_time': elapsed,
        'frames_extracted': frames_extracted,
        'fps': frames_extracted / elapsed if elapsed > 0 else 0
    }


def run_performance_comparison():
    """Run performance comparison between approaches."""
    print("Creating test MKV file...")
    test_file = create_test_mkv(duration_seconds=3)
    
    if not test_file:
        print("Failed to create test file")
        return
    
    try:
        # Read test data
        with open(test_file, 'rb') as f:
            mkv_data = f.read()
        
        print(f"Test file size: {len(mkv_data)} bytes")
        print("\n" + "="*50)
        
        # Test FFmpeg approach
        print("Testing FFmpeg approach...")
        ffmpeg_results = test_ffmpeg_approach(mkv_data)
        
        print(f"FFmpeg Results:")
        print(f"  Time: {ffmpeg_results['elapsed_time']:.2f}s")
        print(f"  Frames: {ffmpeg_results['frames_extracted']}")
        print(f"  FPS: {ffmpeg_results['fps']:.2f}")
        
        print("\n" + "-"*30)
        
        # Test FastProcessor approach
        print("Testing FastVideoProcessor approach...")
        fast_results = test_fast_processor_approach(mkv_data)
        
        print(f"FastProcessor Results:")
        print(f"  Time: {fast_results['elapsed_time']:.2f}s")
        print(f"  Frames: {fast_results['frames_extracted']}")
        print(f"  FPS: {fast_results['fps']:.2f}")
        
        print("\n" + "="*50)
        
        # Calculate improvement
        if ffmpeg_results['elapsed_time'] > 0:
            speedup = ffmpeg_results['elapsed_time'] / fast_results['elapsed_time']
            print(f"Performance Improvement: {speedup:.1f}x faster")
        
        if ffmpeg_results['fps'] > 0:
            fps_improvement = fast_results['fps'] / ffmpeg_results['fps']
            print(f"FPS Improvement: {fps_improvement:.1f}x higher")
            
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    run_performance_comparison()
