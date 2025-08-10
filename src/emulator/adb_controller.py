"""
ADB Controller for Android Emulator Management
Handles communication with Android emulator via ADB
"""

import subprocess
import time
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmulatorConfig:
    """Configuration for emulator connection"""
    host: str = "127.0.0.1"
    port: int = 5555
    device_name: str = "emulator-5554"
    package_name: str = "com.supercell.clashroyale"


class ADBController:
    """Controls Android emulator via ADB commands"""
    
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.device_id = f"{config.host}:{config.port}"
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to emulator via ADB"""
        try:
            # Connect to emulator
            result = subprocess.run(
                ["adb", "connect", self.device_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.is_connected = True
                logger.info(f"Connected to emulator: {self.device_id}")
                return True
            else:
                logger.error(f"Failed to connect: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("ADB connection timeout")
            return False
        except Exception as e:
            logger.error(f"ADB connection error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from emulator"""
        try:
            result = subprocess.run(
                ["adb", "disconnect", self.device_id],
                capture_output=True,
                text=True
            )
            self.is_connected = False
            logger.info("Disconnected from emulator")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            return False
    
    def tap(self, x: int, y: int) -> bool:
        """Tap at specific coordinates"""
        if not self.is_connected:
            logger.error("Not connected to emulator")
            return False
            
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "input", "tap", str(x), str(y)],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Tap error: {e}")
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """Swipe from (x1,y1) to (x2,y2)"""
        if not self.is_connected:
            logger.error("Not connected to emulator")
            return False
            
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "input", "swipe", 
                 str(x1), str(y1), str(x2), str(y2), str(duration)],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Swipe error: {e}")
            return False
    
    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """Get emulator screen resolution"""
        if not self.is_connected:
            return None
            
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "wm", "size"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse output like "Physical size: 1280x720"
                output = result.stdout.strip()
                size_str = output.split(": ")[-1]
                width, height = map(int, size_str.split("x"))
                return (width, height)
        except Exception as e:
            logger.error(f"Get screen size error: {e}")
        
        return None
    
    def capture_screen(self, save_path: str = "/tmp/screen.png") -> bool:
        """Capture screenshot from emulator"""
        if not self.is_connected:
            return False
            
        try:
            # Take screenshot on device
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "screencap", "-p", "/sdcard/screen.png"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return False
            
            # Pull screenshot to local machine
            result = subprocess.run(
                ["adb", "-s", self.device_id, "pull", "/sdcard/screen.png", save_path],
                capture_output=True,
                timeout=10
            )
            
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return False
    
    def start_app(self) -> bool:
        """Start Clash Royale app"""
        if not self.is_connected:
            return False
            
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "am", "start", 
                 "-n", f"{self.config.package_name}/.GameApp"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("Clash Royale started")
                time.sleep(3)  # Wait for app to load
                return True
            else:
                logger.error(f"Failed to start app: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Start app error: {e}")
            return False
    
    def stop_app(self) -> bool:
        """Stop Clash Royale app"""
        if not self.is_connected:
            return False
            
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "am", "force-stop", 
                 self.config.package_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Stop app error: {e}")
            return False
    
    def is_app_running(self) -> bool:
        """Check if Clash Royale is currently running"""
        if not self.is_connected:
            return False
            
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "pidof", self.config.package_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and result.stdout.strip()
        except Exception as e:
            logger.error(f"Check app running error: {e}")
            return False
