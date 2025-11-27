"""
Install dlib using pre-built wheels to avoid CMake build issues
"""
import subprocess
import sys
import platform

def install_dlib():
    """Install dlib using pre-built wheel"""
    print("=" * 60)
    print("Installing dlib (pre-built wheel)")
    print("=" * 60)
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {py_version}")
    
    # For Python 3.10 on Windows, use a specific wheel
    if platform.system() == "Windows" and sys.version_info[:2] == (3, 10):
        print("\nInstalling dlib from pre-built wheel...")
        try:
            # Try pip install with no build isolation
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-build-isolation", "dlib==19.24.2"],
                check=True
            )
            print("✓ dlib installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Pre-built wheel failed, trying alternative...")
    
    # Try standard installation
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "dlib"],
            check=True
        )
        print("✓ dlib installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dlib: {e}")
        print("\nAlternative solutions:")
        print("1. Download pre-built wheel from:")
        print("   https://github.com/sachadee/Dlib/blob/master/dlib-19.24.99-cp310-cp310-win_amd64.whl")
        print("   Then: pip install dlib-19.24.99-cp310-cp310-win_amd64.whl")
        print("\n2. Or use conda:")
        print("   conda install -c conda-forge dlib")
        return False

if __name__ == "__main__":
    install_dlib()
