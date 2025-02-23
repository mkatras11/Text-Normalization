import gdown
import sys

# Google Drive file URL (zipped file)
URL = " https://drive.google.com/uc?id=12bZiqcEYVmAe1pS94oHNgKmjFsZkhCYI"

def main():
    if len(sys.argv) >= 2:
        output = sys.argv[1]
    else:
        output = "model.tar.gz"
    print(f"Downloading zipped file to {output}")
    gdown.download(URL, output, use_cookies=False, quiet=False)

if __name__ == "__main__":
    main()