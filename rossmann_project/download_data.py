import os
import zipfile
import subprocess

RAW = os.path.join(os.path.dirname(__file__), "data", "raw")


def main():
    os.makedirs(RAW, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "rossmann-store-sales",
         "-p", RAW], check=True)
    zpath = os.path.join(RAW, "rossmann-store-sales.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW)
    print("Dataset listo en", RAW)


if __name__ == "__main__":
    main()
