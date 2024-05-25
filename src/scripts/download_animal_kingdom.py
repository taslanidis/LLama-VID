
import os
import subprocess
import gdown

# Change directory to /scratch-shared/scur0405/data/LLaMA-VID-Eval
os.chdir("/scratch-shared/scur0405/data/LLaMA-VID-Eval")

# Create animal-kingdom directory if it doesn't exist
os.makedirs("animal-kingdom", exist_ok=True)

# Change directory to animal-kingdom
os.chdir("animal-kingdom")

# Download the file using gdown
gdown_url = " https://drive.google.com/uc?id=1X4rL5ey7M1_YM4GDa1DvvVdHoUfuHeJp"
output_file = "animals"  # Provide your desired output file name
gdown.download(gdown_url, output_file, quiet=False)