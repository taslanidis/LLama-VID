import gdown
url = 'https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev'
gdown.download_folder(url, quiet=True, remaining_ok=True, use_cookies=False)