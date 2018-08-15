import requests
import re
import os


def get_img_url(url):
    bing_text = requests.get(url).text
    return url+re.search('url: \\"(.*?\\.jpg)\\"', bing_text).group(1)


def save_img(image_url):
    r = requests.get(image_url, stream=True)
    with open("C:/Users/AlphaGo/Pictures/windows_background/background.jpg", "wb") as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


def empty_old_dir(path):
    for f in os.listdir(path):
        os.remove(path+'/'+f)

if __name__ == '__main__':
    img_url = get_img_url("http://cn.bing.com")
    empty_old_dir("C:/Users/AlphaGo/Pictures/windows_background")
    save_img(img_url)