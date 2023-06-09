import argparse
import glob
import os
import time
import urllib

import librosa
import pandas as pd
import requests
import soundfile as sf
from bs4 import BeautifulSoup
from omegaconf import OmegaConf
from pydub import AudioSegment
from requests.exceptions import Timeout


class Scraper:
    def __init__(self, config):
        self.base_url = "https://soundeffect-lab.info/"
        self.df = pd.DataFrame([], columns=["filename", "title", "category", "url"])
        self.idx = 0
        self.config = OmegaConf.load(config)
        self.setup()
        os.makedirs(self.config.path_data, exist_ok=True)
        self.history = []

    def run(self):
        self.all_get()
        self.preprocess()

    def setup(self):
        try:
            html = requests.get(self.base_url, timeout=5)
        except Timeout:
            raise ValueError("Time Out")
        soup = BeautifulSoup(html.content, "html.parser")
        tags = soup.select("a")
        self.urls = []
        self.categories = []
        for tag in tags:
            category = tag.text
            url = tag.get("href")
            if "/sound/" in url:
                self.urls.append(url)
                self.categories.append(category)

    def all_get(self):
        for i in range(len(self.urls)):
            now_url = self.base_url + self.urls[i][1:]
            self.download(now_url, self.categories[i])
        self.df.to_csv(self.config.path_csv)

    def download(self, now_url, category):
        try:
            html = requests.get(now_url, timeout=5)
            soup = BeautifulSoup(html.content, "html.parser")
            body = soup.find(id="wrap").find("main")
            tags = body.find(id="playarea").select("a")
            count = 0
            for tag in tags:
                name = tag.get("download")
                url = tag.get("href")
                filename = os.path.join(self.config.path_data, name)
                if os.path.exists(filename):
                    continue
                try:
                    urllib.request.urlretrieve(now_url + url, filename)
                    title = name.replace(".mp3", "")
                    self.df.loc[self.idx] = {
                        "filename": filename,
                        "title": title,
                        "category": category,
                        "url": f"https://soundeffect-lab.info/sound/search.php?s={title}",
                    }
                    self.idx += 1
                    time.sleep(2)
                    count += 1
                except Exception:
                    continue
            self.history.append(category)
            print(now_url, category, len(tags), count)
            paths = glob.glob(os.path.join(self.config.path_data, "*"))
            assert len(paths) == len(self.df)

            others = body.find(id="pagemenu-top").select("a")
            other_urls, other_categories = [], []
            for other in others:
                other_url = other.get("href")
                other_name = other.find("img").get("alt")
                if other_name in self.history:
                    continue
                other_urls.append(other_url)
                other_categories.append(other_name)
            for i in range(len(other_urls)):
                self.download(self.base_url + other_urls[i][1:], other_categories[i])
        except Timeout:
            print(f"Time Out: {now_url}")

    def preprocess(self):
        for i in range(len(self.df)):
            song = AudioSegment.from_mp3(
                os.path.join(self.config.path_data, self.df.iloc[i]["filename"])
            )
            song.export(
                os.path.join(
                    self.config.path_data,
                    self.df.iloc[i]["filename"].replace(".mp3", ".wav"),
                ),
                format="wav",
            )

        for i in range(len(self.df)):
            file = os.path.join(
                self.config.path_data,
                self.df.iloc[i]["filename"].replace(".mp3", ".wav"),
            )
            y, sr = librosa.core.load(file, sr=self.config.sample_rate, mono=True)
            dir, name = os.path.split(file)
            sf.write(os.path.join(dir, "new_" + name), y, sr, subtype="PCM_16")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="File path for config file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    scraper = Scraper(args.config)
    scraper.run()
