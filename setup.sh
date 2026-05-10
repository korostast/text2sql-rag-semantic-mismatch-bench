#!/bin/bash

echo ">>> Initializing submodules..."
set -e
git submodule update --init --recursive
git submodule status

echo ">>> Setting up virtual environment..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -r requirements.txt

echo ">>> Downloading datasets: mini_dev, train, dev. Note, that it will take around 15 GB!"

echo ">>> Downloading mini_dev..."
wget -O data/mini_dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
echo ">>> mini_dev downloaded successfully. Unzipping..."
unzip data/mini_dev.zip -d data/
mv data/minidev data/mini_dev
mv data/mini_dev/MINIDEV data/mini_dev/MINIDEV_sqlite
echo ">>> mini_dev unzipped successfully. Dataset is located at data/mini_dev, subfolders 'MINIDEV_sqlite', 'MINIDEV_mysql', 'MINIDEV_postgresql'"
rm data/mini_dev.zip

echo ">>> Downloading train..."
wget -O data/train.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
echo ">>> train downloaded successfully. Unzipping..."
unzip data/train.zip -d data/
echo ">>> train unzipped successfully. Dataset is located at data/train"
rm -rf data/train.zip data/__MACOSX
echo ">>> Note: data/train/train_databases.zip is not unzipped because it is nmot used in ours experiments. If you need it, then proceed unzipping manually"

echo ">>> Downloading dev..."
wget -O data/dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
echo ">>> dev downloaded successfully. Unzipping..."
unzip data/dev.zip -d data/
mv data/dev_20240627 data/dev
unzip data/dev/dev_databases.zip -d data/dev
echo ">>> dev unzipped successfully. Dataset is located at data/dev"
rm -rf data/dev.zip data/dev/dev_databases.zip data/__MACOSX data/dev/__MACOSX

echo ">>> Setup complete. To activate the virtual environment, run: source .venv/bin/activate"
