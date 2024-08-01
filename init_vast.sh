GIT_PAT="FILL_WITH_GIT_PAT" 
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y update 
sudo apt install -y python3.10 python3.10-venv psmisc neovim unzip git
python3.10 -m venv env
. env/bin/activate
rm -rf road_segmentation && git clone "https://{GIT_PATH}@github.com/thesstefan/road_segmentation.git"
cd road_segmentation && pip install --no-input ".[dev]"
cd ..
mkdir -p data && cd data
pip install --no-input gdown
gdown --fuzzy https://drive.google.com/file/d/1ocUztpgoYB70ex0-M_DQQJvzX77i0OQa/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1jvcXcMOac_oaKBWSxEoBxLa-oWok7GtN/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1fBje-4qBP5jlfiq9cPr4uXLBTwqT28gM/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1AHeXosCm5c58I_TcCDM37ha33diWuKnJ/view?usp=drive_link 
unzip ETHZ_Dataset.zip
unzip Massachusetts.zip
unzip DeepGlobe.zip
unzip EPFL.zip
cd ..
