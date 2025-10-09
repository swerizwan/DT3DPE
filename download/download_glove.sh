echo -e "Downloading glove"
gdown --fuzzy https://drive.google.com/file/d/1d1ZYBYZ6zcFq42eKeLARgJpGEqLFjOK2/view?usp=sharing
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"