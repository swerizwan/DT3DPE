cd checkpoints

cd t2m 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1tuHdgHGMeSSPOy2xPpZLBKMa4YSiX5Dy/view?usp=sharing
echo -e "Unzipping humanml3d_evaluator.zip"
unzip humanml3d_evaluator.zip

echo -e "Clearning humanml3d_evaluator.zip"
rm humanml3d_evaluator.zip

cd ../kit/
echo -e "Downloading pretrained models for KIT-ML dataset"
gdown --fuzzy https://drive.google.com/file/d/1k-k1AU40WwvYznqe-eqObOU-__HkG1Dh/view?usp=sharing

echo -e "Unzipping kit_evaluator.zip"
unzip kit_evaluator.zip

echo -e "Clearning kit_evaluator.zip"
rm kit_evaluator.zip

cd ../../

echo -e "Downloading done!"
