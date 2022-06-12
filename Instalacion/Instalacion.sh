sudo apt-get install python3-pip
sudo apt-get install python3-picamera

sudo apt-get update
sudo apt-get upgrade


wget https://github.com/PINTO0309/Tensorflow-bin/blob/main/previous_versions/download_tensorflow-2.5.0-cp39-none-linux_aarch64_numpy1195.sh

chmod +x download_tensorflow-2.5.0-cp39-none-linux_aarch64_numpy1195.sh

./download_tensorflow-2.5.0-cp39-none-linux_aarch64_numpy1195.sh

python3 -m pip uninstall tensorflow
python3 -m pip install tensorflow-2.5.0-cp39-none-linux_aarch64_numpy1195.whl

pip3 install opencv-python==4.5.2.54
pip3 install scikit-learn==0.24.2