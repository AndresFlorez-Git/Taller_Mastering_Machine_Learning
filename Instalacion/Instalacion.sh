sudo apt-get install python3-pip
sudo apt-get install python3-picamera

sudo apt-get update -y
sudo apt-get upgrade -y


wget https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/main/previous_versions/download_tensorflow-2.8.0-cp39-none-linux_aarch64_numpy1221.sh

chmod +x download_tensorflow-2.8.0-cp39-none-linux_aarch64_numpy1221.sh

./download_tensorflow-2.8.0-cp39-none-linux_aarch64_numpy1221.sh

sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython3 libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev build-essential cmake pkg-config libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libhdf5-serial-dev libhdf5-103 libqt5gui5 libqt5webkit5 libqt5test5 python3-pyqt5

python3 -m pip install -U wheel mock six

sudo pip uninstall tensorflow
python3 -m pip uninstall tensorflow

python3 -m pip install protobuf==3.20.1
python3 -m pip install numpy==1.22.1

python3 -m pip install tensorflow-2.8.0-cp39-none-linux_aarch64.whl

pip3 install opencv-python==4.5.2.54
pip3 install scikit-learn==0.24.2