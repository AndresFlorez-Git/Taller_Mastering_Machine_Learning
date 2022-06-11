sudo apt-get install python3-pip
sudo apt-get install python3-picamera

sudo apt-get update
sudo apt-get upgrade


wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
python3 -m pip uninstall tensorflow
python3 -m pip install tensorflow-2.4.0-cp37-none-linux_armv7l.whl

pip3 install opencv-contrib-python; sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev  libqtgui4  libqt4-test
pip3 install opencv-python