# Installing Dependencies

```
sudo apt-get install python-opencv
```
```
sudo apt-get install cmake python2.7-dev
sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev \
libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
libatlas-base-dev gfortran
```

### Get source of OpenCV
```
wget -O opencv.tgz https://github.com/Itseez/opencv/archive/3.1.0.tar.gz
tar xf opencv.tgz
mv opencv-3.1.0/ opencv
```

### Get SIFT
```
wget -O opencv_contrib.tgz https://github.com/Itseez/opencv_contrib/archive/3.1.0.tar.gz
tar xf opencv_contrib.tgz
mv opencv_contrib-3.1.0/ opencv_contrib
```

### CMake
```
cd opencv
mkdir build/
cd build/
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ../..
```

### Build OpenCV with SIFT
```
make -j4
```

### Install OpenCV with SIFT
```
sudo make install
```
# Running the code
Check and set the configurations in `config.py` file. Then to run the training on the dataset, run
```
python run.py
```

## Testing / Verifying the trained code with any image
Run
```
python check.py
```
and provide the result directory generated by run.py script inside the results directory when asked. After that, provide the file path to the image which has to be tested.