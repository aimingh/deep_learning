# jetbot with tensorflow

    
## 1. 기존환경
### jetbot software install using SD card Image [Jetbot documents](https://jetbot.org/master/software_setup/sd_card.html)  
jetpack에 따라 가능한 tensorflow 버전이 다르니 확인 후 업그레이드 [TensorFlow For Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel)
* jetpack 4.4.1
* jetbot version 0.4.2
* tensorflow 1.15.2
* cuda 10.2.89
* cudnn 8.0.0

## 2. Upgrade the tensorflow from 1.15.2 to 2.3.1 in jetbot
Tensorflow 업그레이드 (1.15.2 -> 2.3.1)

* 기존 tensorflow 제거
    ```
    pip3 uninstall -y tensorflow
    ```

* Prerequisites and Dependencies
    * Tensorflow에서 요구하는 시스템 패키지 설치
    ```
    apt update
    apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    ```

    * pip3 업그레이드
    ```
    apt install python3-pip
    pip3 install -U pip testresources setuptools==49.6.0 
    ```
    * 파이썬 의존성 패키지 설치
    ```
    pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
    ```

* Tensorflow 설치
    ```
    pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow
    ```

## 참고
* [Installing TensorFlow For Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)