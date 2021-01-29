# jetson-inference

## installation
### 1.Running the Docker Container
*     
    ```
    git clone --recursive https://github.com/dusty-nv/jetson-inference
    cd jetson-inference
    docker/run.sh
    ```

### 2.Building the Project from Source
* 
    ```
    sudo apt-get update
    sudo apt-get install git cmake libpython3-dev python3-numpy
    git clone --recursive https://github.com/dusty-nv/jetson-inference
    cd jetson-inference
    mkdir build
    cd build
    cmake ../
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    ```

### 3.Building the Project from Source in jetbot docker container
* 
    ```
    sudo apt-get update
    sudo apt-get install git cmake libpython3-dev python3-numpy
    git clone --recursive https://github.com/dusty-nv/jetson-inference
    cd jetson-inference
    ```
* 다음 리스트에 있는 CMakeLists.txt 파일에서 nvcaffe_parser를 nvparser로 바꿔준다
    ```
    jetson-inference/CMakeLists.txt
    jetson-inference/tools/trt-bench/CMakeLists.txt
    jetson-inference/tools/trt-console/CMakeLists.txt
    ```
* 나머지 과정은 2와 같다.
    ```
    mkdir build
    cd build
    cmake ../
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    ```

## 참고문헌
* [Deploying Deep Learning](https://github.com/dusty-nv/jetson-inference)  
* [Build error when using make in official nvidia docker](https://github.com/dusty-nv/jetson-inference/issues/610)