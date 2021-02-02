# jetson-inference

## installation
### 1.Running the Docker Container
* docker를 이용하여 jetson-inference가 설치된 컨테이너를 사용할 수 있다.
    ```
    git clone --recursive https://github.com/dusty-nv/jetson-inference
    cd jetson-inference
    docker/run.sh
    ```

### 2.Building the Project from Source
* 직접 소스를 빌드하여 사용
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
jetbot에서 사용하고 있는 기존 jetbot_jupyter에 설치하여 사용
* 기존 과정과 비슷하게 시작한다. hetbot에서 jetbot_jupyter 컨테이너에 접속하고 시작한다.
    ```
    sudo apt-get update
    sudo apt-get install git cmake libpython3-dev python3-numpy
    git clone --recursive https://github.com/dusty-nv/jetson-inference
    cd jetson-inference
    ```
* 다음 리스트에 있는 CMakeLists.txt 파일에서 nvcaffe_parser를 nvparsers로 바꿔준다   
    * jetson-inference/CMakeLists.txt
    * jetson-inference/tools/trt-bench/CMakeLists.txt
    * jetson-inference/tools/trt-console/CMakeLists.txt
    ```
    # from
    target_link_libraries(jetson-inference jetson-utils nvinfer nvinfer_plugin nvcaffe_parser)
    # to
    target_link_libraries(jetson-inference jetson-utils nvinfer nvinfer_plugin nvparsers)
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