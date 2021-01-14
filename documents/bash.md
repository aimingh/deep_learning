# Bash 명령어 기록
Bash에서 사용하는 명령어 모음

## 시스템 system
* Ubuntu version 우분투 버전
    ```
    cat /etc/*release
    ```
* Ubuntu kernel version 우분투 커널 버전
    ```
    uname -a
    ```

## Tensorflow 
* tensorflow version 텐서플로우 버전
  python에서 확인
    ```
    import tensorflow as tf
    tf.__version__
    ```
* cuda version
    ```
    cat /usr/local/cuda/version.txt
    ```
* cudnn version
    cudnn.h에서 안나오면 cudnn_version.h에서 확인
    ```
    cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
    cat /usr/include/cudnn_version.h
    ```

## Docker 
* Docker version
    ```
    docker version
    ```
* Container 생성
    ```
    docker run
    ```
* Container 시작
    ```
    docker start [container_id or container_name]
    ```
* Container 종료
    ```
    docker strop [container_id or container_name]
    ```
* Container 강제 종료
    ```
    docker kill [container_id or container_name]
    ```
* Container 삭제
    ```
    docker rm [container_id or container_name]
    ```

## 참고문헌