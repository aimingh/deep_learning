# Tensorflow


## Tensorboard
참고: [Tensorboard getting started](https://www.tensorflow.org/tensorboard/get_started)
```
tensorboard --logdir=path/to/logs --port=6006 --host=0.0.0.0
```	
* --logdir: 로그 파일 경로
* --port: 포트 
* --host: 외부 ip에 대한 허가

## Docker tensorflow service
Docker tensorflow를 서비스로 등록하여 부팅시 도커 서버가 자동으로 실행  
참고: [Docker 컨테이너 자동시작](https://help.iwinv.kr/manual/read.html?idx=572)

* 작업 전 도커 컨테이너 중지
tensorflow 컨테이너 이름: tf
    ```
    docker ps
    docker stop tf
    ```
* 서비스 파일 생성
    ```
    cd /etc/systemd/system
    vim docker_tf.service
    ```
* 서비스 파일 내용 추가
    ```
    [Unit]
    Wants=docker.service
    After=docker.service
    
    [Service]
    RemainAfterExit=yes
    ExecStart=/usr/bin/docker start tf
    ExecStop=/usr/bin/docker stop tf
    
    [Install]
    WantedBy=multi-user.target
    ```
* 서비스 활성화
    ```
    systemctl enable docker_tf.service
    ```
* 서비스가 활성화 되었는지 확인
    ```
    systemctl list-unit-files | grep docker_tf
    ```
* 재시작하여 자동으로 실행되는지 확인
    ```
    sudo reboot
    ```


