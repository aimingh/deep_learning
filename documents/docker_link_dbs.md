# Docker 컨테이너 연결
mariadb container와 mongodb container를 만들고 django container에서 사용할 수 있도록 연결한다.

## docker mariadb 설치
```
docker pull mariadb
docker run --name mariadb_name \
            -e MYSQL_ROOT_PASSWORD=your_password \
            mariadb
```

## docker mongodb 설치
```
docker pull mongo
docker run --name mongodb_name \
        -d mongo
```

## tensorflow django container와 --link 옵션을 이용하여 연결
지난번에 설치했던 텐서플로우 이미지를 이용하여 컨테이너 생성 후 장고를 설치하여 db 사용
```
docker run --name web_server_name \
            --gpus all \
            --link mariadb_name \
            --link mongodb_name \
            -it \
            -p 8888:8888 \
            -p 6006:6006 \
            tensorflow/tensorflow:latest-gpu-jupyter
```

## db에 맞는 클라이언트 설치
```
apt update
apt install mysqlclient
apt install mongodb-clients
```

## container에서 db가 작동하는지 확인
```
mysql -h mariadb_name -u root -p
mongo mongodb_name:27017
```