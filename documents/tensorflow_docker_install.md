# Tensorflow-GPU Install with docker (Docker를 이용하여 텐서플로우-GPU 설치)

## 설치 환경
* Ubuntu 20.04

## 1. 그래픽 드라이버 설치 
* 그래픽 카드 확인
	```
	lshw -C display
	lspci | grep VGA
	```
* 드라이버 확인 (마지막에 recommended 되어있는 드라이버가 추천 드라이버)
	```
	ubuntu-drivers devices
	```
* 드라이버 저장소 추가
	```
	sudo add-apt-repository ppa:graphics-drivers/ppa
	sudo apt update
	```
* 설치 가능한 드라이버 목록 출력 
	```
	apt-cache search nvidia
	```
* 목록 중에 추천 드라이버 확인 (위의 추천 드라이버로 검색)
	```
	apt-cache search nvidia | grep nvidia-driver-460
	```
* 드라이버 설치 (위의 추천 드라이버로 설치)
	```
	sudo apt-get install nvidia-driver-460
	```
* 재시작
	```
	sudo reboot
	```

## 2. docker 설치
### 2.1 이전 버전 제거
* 만약 이전 버전이 설치 되어 있으면 삭제 (처음 설치라면 무시)
	```
	sudo apt-get remove docker docker-engine docker.io containerd runc
	```
### 2.2 저장소를 사용한 설치
* apt update 및 필요한 패키지 설치
	```
	sudo apt-get update
	sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
	```
* GPG key 추가
	```
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	```
* key가 추가 되었는지 지문의 마지막을 이용하여 확인
	```
	sudo apt-key fingerprint 0EBFCD88
	```
* 시스템 아키텍처 확인
	```
	arch
	```
* 시스템 아키텍처에 따라 저장소 설정
	```
	sudo add-apt-repository \
		"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
		$(lsb_release -cs) \
		stable"
	```
	* x86_64 or amd64 -> arch=amd64
	* armhf -> arch=armhf
	* arm64 -> arch=arm64
### 2.3 도커 엔진 설치
* 	apt pakage update 후 docker engine 설치
	```
	sudo apt-get update
	sudo apt-get install docker-ce docker-ce-cli containerd.io
	```
* 설치가 되었는지 도커 확인   
	* [root 권한을 사용하지 않고 docker 관리](docker.md)  
	계정을 docker 그룹에 추가하면 sudo를 안써도 된다.   
	위의 과정을 안했을 경우에는 docker에 root 권한이 필요하므로 sudo를 붙여서 실행

	```
	docker -v
	docker run hello-world
	```

## 3. nvidia-docker 설치
* stable 버전 저장소 설치 및 GPG 키 등
	```
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
	   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
	   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
	   | sudo tee /etc/apt/sources.list.d/nvidia-docker.list	
	```	   
* nvidia-docker2 설치
	```
	sudo apt-get update
	sudo apt-get install -y nvidia-docker2
	```
* docker deamon 재시작
	```
	sudo systemctl restart docker
	```	
* base CUDA 컨테이너가 작동하는지 테스트
	```
	docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
	```

## 4. Docker를 이용하여 Tensorflow 설치
* Tensorflow 이미지 다운로드 gpu, jupyter 버전
	```
	docker pull tensorflow/tensorflow:latest-gpu-jupyter
	```
* Tensorflow 도커 컨테이너 시작 (포트 충돌이 있을 수 있으니 포트 확인 필요)
	```
	docker run \
		--name tf \
		--gpus all \
		-it \
		-p 8888:8888 \
		-p 6006:6006 \
		-v ~/docker/data:/data \
		tensorflow/tensorflow:latest-gpu-jupyter
	```
	--name: 컨테이너 이름   
	--gpus all: nvidia-docker gpu 사용   
	-i: 상호 입출력   
	-t: tty 활성화 bash셀 사용 가능   
	-p: 포트 연결 (8888 쥬피터 노트북을 위해 사용, 6006 텐서보드를 위해 사용)   
	-v: 볼륨 컴퓨터와 공유하는 폴더 지정
* Tensorflow 이전 버전 설치  
	```
	docker run tensorflow/tensorflow:1.15.0-gpu-jupyter
	```
	* docker image 이름에 tag를 추가  
	* tensorflow에서 제공하는 다른 이미지에 대한 정보는 다음 링크를 통해 참고
	[tensorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow/)
* jupyter notebook 시작	  
	http://127.0.0.1:8888
	
* 컨테이너 bash terminal 시작  
	```
	docker exec -it tf /bin/bash
	```

## 참고문서
1. 도커 텐서플로우 설치 전체 과정 참고 
	* https://www.tensorflow.org/install/docker
2. 그래픽 드라이버 설치 참고
	* https://codechacha.com/ko/install-nvidia-driver-ubuntu/
3. docker 설치 참고
	* https://docs.docker.com/engine/install/ubuntu/
	* https://blog.dalso.org/linux/ubuntu-20-04-lts/13118
4. docker nvidia
	* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker