# Practical Deep Learning

## Docker 사용 방법
 ``` 
 ## 소스 다운로드
 git clone https://github.com/OkDoky/PracticalDeepLearning.git -b master
 
 ## dockerfile 경로로 이동
 cd PracticalDeepLearning/docker

 ## docker image build(이미지 이름은 pdl)
 docker build -t pdl:latest .  

 ## docker build 이후 컨테이너 실행 및 프로그램 실행
 docker run -it -d --restart always --name "pdl" pdl:latest /bin/bash

 ## docker container 접근
 docker exec -it pdl /bin/bash
  ```

