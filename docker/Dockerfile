FROM python:3.8

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git

RUN git clone https://github.com/OkDoky/PracticalDeepLearning.git -b master /root/PracticalDeepLearning
WORKDIR /root/PracticalDeepLearning
RUN pip install -r requirements.txt

CMD [ "bash" ]
