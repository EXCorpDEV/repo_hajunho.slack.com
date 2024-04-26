#!/bin/bash

# dpkg 명령어를 사용하여 "cuda"와 관련된 패키지 목록을 가져옵니다.
packages=$(dpkg -l | grep -i cuda | awk '{print $2}')

# 패키지 목록을 반복하면서 각 패키지를 삭제합니다.
for package in $packages; do
    sudo dpkg --purge --force-all "$package"
done

echo "CUDA 관련 패키지 삭제가 완료되었습니다."
#sudo apt-cache madison cuda
#sudo apt remove --purge ^cuda-12.*
#sudo apt remove --purge -s ^cuda-12.*
#sudo apt autoremove
#sudo apt clean
