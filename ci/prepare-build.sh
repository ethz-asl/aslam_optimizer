#!/bin/bash -e

if [[ $(uname) == "Linux" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get install -y libsuitesparse-dev
else
  echo "Platform $(uname) is not supported!"
fi