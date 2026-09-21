#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
apt_options=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
if [ -n "${ARBOIDS_OSRF_PROXY:-}" ]; then
  apt_options+=(-o "Acquire::https::Proxy::packages.osrfoundation.org=$ARBOIDS_OSRF_PROXY")
fi
. /etc/os-release
if [ "${ID}" != ubuntu ] || [ "${VERSION_ID}" != 22.04 ]; then
  echo 'ARBoids VRX requires Ubuntu 22.04 for ROS Humble / Gazebo Garden.' >&2
  exit 2
fi
apt-get "${apt_options[@]}" update -qq
apt-get "${apt_options[@]}" install -y --no-install-recommends ca-certificates curl gnupg lsb-release
if [ ! -s /usr/share/keyrings/arboids-ros-keyring.gpg ]; then
  curl --fail --location --retry 3 --silent --show-error --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/arboids-ros-keyring.gpg
fi
if [ ! -s /usr/share/keyrings/arboids-gazebo-keyring.gpg ]; then
  curl --fail --location --retry 3 --silent --show-error --connect-timeout 15 --max-time 120 https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/arboids-gazebo-keyring.gpg
fi
gpg --batch --show-keys /usr/share/keyrings/arboids-ros-keyring.gpg > /dev/null
gpg --batch --show-keys /usr/share/keyrings/arboids-gazebo-keyring.gpg > /dev/null
printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/arboids-ros-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main' > /etc/apt/sources.list.d/arboids-ros2.list
printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/arboids-gazebo-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main' > /etc/apt/sources.list.d/arboids-gazebo.list
apt-get "${apt_options[@]}" update -qq
apt-cache policy gz-garden ros-humble-ros-gzgarden python3-sdformat13
apt-get "${apt_options[@]}" install -y --no-install-recommends \
  ros-humble-ros-base ros-humble-ros-gzgarden ros-humble-xacro \
  ros-humble-tf-transformations gz-garden python3-sdformat13 \
  python3-colcon-common-extensions python3-rosdep python3-venv python3-dev \
  build-essential cmake git libeigen3-dev mesa-utils xvfb xauth \
  libgl1-mesa-dri libegl1 libglib2.0-0
echo 'ARBOIDS_VRX_DEPENDENCIES_INSTALLED'
