#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUTF8=1
workspace=${ARBOIDS_VRX_WS:-"$repo/vrx_ws"}
venv=${ARBOIDS_VRX_VENV:-"$repo/.vrx-venv"}
jobs=${ARBOIDS_BUILD_JOBS:-4}
if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo 'First install ROS/Gazebo using scripts/setup_vrx_ubuntu.sh.' >&2
  exit 2
fi
if [ ! -f "$repo/.vrx-assets/manifest.json" ]; then
  /usr/bin/python3 -X utf8 "$repo/scripts/fetch_vrx_assets.py" --repo "$repo"
fi
mkdir -p "$workspace/src"
if [ ! -e "$workspace/src/arboids_vrx" ]; then
  ln -s "$repo/vrx" "$workspace/src/arboids_vrx"
fi
set +u
source /opt/ros/humble/setup.bash
set -u
cd "$workspace"
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs" MAKEFLAGS="-j$jobs"
colcon build --symlink-install --parallel-workers 2 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3
if [ ! -x "$venv/bin/python" ]; then
  /usr/bin/python3 -m venv --system-site-packages "$venv"
fi
"$venv/bin/python" -X utf8 -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.1'
"$venv/bin/python" -X utf8 -m pip install 'numpy==1.26.4'
"$venv/bin/python" -X utf8 -m pip check
cat > "$workspace/activate.bash" <<EOF
source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
source "$venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUTF8=1
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=86
export GZ_FUEL_CACHE_PATH="$repo/.vrx-assets/fuel"
export GZ_PARTITION=arboids
EOF
echo "ARBOIDS_VRX_BUILD_COMPLETE: source $workspace/activate.bash"
