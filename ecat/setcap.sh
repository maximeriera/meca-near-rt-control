#!/bin/sh
# Capabilities meca_ecat needs without running as root: raw EtherCAT frames (cap_net_raw,
# cap_net_admin), SCHED_FIFO threads (cap_sys_nice), mlockall (cap_ipc_lock).
# File capabilities are lost every time the binary is rebuilt: run this after each build.
#   ecat/setcap.sh                      # target/release/meca_ecat
#   ecat/setcap.sh path/to/meca_ecat
set -e
BIN=${1:-$(dirname "$0")/target/release/meca_ecat}
sudo setcap cap_net_raw,cap_net_admin,cap_sys_nice,cap_ipc_lock+ep "$BIN"
getcap "$BIN"
