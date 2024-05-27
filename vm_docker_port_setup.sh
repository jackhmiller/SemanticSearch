#!/bin/bash

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -w "$1" > /dev/null 2>&1
}

restart_container() {
  local container_name="$1"

  if container_exists "$container_name"; then
    echo "Restarting container: $container_name"
    sudo docker restart "$container_name"
    if [ $? -eq 0 ]; then
      echo "Container $container_name restarted successfully."
    else
      echo "Failed to restart container $container_name."
    fi
  else
    echo "Container $container_name does not exist."
  fi
}


if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <container_name>"
  exit 1
fi

container_name="elasticsearch"

restart_container "$container_name"


REMOTE_USER=$1
REMOTE_HOST=$2
REMOTE_PORT=5000
LOCAL_PORT=5000

ssh -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}
