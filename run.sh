#!/usr/bin/env bash
DOCKER_FLAGS=""
ARGS=""
USER=$(whoami)
HOSTNAME="IOU-CONTAINER"

error() {
    echo "u do sumting wong"
}

while getopts "df:g:ijpu:*" option; do
    case $option in
        d) DOCKER_FLAGS+="-d ";;
        g) DOCKER_FLAGS+="--gpus ${OPTARG} ";;
        i) DOCKER_FLAGS+="--net host ";;
        p) DOCKER_FLAGS+="--publish 5555:8888 ";;
        u) USER=${OPTARG};;
        *) error; exit;;	
    esac
done

# $@ is an array or something, start at $OPTIND and rest
ARGS+=${@:$OPTIND}

docker run ${DOCKER_FLAGS} -it --hostname ${HOSTNAME} \
    --user ${USER} \
    -v "$(pwd)/volume":/project \
    --rm --name iou-nam012-cntr nam012-iou ${ARGS}

