#!/bin/sh

# in seconds
PERIOD=600
RESETGUARD=60
SINK=157
LOCATION="lille"
PLATFORM="iotlab-m3"
ARCH="m3"
RESET=200

_resources() {
    experiment-cli get -r
}

_reset_node() { # 1: node
    # pick random point in time between guard times
    SELECTED=$(shuf -i ${RESETGUARD}-$((${PERIOD}-${RESETGUARD})) -n 1)
    sleep ${SELECTED}
    iotlab-node -sto -l "${LOCATION},${ARCH},${1}"
    iotlab-node -sta -l "${LOCATION},${ARCH},${1}"
}

# for some reason, no DAG is built if all nodes start up simultaneously
_phased_start() {
    iotlab-node -sto
    iotlab-node -sta -l lille,m3,47+49+51+53+${SINK}
    sleep 5
    iotlab-node -sta
}

_run_firmware() {
    echo "Flashing ${1} and ${2}"
    iotlab-node -up ${1} -e "${LOCATION},${ARCH},${SINK}"
    iotlab-node -up ${2} -l "${LOCATION},${ARCH},${SINK}"

    sleep ${PERIOD}

    _reset_node ${RESET} &
    sleep ${PERIOD}
}

_recompile_firmwares() { # 1: BUILDDIR, 2: EXPDIR

    echo Recompiling firmwares

    make -C ${1} TARGET=iotlab-m3 clean
    rm ${1}/rpl-eval-sink.${PLATFORM}
    rm ${1}/rpl-eval-source.${PLATFORM}
    make -C ${1} -j4 TARGET=iotlab-m3 WITH_COMPOWER=1
    install ${1}/rpl-eval-sink.${PLATFORM} ${2}/n-sink
    install ${1}/rpl-eval-source.${PLATFORM} ${2}/n-source

    make -C ${1} TARGET=iotlab-m3 clean
    rm ${1}/rpl-eval-sink.${PLATFORM}
    rm ${1}/rpl-eval-source.${PLATFORM}
    make -C ${1} -j4 TARGET=iotlab-m3 WITH_COMPOWER=1 RPL_RESTORE=1
    install ${1}/rpl-eval-sink.${PLATFORM} ${2}/h-sink
    install ${1}/rpl-eval-source.${PLATFORM} ${2}/h-source

    make -C ${1} TARGET=iotlab-m3 clean
    rm ${1}/rpl-eval-sink.${PLATFORM}
    rm ${1}/rpl-eval-source.${PLATFORM}
    make -C ${1} -j4 TARGET=iotlab-m3 WITH_COMPOWER=1 RPL_RESTORE=1 RPL_RESTORE_USE_UIDS=1
    install ${1}/rpl-eval-sink.${PLATFORM} ${2}/hs-sink
    install ${1}/rpl-eval-source.${PLATFORM} ${2}/hs-source
}

pre() { # 1: expdir

    if [ -z ${BUILDDIR+x} ]; then
        echo "BUILDDIR not defined..."
        BUILDDIR="/home/tim/src/contiki-inga/examples/ipv6/rpl-print-topo"
    fi

    if [ ! -z ${WITH_RECOMPILE+x} ]; then
        _recompile_firmwares ${BUILDDIR} ${1}
    fi
}

during() { # 1: expdir
    _run_firmware ${1}/n-source ${1}/n-sink
    _run_firmware ${1}/h-source ${1}/h-sink
    _run_firmware ${1}/hs-source ${1}/hs-sink
}

post() {
    echo Finishing experiment $1
}
