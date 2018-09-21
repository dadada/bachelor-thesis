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

    iotlab-node -sto
    iotlab-node -sta

    _reset_node ${RESET} &
    sleep ${PERIOD}
}

_recompile_firmware() { # 1: BUILDDIR, 2: EXPDIR, 3: MODE, 4: OPTS

    echo Recompiling firmware ${3}

    rm ${1}/rpl-eval-sink.${PLATFORM}
    make -C ${1} TARGET=${PLATFORM} clean
    make -C ${1} -j4 TARGET=${PLATFORM} WITH_RPL_RESTORE_NO_INVOKE=1 $4
    install ${1}/rpl-eval-sink.${PLATFORM} ${2}/${3}-sink

    rm ${1}/rpl-eval-source.${PLATFORM}
    make -C ${1} TARGET=${PLATFORM} clean
    make -C ${1} -j4 TARGET=${PLATFORM} $4
    install ${1}/rpl-eval-source.${PLATFORM} ${2}/${3}-source
}

pre() { # 1: expdir

    if [ -z ${BUILDDIR+x} ]; then
        echo "BUILDDIR not defined..."
        BUILDDIR="/home/tim/src/contiki-inga/examples/ipv6/rpl-print-topo"
    fi

    if [ ! -z ${WITH_RECOMPILE+x} ]; then
        _recompile_firmware ${BUILDDIR} ${1} "n" ""
        _recompile_firmware ${BUILDDIR} ${1} "h" "RPL_RESTORE=1"
        _recompile_firmware ${BUILDDIR} ${1} "hs" "RPL_RESTORE=1 RPL_RESTORE_USE_UIDS=1"
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
