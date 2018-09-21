#!/bin/bash

# experiments are kept like this:
# EXPID/{files}
# EXPID2/{files}

set -u
set -e

AUTHRC=${HOME}/.iotlabrc
RUSER=$(cat ${AUTHRC} | cut -d':' -f1)
# dummy
RHOST="localhost"
OLDEXPDIR=$(dirname ${2})

# find out the host for the site we are running the experiment on
RHOST=$(jq '.nodes[0]' ${2} | cut -d . -f2- | cut -d'"' -f-1)

# source hooks specific for experiment
if [ -f ${OLDEXPDIR}/hooks.sh ]; then
    . ${OLDEXPDIR}/hooks.sh
else
    # hook functions
    pre() {
        echo $1
    }
    during() {
        echo $1
    }
    post() {
        echo $1
    }
fi

_firmwares() { # 1: experiment.json
    FIRMWARENAMES=$(jq '.firmwareassociations[].firmwarename' ${1} | cut -d'"' -f2)
    for f in ${FIRMWARENAMES}; do
        echo -n " -l $(dirname ${1})/${f}"
    done
}

_name() { # 1: experiment.json
    jq '.name' ${1} | cut -d '"' -f1
}

auth() {
    xargs -n 1 auth-cli -u
}

track() { # 1: expdir
    mkdir -p ${1}
    (yes | ssh -l ${RUSER} ${RHOST} serial_aggregator > ${1}/serial.log 2> ${1}/serial_error.log) &
    (ssh -n -l ${RUSER} ${RHOST} sniffer_aggregator -o - > ${1}/sniffer.pcap 2> ${1}/sniffer_error.log) &
}

load() { # 1: experiment.json
    iotlab-experiment load -f ${1} $(_firmwares ${1}) | jq '.id'
}

save() { # 1: id
    (
        iotlab-experiment get -i ${1} -a
        tar -xvf ${1}.tar.gz
    )
    scp -r ${RUSER}@${RHOST}:.iot-lab/${1} .
}

record() { #1: odlexp json, 2: new expid
    EXPID=$2

    # monitor and control experiment
    while iotlab-experiment wait -i ${EXPID} --state=Running --step 1 --timeout 36000 ; [ $? -ne 0 ]; do
        echo ${EXPID} waiting
    done

    track ${EXPID} &
    during ${OLDEXPDIR} ${EXPID} &

    # wait until finished and execute post hook
    iotlab-experiment wait -i ${EXPID} --state=Terminated --timeout 36000 --step 1
    post ${EXPID}

    # save experiment files from API
    save ${EXPID}

    # make copy of used hooks
    #install ${PWD}/${OLDID}/hooks.sh ${PWD}/${EXPID}
}

run() { # 1: path to experiment.json
    # hook for further preparations
    pre ${OLDEXPDIR}

    # load the experiment and get the ID of the new run
    EXPID=$(load ${1})

    #record ${1} ${EXPID}
}

if [ ! -f ${AUTHRC} ]; then
    auth
fi


if [ ${1} == 'run' ]; then
    run ${2}
fi

if [ ${1} == 'record' ]; then
    record ${2} ${3}
fi

if [ ${1} == 'record-all' ]; then
    for exp in $(iotlab-experiment get -l --state=Waiting | jq '.[][] | .id' | sort); do
        record ${2} $exp
    done
fi
