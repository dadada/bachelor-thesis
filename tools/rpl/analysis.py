#!/usr/bin/env/python


import json
import numpy

from sys import argv
from datetime import time, datetime


def node_name(node_name):
    return node_name.split('.')[0]


CPU_M3 = { 'current': 0.050, 'voltage': 3.3 }

RADIO_M3_SLEEP = { 'current': 0.00002, 'voltage': 3.3 }
RADIO_M3_TRX_OFF = { 'current': 0.0004, 'voltage': 3.3 }
RADIO_M3_RX_ON = { 'current': 0.0103, 'voltage': 3.3 }

""" TODO not included in the data sheet, only values for +3dBm, 0dBm and -17dBm,
so more a guess based on what the CN measured
"""
RADIO_M3_BUSY_TX = { 'current': 0.010, 'voltage': 3.3 }

RTIME_SECOND_M3 = 1
POWERTRACE_INTERVAL = 1


def consumption(energest, current, voltage):
    """mW"""
    return energest * current * voltage / (RTIME_SECOND_M3 * POWERTRACE_INTERVAL)


def duty_cycle(tx, rx, cpu, lpm):
    """%"""
    return (tx + rx) / (cpu, lpm)
