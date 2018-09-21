#!/usr/bin/env python3

import argparse
import rpl.data as p
import os
import cProfile
import pandas as pd

from sys import exit
from os.path import dirname, basename


def dumptable(db, tablename):
    dump = db.execute("SELECT * FROM %s" % tablename)

    for l in dump:
        print(l)


def parselogs(args):

    db = p.init_db(args.database[0])
    expid = args.experiment[0]

    if args.logs:
        for f in args.logs:
            if 'm3-200' in f:
                args.logs.remove(f)
                print('Ignoring log file for resetting node %s' % 'm3-200')
        try:
            logs = [open(log, 'r') for log in args.logs]
            p.parse_events(db, expid, logs)
        except IOError:
            print("Failed to open files")

    phases = db.execute('''
    SELECT phase, tstart, tstop
    FROM phases
    WHERE expid = ?
    ''', (expid, )).fetchall()

    if args.serial:
        with open(args.serial[0], 'r') as serial:
            p.parse_addresses(db, expid, serial)

        addresses = pd.DataFrame(list(db.execute('SELECT * FROM addresses')), columns=['host', 'address']).set_index('address')

        with open(args.serial[0], 'r') as serial:
            p.parse_end_to_end(db, expid, phases, addresses, serial)

        with open(args.serial[0], 'r') as serial:
            p.parse_count_messages(db, expid, phases, serial)

        with open(args.serial[0], 'r') as serial:
            p.parse_dag(db, expid, phases, serial)

        with open(args.serial[0], 'r') as serial:
            p.parse_parents(db, expid, phases, serial)

        with open(args.serial[0], 'r') as serial:
            p.parse_default_routes(db, expid, phases, serial)

    if args.consumption:
        for log in args.consumption:
            hostname = basename(log).split('.')[0]

            with open(log, 'r') as oml:
                p.parse_consumption(db, expid, phases, hostname, oml)

    db.commit()

    #dumptable(db, 'addresses')
    #dumptable(db, 'phases')
    #dumptable(db, 'consumption')
    #dumptable(db, 'dag_nodes')
    #dumptable(db, 'dag_edges')
    #dumptable(db, 'resets')
    #dumptable(db, 'end_to_end')
    #dumptable(db, 'dag_evolution')
    #dumptable(db, 'default_route_changes')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process file names.')
    parser.add_argument('experiment', nargs=1, help='experiment id')
    parser.add_argument('--database', '-b', type=str, nargs=1, help='a sqlite3 database file', default=[':memory:'])
    parser.add_argument('--serial', '-s', type=str, nargs=1, help='a serial log file (ASCII)')
    parser.add_argument('--sniffer', '-n', type=str, nargs='+', help='a sniffer log file (PCAP)')
    parser.add_argument('--consumption', '-c', type=str, nargs='+', help='a list of consumption logs (OML)')
    parser.add_argument('--logs', '-l', type=str, nargs='+', help='a list of experiment logs (OML)')
    parser.add_argument('--radio', '-r', type=str, nargs='+', help='a list of RSSI logs (OML)')

    args = parser.parse_args()
    parselogs(args)
