#!/usr/bin/env python

import argparse
import rpl.data as p
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sys import exit
from os.path import dirname, basename


tubspalette = ['#be1e3c', '#ffc82a', '#e16d00', '#711c2f', '#acc13a', '#6d8300', '#00534a', '#66b4d3', '#007a9b', '#003f57', '#8a307f', '#511246', '#4c1830']
phase_names = ['N', 'R', 'H', 'HR', 'HS', 'HSR']
versions = ['Contiki', 'Hardened', 'Hardened with UIDs']


def plot_consumption(db, args):
    data = db.execute(
        '''
        SELECT c.phase, SUM(consumption), SUM(dios), SUM(daos), SUM(dis)
        FROM overhead AS o
        JOIN consumption AS c ON c.expid = o.expid AND c.phase = o.phase AND c.host = o.source
        GROUP BY o.expid, o.phase
        ''')

    data = pd.DataFrame(data.fetchall(), columns=['phase', 'consumption', 'dios', 'daos', 'dis'])
    data['phase'] = data['phase'].map({1: 'N', 2: 'R', 3: 'H', 4: 'HR', 5: 'HS', 6: 'HSR'})

    sns.pairplot(data, hue='phase', x_vars=['dios', 'daos', 'dis'], y_vars=['consumption'], kind='reg', markers='.')


def plot_overhead(db, args):
    overheads = db.execute(
        '''
        SELECT phase, SUM(dios), SUM(daos), SUM(dis), (phase - 1) % 2 == 1
        FROM overhead
        WHERE source != 'm3-200'
        GROUP BY expid, phase
        ''')

    data = pd.DataFrame(overheads.fetchall(), columns=['phase', 'dios', 'daos', 'dis', 'reset'])

    dios = data[['phase', 'reset', 'dios']]
    dios['message type'] = 'dio'
    dios = dios.rename(index=str, columns={'dios': 'count'})

    daos = data[['phase', 'reset', 'daos']]
    daos['message type'] = 'dao'
    daos = daos.rename(index=str, columns={'daos': 'count'})

    dis= data[['phase', 'reset', 'dis']]
    dis['message type'] = 'dis'
    dis = dis.rename(index=str, columns={'dis': 'count'})

    frame = [dios, daos, dis]
    data = pd.concat(frame)
    data['phase'] = data['phase'].map({1: 'N', 2: 'R', 3: 'H', 4: 'HR', 5: 'HS', 6: 'HSR'})
    print(data)

    sns.barplot(data=data, hue='phase', y='message type', x='count')


def plot_packet_loss(db, args):

    changes_vs_delay = db.execute('''
    WITH s AS (
    SELECT expid, phase, source, COUNT(nexthop) AS stab
    FROM default_route_changes
    GROUP BY expid, phase, source)
    SELECT e.phase, s.stab / (tstop - tstart), loss, (e.phase - 1) / 2
    FROM end_to_end AS e
    JOIN s ON s.expid = e.expid AND e.phase = s.phase AND s.source = e.source
    JOIN phases AS p ON p.expid = e.expid AND p.phase = e.phase
    ''')

    data = pd.DataFrame(changes_vs_delay.fetchall(), columns=['phase', 'changes', 'loss', 'firmware'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    g = sns.pairplot(data, diag_kind='kde', kind='reg', hue='phase', vars=['changes', 'loss'])


def plot_rank_loss(db, args):

    rank_vs_loss = db.execute(
        '''
        SELECT de.phase, loss, avg_metric, avg_rank
        FROM dag_edges AS de
        JOIN end_to_end AS e2e ON e2e.expid = de.expid AND e2e.phase = de.phase AND e2e.source = de.source
        GROUP BY de.expid, de.phase
        ''')

    data = pd.DataFrame(rank_vs_loss.fetchall(), columns=['phase', 'loss', 'metric', 'rank'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    g = sns.pairplot(data, diag_kind='hist', kind='reg', hue='phase', vars=['loss', 'metric', 'rank'])


def plot_delay(db, args):

    data = db.execute(
        '''
        SELECT phase, delay
        FROM end_to_end
        WHERE source != 'm3-200'
        GROUP BY expid, phase
        ''')

    data = pd.DataFrame(data.fetchall(), columns=['phase', 'delay'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    for phase in range(1,7):
        pdata = data[data['phase'] == phase]
        ax = sns.distplot(pdata[['delay']], hist=False, label=phase_names[phase-1])

    ax.set_xlabel('delay')
    ax.legend()


def plot_jitter(db, args):

    data = db.execute(
        '''
        SELECT phase, jitter
        FROM end_to_end
        WHERE source != 'm3-200'
        GROUP BY expid, phase
        ''')

    data = pd.DataFrame(data.fetchall(), columns=['phase', 'jitter'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    for phase in range(1,7):
        pdata = data[data['phase'] == phase]
        ax = sns.distplot(pdata[['jitter']], hist=False, label=phase_names[phase-1])

    ax.set_xlabel('jitter')
    ax.legend()


def plot_loss(db, args):

    data = db.execute(
        '''
        SELECT phase, loss
        FROM end_to_end
        WHERE source != 'm3-200'
        GROUP BY expid, phase
        ''')

    data = pd.DataFrame(data.fetchall(), columns=['phase', 'loss'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    for phase in range(1,7):
        pdata = data[data['phase'] == phase]
        ax = sns.distplot(pdata[['loss']], hist=False, label=phase_names[phase-1])

    ax.set_xlabel('loss')
    ax.legend()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process file names.')

    subp = parser.add_subparsers(help='command')

    # overhead (dio,dao,dis) vs. phase
    overhead = subp.add_parser('overhead')
    overhead.set_defaults(func=plot_overhead)
    overhead.add_argument('--database', '-d', nargs=1)
    overhead.add_argument('--output'  , '-o', nargs=1)

    consumption = subp.add_parser('consumption')
    consumption.set_defaults(func=plot_consumption)
    consumption.add_argument('--database', '-d', nargs=1)
    consumption.add_argument('--output'  , '-o', nargs=1)

    loss = subp.add_parser('loss')
    loss.set_defaults(func=plot_loss)
    loss.add_argument('--database', '-d', nargs=1)
    loss.add_argument('--output', '-o', nargs=1)

    delay = subp.add_parser('delay')
    delay.set_defaults(func=plot_delay)
    delay.add_argument('--database', '-d', nargs=1)
    delay.add_argument('--output', '-o', nargs=1)

    jitter = subp.add_parser('jitter')
    jitter.set_defaults(func=plot_jitter)
    jitter.add_argument('--database', '-d', nargs=1)
    jitter.add_argument('--output', '-o', nargs=1)

    args = parser.parse_args()
    db = p.init_db(args.database[0])

    plt.figure()

    sns.set()
    sns.set(font='NexusSerifPro')
    sns.set_palette(tubspalette)

    args.func(db, args)

    plt.savefig(args.output[0])
