#!/usr/bin/env python

import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from rpl import analysis, data


tubspalette = ['#be1e3c', '#ffc82a', '#e16d00', '#711c2f', '#acc13a', '#6d8300', '#00534a', '#66b4d3', '#007a9b', '#003f57', '#8a307f', '#511246', '#4c1830']
phase_names = ['N', 'R', 'H', 'HR', 'HS', 'HSR']
versions = ['Contiki', 'Hardened', 'Hardened with UIDs']
node_positions = {
    'm3-59': (0,0),
    'm3-57': (0,1),
    'm3-53': (0,3),
    'm3-51': (0,4),
    'm3-49': (0,5),
    'm3-47': (0,6),

    'm3-95': (1,0),
    'm3-93': (1,1),
    'm3-91': (1,2),
    'm3-89': (1,3),
    'm3-87': (1,4),
    'm3-85': (1,5),
    'm3-83': (1,6),

    'm3-133': (2,1),
    'm3-131': (2,2),
    'm3-127': (2,4),
    'm3-123': (2,6),

    'm3-161': (3,1),
    'm3-159': (3,2),
    'm3-157': (3,3),
    'm3-155': (3,4),
    'm3-153': (3,5),
    'm3-151': (3,6),

    'm3-204': (4,0),
    'm3-202': (4,1),
    'm3-200': (4,2),
    'm3-198': (4,3),
    'm3-196': (4,4),
    'm3-194': (4,5),
    'm3-192': (4,6),

    'm3-230': (5,0),
    'm3-228': (5,1),
    'm3-226': (5,2),
    'm3-224': (5,3),
    'm3-222': (5,4),
    'm3-220': (5,5),
    'm3-218': (5,6),

    'm3-256': (6,0),
    'm3-254': (6,1),
    'm3-252': (6,2),
    'm3-250': (6,3),
    'm3-248': (6,4),
    'm3-246': (6,5),
    'm3-244': (6,6)
}

def consumption_network_phases(db, phaselen, out):

    consumptions = db.execute('''SELECT phase, (phase - 1)/ 2, SUM(consumption) / ? FROM consumption GROUP BY phase, expid''', (phaselen,))

    data = np.array([(phase_names[phase-1], versions[int(reset)], cons) for phase, reset, cons in consumptions],
                    dtype=[('Phase', 'U3'), ('Version', 'U30'), ('Consumption', 'f')])

    sns.boxplot(x='Phase', y='Consumption', hue='Version', data=pd.DataFrame(data))


def consumption_node_phases(db, phaselen, out):

    consumptions = db.execute('''
    SELECT host, phase, consumption / ?
    FROM consumption
    ''', (phaselen,))

    data = np.array(
        [(host, phase_names[phase-1], cons) for host, phase, cons in consumptions],
        dtype=[('node', 'U6'), ('phase', 'U3'), ('consumption', np.float16)])

    sns.boxplot(x='phase', y='consumption', hue='node', data=pd.DataFrame(data))


def restart_energy_consumption(db, out):

    consumptions = db.execute('''
    SELECT resets.phase, resets.timestamp, SUM(consumption)
    FROM resets
    JOIN consumption
    ON resets.phase = consumption.phase
    AND resets.expid = consumption.expid
    GROUP BY consumption.phase, consumption.expid
    ''')

    data = np.array(
        [(versions[int((phase / 2) - 1)], reset, cons) for phase, reset, cons in consumptions.fetchall()], dtype=[('Phase', 'U20'), ('Reset Time', 'f'), ('Consumption', 'f')]
    )

    grid = sns.FacetGrid(pd.DataFrame(data), col='Phase')
    grid = grid.map(plt.scatter, 'Reset Time', 'Consumption')


def consumption_phases(db):

    # NOTE excludes sink node and resetting node
    consumptions = db.execute('''
    SELECT ps.phase,
    SUM(consumption),
    (ps.phase - 1) % 2 = 1
    FROM consumption AS c
    JOIN phases AS ps
    ON c.phase = ps.phase AND c.expid = ps.expid
    WHERE c.host != 'm3-200' AND c.host != 'm3-157'
    GROUP BY ps.expid, ps.phase''')

    def _format():
        for p, c, m in consumptions:
            yield phase_names[p-1], c, m

    data = np.array(list(_format()),
                    dtype=[('phase', 'U3'), ('consumption', 'f'), ('reset', bool)])

    return pd.DataFrame(data)


def plot_consumption_phases(db, args):

    data = consumption_phases(db)

    sns.boxplot(x='reset', y='consumption', hue='phase', hue_order=phase_names, data=data, palette=tubspalette)


def consumption_nodes(db):

    consumptions_nodes = db.execute('''
    SELECT p.phase, c.host, AVG(c.consumption)
    FROM consumption AS c
    JOIN phases AS p
    ON p.phase = c.phase AND p.expid = c.expid
    GROUP BY c.host, p.phase
    ''')

    def _format():
        for p, h, c in consumptions_nodes:
            host_x, host_y = node_positions[h]
            yield phase_names[p-1], int(h[3:]), host_x, host_y, c, ((p - 1) % 2 == 1)

    dtypes = [
        ('phase', 'U3'),
        ('host', 'd'),
        ('pos_x', 'd'),
        ('pos_y', 'd'),
        ('consumption', 'f'),
        ('reset', bool)
    ]

    data = np.array(list(_format()), dtype=dtypes)

    return pd.DataFrame(data)


def plot_nodes_consumption(db, args):

    def phase_heatmap(x, y, val, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=x, columns=y, values=val)
        print(d)
        hostnames = data.pivot(index=x, columns=y, values='host')
        hostnames.fillna(0)
        print(hostnames)
        ax = sns.heatmap(d, annot=hostnames, fmt='.0f', **kwargs)
        ax.invert_yaxis()
        #ax.invert_xaxis()

    data = consumption_nodes(db)
    fgrid = sns.FacetGrid(data, col='phase', col_wrap=3, col_order=['N', 'H', 'HS', 'R', 'HR', 'HSR'])
    fgrid = fgrid.map_dataframe(phase_heatmap, 'pos_y', 'pos_x',
                                'consumption', cbar=False, square=False,
                                cmap=sns.light_palette('#711c2f'))


def rank_neighbors_consumption(db):
    rnc = db.execute('''
    SELECT n.phase, n.host, n.avg_rank, n.avg_neighbors, c.consumption
    FROM dag_nodes AS n
    JOIN consumption AS c ON c.expid = n.expid AND c.phase = n.phase
    JOIN phases AS p ON p.expid = c.expid AND p.phase = c.phase
    GROUP BY n.expid, n.phase, n.host
    ''')

    def _format():
        for p, h, r, n, c in rnc:
            yield phase_names[p-1], h[3:], r, n, c

    data = np.array(list(_format()), dtype=[('phase', 'U3'), ('host', 'U10'),
                                            ('rank', 'f'), ('neighbors', 'f'),
                                            ('consumption', 'f')])

    return pd.DataFrame(data)


def plot_rank_neighbors_consumption(db, args):

    data = rank_neighbors_consumption(db)

    #sns.pairplot(data,
    #             hue='phase',
    #             kind='reg',
    #             diag_kind='hist',
    #             vars=['rank', 'consumption'],
    #             markers="+",
    #             palette=tubspalette)
    sns.lmplot(x="rank", y="consumption", hue="phase",
               truncate=True, data=data, markers='+')


def consumption_hosts(db, hosts):

    qms = '?' * len(hosts)
    query = 'SELECT c.phase, c.host, c.consumption FROM consumption AS c JOIN phases AS p ON p.phase = c.phase AND p.expid = c.expid WHERE c.host IN ({})'.format(','.join(qms))
    csh = db.execute(query, hosts)

    def _format():
        for p, h, c in csh:
            yield phase_names[p-1], h[3:], c

    data = np.array(list(_format()),
                    dtype=[('phase', 'U3'), ('host', 'U10'), ('consumption', 'f')])

    return pd.DataFrame(data)


def plot_consumption_hosts(db, args):

    hcons = consumption_hosts(db, args.hosts)
    sns.boxplot(x='phase', y='consumption', hue='host', data=hcons)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process file names.')
    subparsers = parser.add_subparsers(help='subcommand')

    phases_cmd = subparsers.add_parser('phases')
    phases_cmd.add_argument('--database', '-d', type=str, nargs=1, help='a sqlite3 database file', default=['db.sqlite3'])
    phases_cmd.add_argument('--out', '-o', type=str, nargs=1, help='output file for the plot')
    phases_cmd.set_defaults(func=plot_consumption_phases)

    nodes_cmd = subparsers.add_parser('nodes')
    nodes_cmd.add_argument('--database', '-d', type=str, nargs=1, help='a sqlite3 database file', default=['db.sqlite3'])
    nodes_cmd.add_argument('--out', '-o', type=str, nargs=1, help='output file for the plot')
    nodes_cmd.set_defaults(func=plot_nodes_consumption)

    rank_neighbors_cmd = subparsers.add_parser('regress')
    rank_neighbors_cmd.add_argument('--database', '-d', type=str, nargs=1, help='a sqlite3 database file', default=['db.sqlite3'])
    rank_neighbors_cmd.add_argument('--out', '-o', type=str, nargs=1, help='output file for the plot')
    rank_neighbors_cmd.set_defaults(func=plot_rank_neighbors_consumption)

    hosts_cmd = subparsers.add_parser('hosts')
    hosts_cmd.add_argument('--database', '-d', type=str, nargs=1, help='a sqlite3 database file', default=['db.sqlite3'])
    hosts_cmd.add_argument('--out', '-o', type=str, nargs=1, help='output file for the plot')
    hosts_cmd.add_argument('hosts', type=str, nargs='+', help='hosts to plot')
    hosts_cmd.set_defaults(func=plot_consumption_hosts)

    plt.figure()

    sns.set()
    sns.set(font='NexusSerifPro')
    sns.set_palette(tubspalette)

    args = parser.parse_args()
    db = data.init_db(args.database[0])
    args.func(db, args)

    plt.savefig(args.out[0])
