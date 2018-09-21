#!/usr/bin/env python

import sqlite3
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from rpl import analysis as an, data as dat
from matplotlib import colors, cm


tubspalette = ['#be1e3c', '#ffc82a', '#e16d00', '#711c2f', '#acc13a', '#6d8300', '#00534a', '#66b4d3', '#007a9b', '#003f57', '#8a307f', '#511246', '#4c1830']
phase_names = ['N', 'R', 'H', 'HR', 'HS', 'HSR']
versions = ['Contiki', 'Hardened', 'Hardened with UIDs']

def network_dag_evolution(db, phase, expid, resolution):
    """For each slot of length resolution yields a graph containing the last
    DAG during that time"""

    phase_start, phase_stop = db.execute('''
    SELECT tstart, tstop
    FROM phases
    WHERE phase = ? AND expid = ?''', (phase, expid)).fetchone()

    phaselen = int(phase_stop - phase_start)

    for t0 in range(int(phase_start), int(phase_stop), resolution):

        edges = db.execute('''
        WITH resolves AS (
        SELECT MAX(tchange) AS t, host
        FROM dag_evolution
        WHERE tchange < ? AND phase = ? AND expid = ?
        GROUP BY host
        )
        SELECT r.host, d.parent
        FROM dag_evolution AS d
        JOIN resolves AS r
        ON d.host = r.host AND r.t == d.tchange
        ''', (t0 + resolution, phase, expid))

        g = nx.DiGraph()

        for src, dest in edges:
            g.add_edge(src[3:], dest[3:])

        yield g


def draw_dag(g, output):
    nx.set_node_attributes(g, 'NexusSerifPro', name='fontname')
    nx.set_node_attributes(g, '#e0f0f6', name='fillcolor')
    nx.set_node_attributes(g, 'filled', name='style')
    nx.set_edge_attributes(g, 'open', name='arrowhead')

    a = nx.nx_agraph.to_agraph(g)
    a.draw(output, prog='dot')


def network_count_preferred(db):
    edge_weights = db.execute('''
    SELECT ps.phase, source, destination, AVG(count_preferred / (ps.tstop - ps.tstart))
    FROM dag_edges AS de
    JOIN phases AS ps
    ON ps.phase = de.phase AND ps.expid = de.expid
    AND count_preferred > 0
    GROUP BY source, destination, de.phase''')

    def _format():
        for p, s, d, av in edge_weights:
            yield phase_names[p-1], int(s[3:]), int(d[3:]), av

    return np.array(list(_format()), dtype=[('phase', 'U3'), ('source', 'd'), ('next hop', 'd'), ('preferred', 'f')])


def plot_routes_heatmap(phasesroutes_with_weight):

    def phase_heatmap(x, y, val, **kwargs):
        data = kwargs.pop('data')
        print(data)
        d = data.pivot(index=x, columns=y, values=val)
        print(d)
        sns.heatmap(d, **kwargs)

    df = pd.DataFrame(phasesroutes_with_weight)
    fgrid = sns.FacetGrid(df, col='phase', col_wrap=3, col_order=['N', 'H', 'HS', 'R', 'HR', 'HSR'])
    fgrid = fgrid.map_dataframe(phase_heatmap, 'source', 'next hop',
                                'preferred', annot=False, cbar=False, square=True,
                                cmap=sns.light_palette(tubspalette[0]))
    fgrid.set_xticklabels([])
    fgrid.set_yticklabels([])


def preferred_routes(db, args):

    routes = network_count_preferred(db)
    plot_routes_heatmap(routes)


def count_network_distinct_source(db, out):
    counts = db.execute('''
    select phase, COUNT(DISTINCT source), (phase - 1)/ 2 from dag_edges GROUP BY phase, expid ORDER BY phase
    ''')

    def filter_bad_ones():
        for phase, c, r in counts:
            if phase < 7:
                yield phase_names[(phase-1)], c, versions[int(r)]

    data = np.array(list(filter_bad_ones()), dtype=[('Phase', 'U3'), ('Participating Nodes', 'd'), ('Version', 'U20')])

    sns.swarmplot(y='Phase', x='Participating Nodes', hue='Version', data=pd.DataFrame(data))


def weighed_to_graphviz(g, root, cutoff=1):

    nx.set_node_attributes(g, 'NexusSerifPro', name='fontname')
    nx.set_node_attributes(g, '#aaaaff', name='fillcolor')
    nx.set_node_attributes(g, 'filled', name='style')
    nx.set_edge_attributes(g, 'open', name='arrowhead')

    #root = g.nodes(Data=True)[root]
    #root['fillcolor'] = '#ffaaaa'

    deletelist = []

    for x, y, data in g.edges(data=True):
        if data['weight'] < cutoff:
            deletelist.append((x, y))

    for x, y in deletelist:
        g.remove_edge(x, y)

    weights = list(nx.get_edge_attributes(g, 'weight').values())

    for x, y, data in g.edges(data=True):
        #data['xlabel'] = data['weight']
        data['penwidth'] = 10 * data['weight'] / max(weights)
        data['arrowhead'] = 'normal'
        data['dir'] = 'forward'

    return nx.nx_agraph.to_agraph(g)


#def draw_all_topologies(db, root, cutoff=1):
#    for phase, _, _, _ in dat.phases(db):
#        g = network_count_preferred(db, phase)
#        a = weighed_to_graphviz(g, root, cutoff)
#        a.draw('graph-%s.pdf' % phase, prog='dot')


def dag(db, args, fileformat='pdf'):

    step = 0

    for exp in args.experiments:

        for g in network_dag_evolution(db, args.phase, exp, args.resolution[0]):
            step += args.resolution[0]
            draw_dag(g, '%d-%d-%d.%s' % (args.phase, exp, step, fileformat))


def default_route_changes(db, args):

    count_changes = db.execute('''
    WITH change_src AS (
    SELECT expid, phase, source, COUNT(nexthop) cnt
    FROM default_route_changes
    GROUP BY expid, phase, source)
    SELECT phase, SUM(cnt)
    FROM change_src AS csr
    GROUP BY expid, phase''')

    def _format():
        for p, c in count_changes:
            yield phase_names[p-1], c, phase_names[int((p-1)/2) * 2], (p - 1) % 2 == 1

    data = pd.DataFrame(_format(), columns=['phase', 'changes', 'firmware', 'reset'])
    sns.barplot(x='firmware', y='changes', hue='reset', data=pd.DataFrame(data))


def dag_convergence_times(db):

    return db.execute('''
    WITH firstpp AS (
    SELECT ph.phase, ph.expid, host, MIN(tchange - ph.tstart) AS t
    FROM dag_evolution AS de
    JOIN phases AS ph
    ON de.phase = ph.phase AND de.expid = ph.expid
    GROUP BY ph.expid, ph.phase, host
    )
    SELECT phase, MAX(firstpp.t)
    FROM firstpp
    GROUP BY expid, phase''')

def plot_dag_convergence_times(db, args):

    convtimes = dag_convergence_times(db)

    def _format():
        for p, t in convtimes:
            yield phase_names[p-1], t, (p - 1) % 2 == 1

    data = np.array(list(_format()), dtype=[('phase', 'U3'), ('convergence time', 'f'), ('reset', bool)])

    print(data)

    sns.barplot(y='reset', x='convergence time', hue='phase', data=pd.DataFrame(data))


def plot_rank_changes(db, args):

    rank_vs_changes = db.execute(
        '''
        WITH s AS (
        SELECT expid, phase, source, COUNT(nexthop) AS stab
        FROM default_route_changes
        GROUP BY expid, phase, source)
        SELECT s.phase, avg_rank, avg_neighbors, s.stab
        FROM dag_nodes AS n
        JOIN s ON s.expid = n.expid AND s.phase = n.phase AND s.source = n.host
        ''')

    data = pd.DataFrame(rank_vs_changes.fetchall(), columns=['phase', 'rank', 'neighbors', 'changes'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    g = sns.pairplot(data, diag_kind='kde', kind='reg', hue='phase', vars=['rank', 'neighbors', 'changes'])
    print(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bla')


    subparsers = parser.add_subparsers(help='subcommand')

    dag_cmd_parser = subparsers.add_parser('dag')
    dag_cmd_parser.set_defaults(func=dag)
    dag_cmd_parser.add_argument('phase', type=int)
    dag_cmd_parser.add_argument('resolution', nargs=1, type=int)
    dag_cmd_parser.add_argument('experiments', nargs='+', type=int)
    dag_cmd_parser.add_argument('--database', '-d', nargs=1, help='database file')

    default_route_cmd = subparsers.add_parser('stability')
    default_route_cmd.set_defaults(func=default_route_changes)
    default_route_cmd.add_argument('--output', '-o', nargs=1, type=str)
    default_route_cmd.add_argument('--database', '-d', nargs=1, help='database file')

    route_selection_hm = subparsers.add_parser('routes')
    route_selection_hm.set_defaults(func=preferred_routes)
    route_selection_hm.add_argument('--output', '-o', nargs=1, type=str)
    route_selection_hm.add_argument('--database', '-d', nargs=1, help='database file')

    convergence_time = subparsers.add_parser('convergence')
    convergence_time.set_defaults(func=plot_dag_convergence_times)
    convergence_time.add_argument('--output', '-o', nargs=1, type=str)
    convergence_time.add_argument('--database', '-d', nargs=1, help='database file')

    rank_changes = subparsers.add_parser('changes')
    rank_changes.set_defaults(func=plot_rank_changes)
    rank_changes.add_argument('--database', '-d', nargs=1)
    rank_changes.add_argument('--output', '-o', nargs=1)

    args = parser.parse_args()
    db = dat.init_db(args.database[0])

    #draw_all_topologies(db, 'm3-157', 1)

    #count_network_distinct_source(db, 'distinct_sources.pdf')

    plt.figure()

    sns.set()
    sns.set(font='NexusSerifPro')
    sns.set_palette(tubspalette)

    args.func(db, args)

    if args.func in [plot_dag_convergence_times, preferred_routes, default_route_changes]:
        plt.savefig(args.output[0])
