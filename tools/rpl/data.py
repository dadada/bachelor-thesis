import json
import sqlite3
import numpy as np
import pandas as pd

from os.path import basename
from datetime import time, datetime
from parse import parse, compile
from itertools import groupby


dio_parser = compile("{:f};{};DIO\n")
dao_parser = compile("{:f};{};DAO\n")
dis_parser = compile("{:f};{};DIS\n")

node_start_parser = compile("{:f};{};GO!\n")
event_parser = compile("{timestamp} :: {type} :: {text}\n")
consumption_parser = compile("{timestamp:f}\t{key:d}\t{id:d}\t{seconds}\t{subseconds}\t{power:f}\t{voltage:f}\t{current:f}\n")
power_parser = compile("{timestamp:f}\t{key:d}\t{id:d}\t{seconds}\t{subseconds}\t{power:f}\t{voltage}\t{current}\n")
radio_parser = compile("{timestamp:f}\t{key:d}\t{id:d}\t{seconds:d}\t{subseconds:d}\t{channel:d}\t{rssi:d}\n")
address_parser = compile("{timestamp:f};{host};ADDR;{address}\n")
neighbor_parser = compile("{timestamp:f};{host};NEIGHBOR;{address};{isrouter:d};{state:d}\n")
default_route_parser = compile("{timestamp:f};{host};DEFAULT;{address};{lifetime:d};{infinite:d}\n")
route_parser = compile("{timestamp:f};{host};ROUTE;{address};{nexthop};{lifetime:d};{dao_seqno_out:d};{dao_seqno_in:d}\n")
dag_parser = compile("{timestamp:f};{host};DAG;{mop};{ocp};{rank:d};{interval:d};{neighbor_count:d}\n")
parent_parser = compile("{timestamp:f};{host};PARENT;{address};{rank:d};{metric:d};{rank_via_parent:d};{freshness:d};{isfresh:d};{preferred:d};{last_tx:d}\n")
powertrace_parser = compile("{timestamp:f};{host};P;{cpu:d};{lpm:d};{tx:d};{tx_idle:d};{rx:d};{rx_idle:d}\n")
spowertrace_parser = compile("{timestamp:f};{host};SP;{channel:d};{inputs:d};{tx_in:d};{rx_in:d};{outputs:d};{tx_out:d};{rx_out:d}\n")
payload_parser = compile("{:f};{};DATA;{};{};{:d}\n")


def init_db(dbpath):
    db = sqlite3.connect(dbpath)

    db.execute('''
    CREATE TABLE IF NOT EXISTS phases (
    expid INTEGER,
    phase INTEGER,
    tstart FLOAT,
    tstop FLOAT,
    PRIMARY KEY (expid, phase)
    )
    ''')

    # assume static addresses...
    db.execute('''
    CREATE TABLE IF NOT EXISTS addresses (
    host TEXT,
    address TEXT,
    PRIMARY KEY (address)
    )''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS consumption (
    expid INTEGER,
    phase INTEGER,
    host TEXT,
    consumption FLOAT,
    FOREIGN KEY (expid, phase) REFERENCES phases(expid, phase),
    PRIMARY KEY (expid, phase, host)
    )
    ''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS default_route_changes (
    expid INTEGER,
    phase INTEGER,
    source TEXT,
    nexthop TEXT,
    tchange FLOAT,
    FOREIGN KEY (expid, phase) REFERENCES phases(expid, phase)
    FOREIGN KEY (source) REFERENCES addresses(address),
    FOREIGN KEY (nexthop) REFERENCES addresses(address),
    PRIMARY KEY (expid, phase, source, nexthop, tchange)
    )
    ''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS dag_nodes (
    expid INTEGER,
    phase INTEGER,
    host TEXT,
    mop BYTE,
    ocp BYTE,
    avg_rank FLOAT,
    avg_neighbors FLOAT,
    FOREIGN KEY (expid, phase) REFERENCES phases(expid, phase),
    PRIMARY KEY (expid, phase, host)
    )
    ''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS dag_edges (
    expid INTEGER,
    phase INTEGER,
    source TEXT,
    destination TEXT,
    avg_rank FLOAT,
    avg_metric FLOAT,
    count_preferred INTEGER,
    FOREIGN KEY (expid, phase) REFERENCES phases(expid, phase),
    FOREIGN KEY (expid, phase, source) REFERENCES dag_nodes(expid, phase, host),
    FOREIGN KEY (expid, phase, destination) REFERENCES dag_nodes(expid, phase, host),
    PRIMARY KEY (expid, phase, source, destination)
    )
    ''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS dag_evolution (
    expid INTEGER,
    phase INTEGER,
    tchange FLOAT,
    host TEXT,
    parent TEXT,
    FOREIGN KEY (expid, phase, host, parent) REFERENCES dag_edges(expid, phase, source, destination),
    PRIMARY KEY (expid, phase, tchange, host)
    )
    ''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS end_to_end (
    expid INTEGER,
    phase INTEGER,
    source TEXT,
    delay FLOAT,
    jitter FLOAT,
    loss FLOAT,
    FOREIGN KEY (expid, phase) REFERENCES phases(expid, phase),
    FOREIGN KEY (expid, phase, source) REFERENCES dag_nodes(expid, phase, host),
    PRIMARY KEY (expid, phase, source)
    )''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS overhead (
    expid INTEGER,
    phase INTEGER,
    source INTEGER,
    dios INTEGER,
    daos INTEGER,
    dis INTEGER,
    FOREIGN KEY (expid, phase) REFERENCES phases(expid, phase),
    FOREIGN KEY (expid, phase, source) REFERENCES dag_nodes(expid, phase, host),
    PRIMARY KEY (expid, phase, source))
    ''')

    db.execute('''
    CREATE TABLE IF NOT EXISTS resets (
    expid INTEGER,
    phase INTEGER,
    host TEXT,
    timestamp FLOAT,
    FOREIGN KEY (expid, phase) REFERENCES phases(expid, phase),
    PRIMARY KEY (expid, phase)
    )
    ''')

    return db


def find_phase(phases, timestamp):
    for name, start, stop in phases:
        if start <= timestamp and timestamp < stop:
            return name


def __process_consumptions(phases, consumptions):

    def _format():
        for line in consumptions:
            timestamp = float(line['seconds'] + '.' + line['subseconds'])
            yield timestamp, line['power']

    consum = np.array(list(_format()), dtype=[('timestamp', 'f'), ('power', 'f')])
    data = pd.pivot_table(pd.DataFrame(consum), values='power', index='timestamp')

    for name, start, stop in phases:
        vals = data.loc[start:stop]
        mean = vals['power'].mean()
        yield name, float(mean)


def __store_consumption(db, expid, host, consumptions):

    def _format():
        for phase, consumption in consumptions:
            yield expid, phase, host, consumption

    db.executemany(
        '''INSERT OR REPLACE INTO consumption VALUES (?,?,?,?)''',
        _format()
    )


def __run_parser(log, parser):

    for line in log:
        res = parser.parse(line)

        # skip invalid lines
        if not res:
            continue
        else:
            yield res.named


def __process_addresses(db, expid, addresses):

    def _format():
        for addr in addresses:
            yield addr['host'], addr['address']

    db.executemany('''INSERT OR REPLACE INTO addresses VALUES (?,?)''', _format())


def __process_phases(db, expid, phases):

    def _format():
        i = 1
        for start, stop in phases:
            yield expid, i, float(start), float(stop)
            i += 1

    db.executemany('''INSERT OR REPLACE INTO phases VALUES (?,?,?,?)''', _format())


def __parse_weird_iso(something):
    return datetime.strptime(something, '%Y-%m-%d %H:%M:%S,%f').timestamp()

def __phases_from_events(logs, phase_len=600):

    flash = 'Flash firmware on open node'
    pstop = 'Open power stop'
    pstart = 'Open power start'
    phases = pd.DataFrame()

    def parse_node_log(log):
        for event in __run_parser(log, event_parser):
            if event['type'] == 'INFO':
                timestamp = __parse_weird_iso(event['timestamp'])
                if flash in event['text'] or pstart in event['text']:
                    yield timestamp

    for log in logs:
        phases.reset_index(drop=True)
        ts = pd.DataFrame(parse_node_log(log), columns=[log.name]).reset_index(drop=True)
        phases = phases.reset_index(drop=True)
        phases[log.name] = ts

    phases['min'] = phases.min(axis=1)

    for m in phases['min']:
        yield m, m + phase_len


def parse_events(db, expid, logs):

    __process_phases(db, expid, __phases_from_events(logs))


def parse_consumption(db, expid, phases, host, log):

    __store_consumption(db, expid, host, __process_consumptions(phases, __run_parser(log, power_parser)))


def parse_addresses(db, expid, log):

    __process_addresses(db, expid, __run_parser(log, address_parser))


def __process_dag(phases, dags):

    ranks, neighbors = dict(), dict()
    phase = 0

    for p in dags:
        timestamp, host, rank, neighbor_c = p['timestamp'], p['host'], p['rank'], p['neighbor_count']

        name, _, stop = phases[phase]

        if timestamp > stop:
            for host in ranks:
                yield phase+1, host, p['mop'], p['ocp'], np.average(ranks[host]), np.average(neighbors[host])

            ranks, neighbors = dict(), dict()
            phase += 1

        # check if no next phase
        if phase >= len(phases):
            break

        name, start, stop = phases[phase]

        # init lists
        if not host in ranks:
            ranks[host] = []
            neighbors[host] = []

        # check if within phase
        if start <= timestamp and stop > timestamp:
            ranks[host].append(rank)
            neighbors[host].append(neighbor_c)

    for host in ranks:
        yield phase+1, host, p['mop'], p['ocp'], np.average(ranks[host]), np.average(neighbors[host])


def __store_dag_nodes(db, expid, dag_nodes):

    def _format():
        for n in dag_nodes:
            yield (expid, ) + n

    db.executemany('''INSERT OR REPLACE INTO dag_nodes VALUES (?,?,?,?,?,?,?)''', _format())


def parse_dag(db, expid, phases, log):

    __store_dag_nodes(db, expid, __process_dag(phases, __run_parser(log, dag_parser)))


def __process_parents(phases, parents):

    ranks, metrics, prefs = dict(), dict(), dict()
    phase = 0

    for p in parents:
        timestamp, source, dest, rank, metric, rvp, pref = p['timestamp'], p['host'], p['address'], p['rank'], p['metric'], p['rank_via_parent'], p['preferred']

        key = (source, dest)

        name, _, stop = phases[phase]

        if timestamp > stop:
            for s, d in ranks:
                yield name, s, d, np.average(ranks[(s, d)]), np.average(metrics[(s, d)]), prefs[(s, d)]

            ranks, metrics, prefs = dict(), dict(), dict()
            phase += 1

        if not key in ranks:
            ranks[key] = []
            metrics[key] = []
            prefs[key] = 0

        ranks[key].append(rank)
        metrics[key].append(metric)
        prefs[key] += pref

        if phase == len(phases):
            break

    for s, d in ranks:
        yield phase+1, s, d, np.average(ranks[(s, d)]), np.average(metrics[(s, d)]), prefs[(s, d)]

def __store_dag_edges(db, expid, dag_edges):

    def _format():
        for phase, host, dest, rank, metric, pref in dag_edges:
            yield expid, phase, host, rank, metric, pref, dest

    db.executemany('''
    INSERT OR REPLACE INTO dag_edges
    SELECT ?, ?, ?, host, ?, ?, ?
    FROM addresses
    WHERE address LIKE ?
    ''', _format())


def __process_dag_evolution(phases, parents):

    prev_pref_parent = dict()
    phase = 0

    for p in parents:

        def parent_changed():
            if p['host'] in prev_pref_parent and prev_pref_parent[p['host']] == p['address']:
                return False
            else:
                return p['preferred']

        phasename, pstart, pstop = phases[phase]

        if p['timestamp'] > pstop:
            phase += 1
            prev_pref_parent = dict()

            if phase >= len(phases):
                break

        phasename, pstart, pstop = phases[phase]

        if p['timestamp'] < pstop and p['timestamp'] >= pstart:
            # check if in phase, e.g. not between phases
            if parent_changed():
                yield phasename, p['timestamp'], p['host'], p['address']
                prev_pref_parent[p['host']] = p['address']


def __store_dag_evolution(db, expid, evolution):

    def _format():
        for phase, ts, s, d in evolution:
            yield expid, phase, ts, s, d

    db.executemany('''
    INSERT OR REPLACE INTO dag_evolution
    SELECT ?,?,?,?,host
    FROM addresses
    WHERE address LIKE ?''', _format())


def parse_parents(db, expid, phases, log):

    __store_dag_edges(db, expid, __process_parents(phases, __run_parser(log, parent_parser)))

    log.seek(0)
    __store_dag_evolution(db, expid, __process_dag_evolution(phases, __run_parser(log, parent_parser)))


def __process_resets(events):

    second_restart = False

    # filter everyy second restart of m3-200 as "reset"
    for ev in events:
        if 'Open power start' in ev['text']:
            if second_restart:
                second_restart = False
                yield __parse_weird_iso(ev['timestamp'])
            else:
                second_restart = True

def __store_resets(db, expid, host, timestamps):

    def _format():
        for stamp in timestamps:
            yield expid, host, stamp, stamp, stamp, expid

    db.executemany('''
    INSERT OR REPLACE INTO resets
    SELECT ?, phase, ?, ?
    FROM phases
    WHERE tstart < (?)
    AND (?) < tstop
    AND expid = ?
    ''', _format())

def parse_resets(db, expid, host, log):

    __store_resets(db, expid, host, __process_resets(__run_parser(log, event_parser)))


def __process_default_route_changes(phases, default_routes):

    phase = 0
    phase_last_def_rts = dict()

    for rt in default_routes:

        phase_name, _, pstop = phases[phase]

        if rt['timestamp'] > pstop:
            phase += 1
            phase_last_def_rts = dict()
            if phase >= len(phases):
                break

        def route_changed():
            host = rt['host']
            if not host in phase_last_def_rts:
                return True
            else:
                return rt['address'] != phase_last_def_rts[host]

        if route_changed():
            yield phase_name, rt['timestamp'], rt['host'], rt['address'], rt['lifetime'], rt['infinite']
            phase_last_def_rts[rt['host']] = rt['address']


def __store_default_route_changes(db, expid, droutes):

    def _format():
        for p, ts, h, addr, lt, i in droutes:
            yield expid, p, h, ts, addr

    db.executemany('''
    INSERT OR REPLACE INTO default_route_changes
    SELECT ?, ?, ?, host, ?
    FROM addresses
    WHERE address = ?''', _format())


def parse_default_routes(db, expid, phases, log):

    __store_default_route_changes(db, expid, __process_default_route_changes(phases, __run_parser(log, default_route_parser)))

#def parse_routes(db, expid, log):
#
#    __store_routes(db, expid, __process_routes(__run_parser(log, route_parser)))


#def __process_payloads(phases, payloads):
#
#    # timestamp, seqnr, losses
#    sends = dict()
#    recvs = dict()
#
#    for pl in payloads:
#        src = pl['src']
#        mtype = pl['type']
#        seqnr = pl['seqnr']
#        timestamp = pl['timestamp']
#
#        if not sends[src]:
#            sends[src] = []
#        if not recvs[src]:


#def parse_payloads(db, expid, phases, log):
#
#    __store_end_to_end(db, expid, __process_payloads(phases, __run_parser(log, payload_parser)))



#def __process_powertrace(phases, traces):
#
#    energest = dict()
#    phase = 0
#
#    for trace in traces:
#        newphase = find_phase(phases, parent['timestamp'])
#        if newphase != phase:
#            finalized = energest
#            energest = dict()
#
#            for host in finalized:
#                data = finalized[host]
#                yield phase, host, data['cpu'], np.average(data['ranks']), np.average(data['metrics']), data['preferred']
#            phase = newphase
#
#        host = parent['host']
#        timestamp = parent['timestamp']
#
#        if not host in hosts:
#            hosts[host] = {
#                'destination': parent['address'],
#                'ranks': [],
#                'metrics': [],
#                'preferred': 0
#            }
#
#        hosts[host]['ranks'] += [parent['rank']]
#        hosts[host]['metrics'] += [parent['metric']]
#        hosts[host]['preferred'] += parent['preferred']

#def parse_powertrace(db, expid, host, log):
#
#    __store_powertrace(db, expid, __process_powertrace(phases, __run_parser(log, powertrace_parser)))

def parse_count_messages(db, expid, phases, serial):

    def __parse_messages():

        for line in serial:
            res = dio_parser.parse(line)
            if res:
                ts, host = res.fixed
                yield 'dio', ts, host
            res = dao_parser.parse(line)
            if res:
                ts, host = res.fixed
                yield 'dao', ts, host
            res = dio_parser.parse(line)
            if res:
                ts, host = res.fixed
                yield 'dis', ts, host

    data = pd.DataFrame(__parse_messages(), columns=['type', 'ts', 'host'])

    def __count_messages():
        for phase, start, stop in phases:
            msgs = data[(start <= data['ts']) & (data['ts'] < stop)]
            msgs = msgs.groupby([msgs['host'], msgs['type']]).size()

            for host, group in msgs.groupby('host'):
                yield expid, phase, host, int(group.get((host,'dio'), default=0)), int(group.get((host,'dao'), default=0)), int(group.get((host,'dis'), default=0))

    db.executemany(
        '''INSERT OR REPLACE INTO overhead VALUES (?,?,?,?,?,?)''',
        __count_messages()
    )


def parse_end_to_end(db, expid, phases, addresses, serial):

    def __parse_messages():

        for line in serial:
            res = payload_parser.parse(line)
            if res:
                yield res.fixed

    e2e = pd.DataFrame(__parse_messages(), columns=['timestamp', 'host', 'type', 'address', 'seqnr'])

    # loss, delay, jitter
    def __process_delay():
        for phasename, start, stop in phases:
            pe2e = e2e[(start <= e2e['timestamp']) & (e2e['timestamp'] < stop)]

            send = pe2e[pe2e['type'] == 'send']
            send = send.set_index(['host', 'seqnr'])

            recv = pe2e[pe2e['type'] == 'recv']
            recv = recv.join(addresses, lsuffix='_dest', on=['address'], how='inner')
            recv = recv.set_index(['host', 'seqnr'])

            pe2e = send.join(recv, rsuffix='_arrived', sort=False)

            pe2e['delay'] = pe2e['timestamp_arrived'] - pe2e['timestamp']

            for host, group in pe2e.groupby('host'):
                delays = group['delay']
                delay = delays.mean()
                jitter = delays.var()
                if delays.count() == 0:
                    print(host)
                loss = delays.isnull().sum() / delays.count()

                yield expid, phasename, host, delay, jitter, loss

    db.executemany('''INSERT OR REPLACE INTO end_to_end VALUES (?,?,?,?,?,?)''', __process_delay())


def parse_contiki_starts(db, expid, serial, exclude=['m3-200', 'm3-157'], cphases=6, phaselen=600):

    def read_phases_from_log():
        for line in serial:
            res = node_start_parser.parse(line)
            if res:
                yield res.fixed

    starts = pd.DataFrame(np.array(list(read_phases_from_log()), dtype=[('timestamp', 'f'), ('host', 'U10')]))
    phases = pd.DataFrame()

    for name, group in starts.groupby('host'):
        if not name in exclude:
            phases[name] = group['timestamp'].reset_index(drop=True).sort_values()

    phases['min'] = phases.min(axis=1)
    phases['max'] = phases.max(axis=1)
    phases['diff'] = phases['max'] - phases['min']
    print(phases.loc[:,'min':'diff'])

    def _format():
        phase = 0
        for t in phases['min']:
            phase += 1
            yield expid, phase, t

    #db.executemany(''' INSERT OR REPLACE INTO phases VALUES(?,?,?,?) ''', _format())



def phases(db):
    return db.execute('''
    SELECT phase, expid, tstart, tstop
    FROM phases
    ORDER BY phase, expid''')
