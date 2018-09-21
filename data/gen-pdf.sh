../../tools/topology.py dag 1 600 106823 --database db.sqlite3
../../tools/topology.py stability -d db.sqlite3 -o stability.pdf
../../tools/topology.py routes -d db.sqlite3 -o routes.pdf
../../tools/topology.py convergence -d db.sqlite3 -o convergence.pdf
../../tools/topology.py changes -d db.sqlite3 -o changes.pdf
../../tools/consumption.py phases -d db.sqlite3 -o consumption-phases.pdf
../../tools/consumption.py nodes -d db.sqlite3 -o consumption-nodes.pdf
../../tools/consumption.py hosts -d db.sqlite3 -o consumption-hosts.pdf m3-157 m3-200 m3-123
../../tools/performance.py overhead --database db.sqlite3 -o performance-overhead.pdf
../../tools/performance.py loss --database db.sqlite3 -o performance-loss.pdf
../../tools/performance.py rank --database db.sqlite3 -o performance-rank.pdf
