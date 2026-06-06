"""Minimal reproducer for gh-8658: argmap lazy-compile race on free-threaded Python.

Run on a free-threaded build (the GIL hides the race)::

    pixi run -e test-base-py314t python argmap_gh8658_repro.py

Unpatched, this prints stochastic failures such as
``NameError: name 'argmap__not_implemented_for_107' is not defined`` (the same
symptom reported from ``nx.Graph`` / ``nx.is_connected`` on an HPC cluster).
With the fix it prints OK.

Why it triggers: the first call to an argmap-decorated function compiles a
specialized version and swaps it into the wrapper. When many threads race that
first call, one could observe the new ``__code__`` before the mangled names it
references were published into ``__globals__`` -- or two compilations could
collide on a name. Every wrapper here shares a name but returns a distinct
value, so a mis-binding shows up as a wrong result, not only a crash.
"""

import sys
import threading

from networkx.utils import not_implemented_for

if hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled():
    print("warning: GIL is enabled; run with python3.14t to expose the race")

N = 64
errors = []


def make(seed):
    # @not_implemented_for is the argmap decorator seen in the gh-8658 traceback.
    @not_implemented_for("directed")
    def f(G, _seed=seed):
        return _seed

    return f


# Fresh wrappers => each one's first call exercises the lazy compile.
wrappers = [make(s) for s in range(N)]
barrier = threading.Barrier(N)
G = __import__("networkx").Graph()


def worker(start):
    barrier.wait()  # release all threads onto the cold wrappers at once
    try:
        for i in range(N):
            idx = (start + i) % N
            if wrappers[idx](G) != idx:
                raise AssertionError(f"wrong result from wrapper {idx}")
    except Exception as e:  # NameError or wrong-result AssertionError
        errors.append(repr(e))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()

if errors:
    print(f"FAILED: {len(errors)} thread(s) errored, e.g. {errors[0]}")
    sys.exit(1)
print("OK: no errors")
