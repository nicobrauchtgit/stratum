"""Memory benchmark: does eliminating / fusing ops reduce peak memory?

Each NumericOp materializes a fresh array, so fewer ops => fewer live intermediates.
We measure (1) a deterministic intermediate count from the optimized DAG, and (2) peak
process RSS, running each variant in its OWN subprocess (ru_maxrss is monotonic per
process, so per-variant subprocesses give clean peaks). The shared input array cancels
out in the OFF-vs-ON comparison.

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/memory_bench.py
"""
import subprocess
import sys
import numpy as np
import stratum as st
from stratum.optimizer._optimize import optimize, OptConfig
from stratum.runtime._scheduler import SequentialScheduler

N_DEFAULT = 20_000_000  # 160 MB per float64 array -> intermediates are visible


def _pipelines(N):
    rng = np.random.default_rng(0)
    s = rng.random(N) + 0.5
    return {
        # elimination-heavy: many redundant elementwise ops (each = one intermediate)
        "redundant-identities": lambda: (lambda x: (x / 1 * 1 + 0 - 0)
                                          .skb.apply_func(np.negative).skb.apply_func(np.negative))(st.var("x", s)),
        # fusion: exp/sum(exp) chain vs single (stable) softmax op
        "softmax": lambda: (lambda e: e / e.skb.apply_func(np.sum))(st.var("x", s).skb.apply_func(np.exp)),
        # fusion: log(x+1) vs log1p
        "log1p": lambda: (st.var("x", s) + 1).skb.apply_func(np.log),
    }


def _n_ops(builder, on):
    dag = builder(); env = dag.skb.get_data()
    lin, *_ = optimize(dag, config=OptConfig(algebraic_rewrites=on), env=env)
    return len(lin)


def _peak_rss_mb():
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _child(name, on, N):
    builder = _pipelines(N)[name]
    dag = builder(); env = dag.skb.get_data()
    lin, sp, fl = optimize(dag, config=OptConfig(algebraic_rewrites=on), env=env)
    sched = SequentialScheduler(lin, sp, fl, False); sched.mode = "fit_transform"
    for node in lin:
        sched.process_op(node)
    _ = sched.pool.pin(lin[-1])
    print(f"RSS_MB {_peak_rss_mb():.1f}")


def main():
    N = N_DEFAULT
    print("=" * 66)
    print(f"MEMORY: peak RSS + intermediate count, OFF vs ON  (N={N:,})")
    print("=" * 66)
    names = list(_pipelines(N).keys())
    for name in names:
        b = _pipelines(N)[name]
        ops_off, ops_on = _n_ops(b, False), _n_ops(b, True)
        rss = {}
        for on in ("off", "on"):
            out = subprocess.run([sys.executable, __file__, "--child", name, on, str(N)],
                                 capture_output=True, text=True, env={**__import__("os").environ,
                                 "PYTHONPATH": __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__file__)))})
            line = [l for l in out.stdout.splitlines() if l.startswith("RSS_MB")]
            rss[on] = float(line[0].split()[1]) if line else float("nan")
        saved = rss["off"] - rss["on"]
        print(f"\n{name}")
        print(f"    ops: off {ops_off} -> on {ops_on}   (intermediates removed: {ops_off-ops_on})")
        print(f"    peak RSS: off {rss['off']:7.1f} MB   on {rss['on']:7.1f} MB   saved {saved:6.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3] == "on", int(sys.argv[4]))
    else:
        main()
