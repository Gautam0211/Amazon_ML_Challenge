"""Recall@k of a diagnostic candidate file built with block.py --qfrac (only sampled queries count)."""
import sys
import numpy as np, pyarrow.parquet as pq
from common import wpath
from metric import pair_key
fn, qfrac = sys.argv[1], float(sys.argv[2])
c = pq.read_table(wpath("train", fn))
g = pq.read_table(wpath("train", "gt.parquet"))
gs, gq, gr = (g[x].to_numpy() for x in ("s1_row", "q_src", "q_row"))
ins = (gr.astype(np.int64) * 2654435761 % 1000) < qfrac * 1000
ck = pair_key(c["s1_row"].to_numpy(), c["q_src"].to_numpy(), c["q_row"].to_numpy()); rk = c["brank"].to_numpy().astype(np.int32)
o = np.argsort(ck); ck, rk = ck[o], rk[o]
gk = pair_key(gs[ins], gq[ins], gr[ins]); pos = np.searchsorted(ck, gk).clip(0, len(ck) - 1); f = ck[pos] == gk
r = np.where(f, rk[pos], 999)
print(fn, "pairs/query", round(len(ck) / max(1, len(np.unique((c['q_src'].to_numpy().astype(np.int64) << 32) | c['q_row'].to_numpy()))), 2),
      " ".join(f"@{k}={(r < k).mean():.4f}" for k in (1, 3, 5, 10, 20)))
na = np.zeros(len(gk), bool)
for s in (2, 3):
    flag = pq.read_table(wpath("train", f"s{s}.parquet"), columns=["nonascii"])["nonascii"].to_numpy(zero_copy_only=False)
    m = gq[ins] == s
    na[m] = flag[gr[ins][m]]
print("   nonascii @10=%.4f  ascii @10=%.4f" % ((r[na] < 10).mean(), (r[~na] < 10).mean()))
