"""Analysis A4 (stage D5): how much validation data does each decoder need? Panel F.

    python -m decoders.a4_tuning_data [--workers 14]

Each replicate keeps a random fraction (FRACS) of its validation streams (nested: the 25 % subset
is inside the 50 % one; seed = the replicate's fold and seed). Everything a decoder learns or tunes
comes from that subset only, including the HMM / HSMM duration models:
  pass 1: the D2 reduced grid (none, mode, hmm, hsmm) on the subset -> index/a4/grid/<run>__f<frac>.csv;
  selection per configuration and fraction (max mean subset seg F1@25, tie-break frame macro F1);
  pass 2: crf and crf_fixs (initialized at the HMM selected on the same subset) and the refiner,
  trained on the subset; every decoder decoded once on test_wild -> index/a4/test/<run>__f<frac>.csv.gz.
The 100 % fraction repeats the procedure on all validation streams, so the three fractions are
directly comparable (it differs from D1 only by the reduced HMM/HSMM grid).
Intake scores are not recomputed (their thresholds also use the validation streams); the G3 rule is.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from decoders import d2_run as D2
from decoders import dcommon as D
from decoders import decoders as DE
from decoders.dcommon import P4, PP

OUT = D.INDEX / "a4"
FRACS = [0.25, 0.5, 1.0]


class SubRep(D.Rep):
    """A replicate whose validation set is a random fraction of its validation streams (nested fractions)."""
    def __init__(self, rep_dir, frac: float):
        super().__init__(rep_dir)
        sids = super().sids("val")
        seed = 1000 * int(self.info["fold"]) + int(self.info["seed"])
        keep = sorted(np.random.default_rng(seed).permutation(len(sids))[: max(4, int(round(frac * len(sids))))])
        p = self.load("val")
        self.probs["val"] = {sids[i]: p[sids[i]] for i in keep}
        self.y10_val = [self.frames[f"y10/{s}"].astype(np.int64) for s in self.sids("val")]
        self.dpar = P4.fit_log_normal(self.y10_val)


def tag(rep, frac):
    """File stem of a replicate and fraction."""
    return f"{D2.stem(rep)}__f{int(frac * 100):03d}"


def grid_job(args):
    """Reduced grid on the validation subset of one replicate and fraction."""
    rep, frac = args
    f = OUT / "grid" / f"{tag(rep, frac)}.csv"
    if f.exists():
        return
    R = SubRep(rep, frac)
    ys = R.truths("val")
    rows = []
    for p in D2.points(R.ov):
        q = D2.decode(R, "val", p)
        r = pd.DataFrame(D.session_rows(R, "val", q, D2.key(p)))
        rows.append({"decoder": p["decoder"].split("_")[0], "decoder_key": D2.key(p), **{k: p.get(k) for k in ("st", "md", "t", "w")},
                     "tp25": r.tp25.sum(), "fp25": r.fp25.sum(), "fn25": r.fn25.sum(), "frame_macro_f1": D2.frame_macro_f1(q, ys),
                     "n_val": len(ys)})
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").replace(f)


def select(reps) -> dict:
    """Best grid point per configuration, fraction and decoder on the validation subsets."""
    out = {}
    for frac in FRACS:
        df = pd.concat(pd.read_csv(OUT / "grid" / f"{tag(r, frac)}.csv").assign(config=str(r.parent.relative_to(D.C.RUNS)))
                       for r in reps)
        df["f1"] = 2 * df.tp25 / (2 * df.tp25 + df.fp25 + df.fn25).clip(lower=1)
        g = df.groupby(["config", "decoder", "decoder_key"])[["f1", "frame_macro_f1"]].mean().reset_index()
        for (cfg, dec), d in g.groupby(["config", "decoder"]):
            b = d.sort_values(["f1", "frame_macro_f1"], ascending=False).iloc[0]
            pt = df[(df.config == cfg) & (df.decoder_key == b.decoder_key)].iloc[0]
            out.setdefault(frac, {}).setdefault(cfg, {})[dec] = {
                "decoder_key": b.decoder_key, "params": {k: float(pt[k]) for k in ("st", "md", "t", "w") if pd.notna(pt[k])}}
    D.write_json(OUT / "selection.json", {str(k): v for k, v in out.items()})
    return out


def test_job(args):
    """Learned and selected decoders trained on the subset, decoded on test_wild."""
    rep, frac, sel = args
    f = OUT / "test" / f"{tag(rep, frac)}.csv.gz"
    if f.exists():
        return
    R = SubRep(rep, frac)
    seed = 1000 * int(R.info["fold"]) + int(R.info["seed"])
    pts = {dec: {"decoder": s["decoder_key"] if dec == "mode" else dec, "ov": R.ov, **s["params"]} for dec, s in sel.items()}
    hp = pts["hmm"]
    rows = []
    for dec, p in pts.items():
        rows += D.session_rows(R, D.TEST, D2.decode(R, D.TEST, p), dec)
    info = {}
    for name, var in (("crf", "base"), ("crf_fixs", "fixs")):
        m, tr = DE.train_crf(R, hp, False, seed, var)
        rows += D.session_rows(R, D.TEST, DE.crf_decode(R, D.TEST, hp, DE.crf_export(m)), name)
        info[name] = tr["steps"]
    m, tr = DE.train_refiner(R, hp["ov"], seed)
    rows += D.session_rows(R, D.TEST, DE.refiner_decode(m, R, D.TEST, hp["ov"]), "refiner")
    info["refiner"] = tr["steps"]
    meta = {"run_id": str(rep.relative_to(D.C.RUNS)), "experiment": rep.parts[-3][:2], "config_key": rep.parent.name,
            "role": PP.role(rep), "replicate": rep.name, "frac": frac, "n_val": len(R.sids("val")),
            **{f"steps_{k}": v for k, v in info.items()}}
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**meta, **r} for r in rows]).to_csv(f.with_suffix(".tmp"), index=False, compression="gzip")
    f.with_suffix(".tmp").replace(f)


def main():
    """Run analysis A4."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    reps = [r for r in PP.rep_dirs() if not r.name.startswith("fall")][: a.limit or None]
    jobs = [(r, f) for r in reps for f in FRACS]
    print(f"[{D.now()}] A4: {len(reps)} replicates x fractions {FRACS}", flush=True)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, _ in enumerate(pool.imap_unordered(grid_job, jobs), 1):
            if i % 30 == 0 or i == len(jobs):
                print(f"[{D.now()}] grid {i}/{len(jobs)}", flush=True)
        sel = select(reps)
        print(f"[{D.now()}] selection done", flush=True)
        tj = [(r, f, sel[f][str(r.parent.relative_to(D.C.RUNS))]) for r, f in jobs]
        for i, _ in enumerate(pool.imap_unordered(test_job, tj), 1):
            if i % 10 == 0 or i == len(tj):
                print(f"[{D.now()}] test {i}/{len(tj)}", flush=True)
    read = sorted({l.rsplit("/", 1)[-1] for l in D.READ_LOG.read_text().splitlines() if l})
    print(f"sets read: {read}", flush=True)


if __name__ == "__main__":
    main()
