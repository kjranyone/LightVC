"""D587: R の S 形の候補外 8 人の平均 R̄(q, tab) = mean_k R(q, tab, 記憶_k) を、記憶の代わりに学習する小さな記憶(層ごとに M_s 列の鍵と値)を持つ R の写し G で近似する(自前の R の圧縮 = VC の変換音声の蒸留ではない)。
G は R と問いの流れ(qin・qb)を共有し(凍結)、層ごとの鍵・値・qp・op・lb・出力部を学習する。製品の S = R(q, tab, 目標の記憶) − G(q, tab)。
データ = rr_feat の男声の問い(male_realx・male_par)× 実女声の目標の表(real_fem の参照から)・切り出し 2 s・教師 = 凍結 R の 8 人の平均(包絡 24 + c0 1)。
検証 = 学習に使わない男声話者(番号の末尾 5 人)の問いで R̄ の再現(1 − MSE / Var)。
    uv run python d587_rbar_distill.py --out ../results/d587_rbar --steps 6000
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rr_tag", default="diag_rr19L")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=400)
    ap.add_argument("--ms", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    dev = a.device
    import conv_c0 as C0
    from d208_rr_natural import anchors
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt", map_location="cpu").cpu().numpy()
    ck = torch.load(ROOT / "results" / a.rr_tag / "last.pt", map_location="cpu", weights_only=False)
    net = RRm.build(ck["cfg"]).to(dev).eval().requires_grad_(False)
    net.load_state_dict(ck["net"])
    st = RRm.Store()
    man = {it["stem"]: it for it in json.loads((RRm.FEAT / "real_fem/manifest.json").read_text())["items"]}

    def mem_in(p_: np.ndarray, env: np.ndarray) -> tuple:
        n_ = min(env.shape[1], 5000)
        rc0 = (env[0, :n_] - RRm.C0_SIL) / 10
        return p_[:, :n_].astype(np.float32), np.concatenate([env[1:25, :n_] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)

    amem = []
    anc = anchors(8)
    for A_ in anc:
        ds = [torch.load(RRm.FEAT / man[s_]["pt"], weights_only=False) for s_ in A_["ref_stems"] if s_ in man]
        rp, rv = mem_in(np.concatenate([RRm.dense(d["pk"], d["pv"]) for d in ds], 1), np.concatenate([d["env"].float().numpy() for d in ds], 1))
        with torch.no_grad():
            amem.append((net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)), rp.shape[1]))
    anc_t = {A_["target"] for A_ in anc}
    q_keys = sorted(k for k in st.f if k[0] in ("male_realx", "male_par"))
    spk_of = {}
    for k in q_keys:
        if k[0] == "male_par":
            spk_of[k] = k[1]
        else:
            spk_of[k] = None
    items = json.loads((ROOT / "data/realx/items_clone.json").read_text())
    for it in items:
        kk = ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC")
        if kk in spk_of:
            spk_of[kk] = it["spk"]
    q_keys = [k for k in q_keys if spk_of.get(k) in st.stats]
    spks = sorted({spk_of[k] for k in q_keys})
    val_spk = set(spks[-5:])
    tr_keys = [k for k in q_keys if spk_of[k] not in val_spk]
    va_keys = [k for k in q_keys if spk_of[k] in val_spk]
    tgts = sorted(A for A in st.real if A not in anc_t and len(st.real[A]) >= 4)
    print("queries train", len(tr_keys), "val", len(va_keys), "male spk", len(spks), "targets", len(tgts), flush=True)
    tabs: dict = {}

    def table(A: str) -> np.ndarray:
        if A not in tabs:
            rr = [st.f[("real_fem", A, x)] for x in st.real[A]]
            env = np.concatenate([r["env"] for r in rr], 1)[:, :5000]
            u = np.concatenate([r["u"] for r in rr])[:5000]
            tabs[A] = C0.table(env[1:25], u, RRm.K, Cb).astype(np.float32)
        return tabs[A]

    def sample(keys: list, rng: random.Random) -> tuple:
        k = rng.choice(keys)
        b = st.f[k]
        sB = st.stats[spk_of[k]]
        pB = RRm.dense(b["pk"], b["pv"])
        vB = b["f1"] > 0
        q = np.concatenate([pB, (b["e3"] - sB["e3"][:, None]) / 10, (b["c0f"] - sB["c0"])[None], np.where(vB, np.log(np.maximum(b["f1"], 1.0)) - sB["lf"], 0.0)[None], vB[None].astype(np.float32)], 0).astype(np.float32)
        tab = table(rng.choice(tgts)) @ pB
        n = q.shape[1]
        s0 = rng.randrange(0, max(1, n - a.crop))
        q, tab = q[:, s0:s0 + a.crop], tab[:, s0:s0 + a.crop]
        if q.shape[1] < a.crop:
            pad = a.crop - q.shape[1]
            q, tab = np.pad(q, ((0, 0), (0, pad)), mode="edge"), np.pad(tab, ((0, 0), (0, pad)), mode="edge")
        return q, tab.astype(np.float32)

    @torch.no_grad()
    def teacher(q: torch.Tensor, tab: torch.Tensor) -> tuple:
        es, ds = [], []
        for mm, m in amem:
            mem = [(kk.expand(q.shape[0], -1, -1, -1), vv.expand(q.shape[0], -1, -1, -1)) for kk, vv in mm] if isinstance(mm, list) else tuple(z.expand(q.shape[0], -1, -1, -1) for z in mm)
            e, d0, _ = net(q, tab, mem, torch.ones(q.shape[0], m, dtype=torch.bool, device=dev))
            es.append(e); ds.append(d0)
        return torch.stack(es).mean(0), torch.stack(ds).mean(0)

    g = copy.deepcopy(net).train()
    for n_, p_ in g.named_parameters():
        p_.requires_grad_(not (n_.startswith("qin") or n_.startswith("qb")))
    assert isinstance(amem[0][0], list), "R2(読み 2 層)を想定"
    L = len(amem[0][0])
    rng0 = np.random.default_rng(0)
    kv0 = []
    for l in range(L):
        ks = torch.cat([mm[l][0] for mm, _ in amem], -1)
        vs = torch.cat([mm[l][1] for mm, _ in amem], -1)
        idx = torch.from_numpy(rng0.choice(ks.shape[-1], a.ms, replace=False)).to(dev)
        kv0.append((torch.nn.Parameter(ks[..., idx].clone()), torch.nn.Parameter(vs[..., idx].clone())))
    kvp = torch.nn.ParameterList([p for kv in kv0 for p in kv])
    opt = torch.optim.AdamW([p for p in g.parameters() if p.requires_grad] + list(kvp), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def gmem(B: int) -> list:
        return [(kvp[2 * l].expand(B, -1, -1, -1), kvp[2 * l + 1].expand(B, -1, -1, -1)) for l in range(L)]

    def batch(keys: list, rng: random.Random) -> tuple:
        xs = [sample(keys, rng) for _ in range(a.bs)]
        return torch.from_numpy(np.stack([x[0] for x in xs])).to(dev), torch.from_numpy(np.stack([x[1] for x in xs])).to(dev)

    vrng = random.Random(123)
    vb = [batch(va_keys, vrng) for _ in range(8)]
    vt = [teacher(q, t) for q, t in vb]

    def val() -> dict:
        g.eval()
        se = sv = 0.0
        sc = svc = 0.0
        with torch.no_grad():
            for (q, t), (ee, dd) in zip(vb, vt):
                e, d0, _ = g(q, t, gmem(q.shape[0]), torch.ones(q.shape[0], a.ms, dtype=torch.bool, device=dev))
                r = ee - t
                se += float(((e - ee) ** 2).mean()); sv += float(((r - r.mean(-1, keepdim=True)) ** 2).mean())
                sc += float(((d0 - dd) ** 2).mean()); svc += float(((dd - dd.mean(-1, keepdim=True)) ** 2).mean())
        g.train()
        return {"env_r2": round(1 - se / sv, 4), "c0_r2": round(1 - sc / svc, 4)}

    print("val step 0", val(), flush=True)
    rng = random.Random(0)
    t0 = time.time()
    log = open(od / "train.jsonl", "a")
    for step in range(1, a.steps + 1):
        q, t = batch(tr_keys, rng)
        ee, dd = teacher(q, t)
        e, d0, _ = g(q, t, gmem(q.shape[0]), torch.ones(q.shape[0], a.ms, dtype=torch.bool, device=dev))
        loss = (e - ee).abs().mean() + (d0 - dd).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 200 == 0:
            rec = {"step": step, "loss": round(float(loss), 4), "min": round((time.time() - t0) / 60, 1)}
            if step % 1000 == 0:
                rec |= val()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
    torch.save({"g": g.state_dict(), "kv": [p.detach().cpu() for p in kvp], "cfg": ck["cfg"], "ms": a.ms, "rr_tag": a.rr_tag, "val": val()}, od / "g.pt")
    print("saved", od / "g.pt", val(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
