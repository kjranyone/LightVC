"""参照を読む R(diag_rr19L・読み 2 層)と R̄ の近似 G(d587_rbar)を Rust(crates/lightvc-core/src/rr.rs)へ書き出す + 一致検査の固定の入力と期待出力。
製品の S 形 = R(問い, 表, 目標の記憶) − G(問い, 表)(包絡 24・c0 1)。問いの幹(qin・qb)は共有。
名前: qin・qb.i.a・qb.i.b(幹)/ r.qp.l・r.op.l・r.lb.l.a・r.lb.l.b・r.din・r.db.i.a・r.db.i.b・r.dout(R の枝)/ g.(同じ)(G の枝)/ g.k.l・g.v.l(G の学習した記憶 [頭, dk, M])(+ .w / .b)。
fixture/: q.f32 [nq, T]・tab.f32 [24, T]・mem_k{l}.f32・mem_v{l}.f32(目標の記憶 [頭, dk, M])・ds.f32 [24, T](R − G の包絡)・dc.f32 [T](R − G の c0)・meta.json。
    CUDA_VISIBLE_DEVICES= uv run python export_rr.py --out ../results/d587_rbar/export
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def conv_items(prefix: str, mod: torch.nn.Module) -> dict:
    return {f"{prefix}.w": mod.c.weight, f"{prefix}.b": mod.c.bias}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr_tag", default="diag_rr19L")
    ap.add_argument("--g", default=str(ROOT / "results/d587_rbar/g.pt"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    ck = torch.load(ROOT / "results" / a.rr_tag / "last.pt", map_location="cpu", weights_only=False)
    net = RRm.build(ck["cfg"]).eval()
    net.load_state_dict(ck["net"])
    sg = torch.load(a.g, map_location="cpu", weights_only=False)
    g = RRm.build(ck["cfg"]).eval()
    g.load_state_dict(sg["g"])
    kv = [z.float() for z in sg["kv"]]
    cfg = ck["cfg"]
    L = cfg["layers"]
    assert L > 1, "RR2(読み 2 層)を想定"
    ten: dict = {}
    ten |= conv_items("qin", net.qin)
    for i, b in enumerate(net.qb):
        ten |= conv_items(f"qb.{i}.a", b.a) | conv_items(f"qb.{i}.b", b.b)
    for pre, m in (("r", net), ("g", g)):
        for l in range(L):
            ten |= conv_items(f"{pre}.qp.{l}", m.qp[l]) | conv_items(f"{pre}.op.{l}", m.op[l])
            ten |= conv_items(f"{pre}.lb.{l}.a", m.lb[l].a) | conv_items(f"{pre}.lb.{l}.b", m.lb[l].b)
        ten |= conv_items(f"{pre}.din", m.din) | conv_items(f"{pre}.dout", m.dout)
        for i, b in enumerate(m.db):
            ten |= conv_items(f"{pre}.db.{i}.a", b.a) | conv_items(f"{pre}.db.{i}.b", b.b)
    for l in range(L):
        ten[f"g.k.{l}"] = kv[2 * l][0]
        ten[f"g.v.{l}"] = kv[2 * l + 1][0]
    ten |= conv_items("mem.min", net.min)
    for i, b in enumerate(net.mb):
        ten |= conv_items(f"mem.mb.{i}.a", b.a) | conv_items(f"mem.mb.{i}.b", b.b)
    for l in range(L):
        ten |= conv_items(f"mem.kp.{l}", net.kp[l]) | conv_items(f"mem.vp.{l}", net.vp[l])
    assert torch.equal(net.qin.c.weight, g.qin.c.weight), "G の幹は R と共有(凍結)のはず"
    ten = {k: v.detach().float().contiguous() for k, v in ten.items()}
    lb_d = [1, 2, 4, 1, 2, 4][:L]
    man = {"tensors": {}, "cfg": cfg | {"lb_dils": lb_d, "db_dils": [1, 2, 4]}, "const": {"C0_SIL": RRm.C0_SIL, "K": RRm.K, "ms": int(sg["ms"]), "rr_tag": a.rr_tag}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    import conv_c0 as C0
    from d208_rr_natural import anchors
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt", map_location="cpu").cpu().numpy()
    st = RRm.Store()
    items = json.loads((ROOT / "data/realx/items_clone.json").read_text())
    anc_t = {A_["target"] for A_ in anchors(8)}
    rng = random.Random(7)
    it = next(x for x in sorted(items, key=lambda z: (z["item"], z["ti"])) if ("male_realx", "wav", f"p{x['item']:04d}_t{x['ti']:03d}_SRC") in st.f and x["spk"] in st.stats)
    b = st.f[("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC")]
    sB = st.stats[it["spk"]]
    pB = RRm.dense(b["pk"], b["pv"])
    vB = b["f1"] > 0
    q = np.concatenate([pB, (b["e3"] - sB["e3"][:, None]) / 10, (b["c0f"] - sB["c0"])[None], np.where(vB, np.log(np.maximum(b["f1"], 1.0)) - sB["lf"], 0.0)[None], vB[None].astype(np.float32)], 0).astype(np.float32)
    A = rng.choice(sorted(x for x in st.real if x not in anc_t and len(st.real[x]) >= 4))
    rr = [st.f[("real_fem", A, x)] for x in st.real[A]]
    env = np.concatenate([r["env"] for r in rr], 1)[:, :5000]
    u = np.concatenate([r["u"] for r in rr])[:5000]
    rp = np.concatenate([RRm.dense(r["pk"], r["pv"]) for r in rr], 1)[:, :5000].astype(np.float32)
    T_A = C0.table(env[1:25], u, RRm.K, Cb).astype(np.float32)
    rc0 = (env[0] - RRm.C0_SIL) / 10
    rv = np.concatenate([env[1:25] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)
    tab = (T_A @ pB).astype(np.float32)
    with torch.no_grad():
        mem = net.memory(torch.from_numpy(rp)[None], torch.from_numpy(rv)[None])
        qt, tt = torch.from_numpy(q)[None], torch.from_numpy(tab)[None]
        eR, dR, _ = net(qt, tt, mem, torch.ones(1, rp.shape[1], dtype=torch.bool))
        eG, dG, _ = g(qt, tt, [(kv[2 * l], kv[2 * l + 1]) for l in range(L)], torch.ones(1, kv[0].shape[-1], dtype=torch.bool))
    fx = out / "fixture"
    q.astype("<f4").tofile(fx / "q.f32")
    rp.astype("<f4").tofile(fx / "rp.f32")
    rv.astype("<f4").tofile(fx / "rv.f32")
    tab.astype("<f4").tofile(fx / "tab.f32")
    for l in range(L):
        mem[l][0][0].numpy().astype("<f4").tofile(fx / f"mem_k{l}.f32")
        mem[l][1][0].numpy().astype("<f4").tofile(fx / f"mem_v{l}.f32")
    (eR - eG)[0].numpy().astype("<f4").tofile(fx / "ds.f32")
    (dR - dG)[0].numpy().astype("<f4").tofile(fx / "dc.f32")
    (fx / "meta.json").write_text(json.dumps({"T": int(q.shape[1]), "nq": int(q.shape[0]), "M": int(rp.shape[1]), "target": A, "query": f"p{it['item']:04d}_t{it['ti']:03d}_SRC"}))
    print("exported", out, "tensors", len(ten), "T", q.shape[1], "M", rp.shape[1], "ds rms", float((eR - eG).pow(2).mean().sqrt()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
