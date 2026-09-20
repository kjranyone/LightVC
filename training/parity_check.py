"""5.1 の parity / 因果性 / 未来不変性（ネット込み）。

`ship_check.py` は front-end 3 段しか見ず **ckpt を読まない**ので、ネットが
発話全体統計を持っていても素通りする（`rddsp_hf.Wavehax2D` の `GroupNorm(1, ch)` が
`[B,C,F,T]` の T 込みで正規化していた事故が 22 巡素通りした）。ここが唯一の関門。

    uv run python parity_check.py --ckpt ../results/<TAG>/<C>.pt
    uv run python parity_check.py --arch-only     # ckpt 無しで構成だけ検査（2.3 (1)）

⚠ 白色雑音で測らない（argmax/閾値の離散性で偽 PASS が出る）。**実音声プローブ**を使う。
⚠ 編集が出力を動かさない場合は INCONCLUSIVE ＝ PASS にしない。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import torch

import ship_front as SF
import v1d

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREP = ROOT / "data/full_prep"
MIN_PROBES = 16


def _probes(n: int = MIN_PROBES) -> list[torch.Tensor]:
    """実音声プローブ。**白色雑音で代用しない。**"""
    out = []
    for f in sorted(PREP.glob("sh_*.pt")):
        # shard は dict のリスト（キー w / mel / f0 / spk）。話者をばらけさせる
        # ため 1 shard から 4 本までにする。
        items = torch.load(f, map_location="cpu")
        for it in items[:4]:
            w = it["w"]
            if torch.is_tensor(w) and w.numel() >= SF.NFFT_A * 8:
                out.append(w.flatten()[: SF.HOP_S * 200].float())
            if len(out) >= n:
                return out
        if len(out) >= n:
            break
    return out


def _inputs(w: torch.Tensor, net) -> tuple[torch.Tensor, torch.Tensor]:
    mel = SF.mel(w)
    T = min(SF.n_frames(w.shape[-1]), mel.shape[-1] * SF.HOP_A // SF.HOP_S)
    mel_syn = SF.to_frames(mel, T)
    P = torch.randn(net.nbin, T) * 0.1 + 1j * torch.randn(net.nbin, T) * 0.1
    return mel_syn, P


def future_invariance(net, probes) -> dict:
    """入力の t 以降を書き換え、t 以前の出力が動かないことを見る。

    戻り値の `unbounded` は「頭まで動いた」＝発話全体統計、
    `inconclusive` は「編集しても出力が 1 度も動かなかった」＝検査になっていない。
    """
    worst_ms = 0.0
    unbounded = False
    moved = False
    for w in probes:
        mel, P = _inputs(w, net)
        T = mel.shape[-1]
        with torch.no_grad():
            base = net.residual(mel, P)
        for frac in (0.35, 0.5, 0.65, 0.8, 0.95):
            c = max(1, int(T * frac))
            # ⚠ **mel だけを編集する**。P も同時に動かすと S = P + o の素通り分で
            #    必ず動くので、`inconclusive` が常に False になり検査が空になる。
            m2, P2 = mel.clone(), P
            m2[:, c:] += 5.0
            with torch.no_grad():
                alt = net.residual(m2, P2)
            d = (alt - base).abs()
            hit = (d > 1e-5).any(dim=0).nonzero()
            if hit.numel() == 0:
                continue
            moved = True
            first = int(hit[0])
            if first == 0:
                unbounded = True
            la = max(0, c - first) * SF.HOP_S / 44100.0 * 1000.0
            worst_ms = max(worst_ms, la)
    return {"lookahead_ms": round(worst_ms, 4), "unbounded": unbounded,
            "inconclusive": not moved, "probes": len(probes)}


def streaming_parity(net, probes) -> float:
    """streaming ≡ offline の SNR（合格線 80 dB）。"""
    worst = math.inf
    for w in probes:
        mel, P = _inputs(w, net)
        with torch.no_grad():
            full = net.residual(mel, P)
        st = v1d.V1DStream(net)
        outs = [st.step(mel[:, i:i + 2], P[:, i:i + 2])
                for i in range(0, mel.shape[-1] - 1, 2)]
        S = torch.cat(outs, dim=-1)
        m = min(S.shape[-1], full.shape[-1])
        err = (S[:, :m] - full[:, :m]).abs().pow(2).sum().item()
        sig = full[:, :m].abs().pow(2).sum().item()
        worst = min(worst, 10 * math.log10(sig / max(err, 1e-30)))
    return round(worst, 2)


def v2f_main(a) -> int:
    """v2f の未来不変性（ネット込み）。実音声プローブ・mel だけ編集。

    証跡のキーは 4.1 の関門が読む 10 個に合わせる（v2f の対応:
    dim→ch / L→layers / k_in→kf / k→kt / cin→入力面数 4）。
    """
    import v2f as V2
    import train_gvoc as TG
    from rddsp_gpu import mel_to_linear
    net = (V2.V2F(cin=4, ch=a.dim, layers=a.layers, kf=a.k_in, kt=a.k,
                  norm="cummean")
           if not a.ckpt else None)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu")
        ar = ck["args"]
        net = V2.V2F(cin=4, ch=ar["ch"], layers=ar["L"], kf=ar.get("kf", 7),
                     kt=ar.get("kt", 3), norm=ar.get("norm", "cummean"))
        net.load_state_dict(ck["net"])
    else:
        torch.manual_seed(0)
        net.out.reset_parameters()
    net.eval()
    probes = _probes(a.probes)
    if len(probes) < MIN_PROBES:
        print(f"  ⚠ 実音声プローブが {len(probes)} 本（{MIN_PROBES} 要求）")
        return 1
    TG.SHIP = True
    W = mel_to_linear("cpu", nbin=SF.NFFT_S // 2 + 1)
    worst = 0.0
    unbounded = False
    moved = False
    g = torch.Generator().manual_seed(0)
    for w in probes:
        T = SF.n_frames(w.shape[-1])
        f0, _ = SF.causal_f0(w)
        mel = SF.to_frames(W @ SF.mel(w), T)
        P = SF.nhv_spec(mel, f0, w.shape[-1], g, T=T)
        m = min(T, P.shape[-1])
        feat = torch.cat([mel[None, :, :m], P.real[None], P.imag[None],
                          torch.log(P.abs()[None] + 1e-5)], 0)[None]
        with torch.no_grad():
            base = net(feat)
        for frac in (0.35, 0.5, 0.65, 0.8, 0.95):
            c = max(1, int(m * frac))
            f2 = feat.clone()
            f2[:, 0, :, c:] += 5.0
            with torch.no_grad():
                alt = net(f2)
            d = (alt - base).abs()
            hit = (d > 1e-5).flatten(0, 2).any(dim=0).nonzero()
            if hit.numel() == 0:
                continue
            moved = True
            first = int(hit[0])
            if first == 0:
                unbounded = True
            worst = max(worst, max(0, c - first) * SF.HOP_S / 44100.0 * 1000.0)
    rec = {"lookahead_ms": round(worst, 4), "unbounded": unbounded,
           "inconclusive": not moved, "probes": len(probes),
           "dim": net.ch, "L": net.layers, "k_in": net.kf, "k": net.kt,
           "nbin": SF.NFFT_S // 2 + 1, "cin": 4, "prior": True,
           "arch": "v2f", "ckpt": a.ckpt or "(arch-only)"}
    cfg = (f"d{net.ch}_L{net.layers}_ki{net.kf}_k{net.kt}"
           f"_nb{SF.NFFT_S // 2 + 1}_cin4_p1")
    out = ROOT / "results/z0" / f"future_inv_net_{cfg}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    ok = worst == 0.0 and not unbounded and not moved is False or True
    ok = (worst == 0.0 and not unbounded and moved)
    print(f"  未来不変性 {worst} ms / unbounded={unbounded} / inconclusive={not moved}")
    print(f"  -> {out.name}   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--arch", type=str, default="v1d", choices=["v1d", "v2f"])
    ap.add_argument("--arch-only", action="store_true")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--k-in", type=int, default=7)
    ap.add_argument("--probes", type=int, default=MIN_PROBES)
    a = ap.parse_args()

    if a.arch == "v2f":
        return v2f_main(a)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu")
        ar = ck.get("args", {})
        net = v1d.V1D(nbin=ar.get("nbin", SF.NFFT_S // 2 + 1),
                      dim=ar.get("dim", a.dim), L=ar.get("L", a.layers),
                      k_in=ar.get("k_in", a.k_in), k=ar.get("k", a.k))
        net.load_state_dict(ck["net"])
    else:
        net = v1d.V1D(dim=a.dim, L=a.layers, k_in=a.k_in, k=a.k)
        # ⚠ `out` はゼロ初期化なので、そのままだと S = P（prior の素通り）になり、
        #    「動いた」のは prior の編集ぶんだけ＝**ネットを一度も試さない空検査**。
        #    構成だけを見る用途では重みを撹拌してから測る。
        torch.manual_seed(0)
        torch.nn.init.normal_(net.out.weight, 0, 0.02)
        torch.nn.init.normal_(net.out.bias, 0, 0.02)
    net.eval()

    probes = _probes(a.probes)
    if len(probes) < MIN_PROBES:
        print(f"  ⚠ 実音声プローブが {len(probes)} 本しか取れない"
              f"（{MIN_PROBES} 本を要求。白色雑音で代用しない）")
        return 1

    fi = future_invariance(net, probes)
    snr = streaming_parity(net, probes)
    rec = dict(fi)
    rec.update({"dim": net.dim, "L": net.layers, "k_in": net.k_in, "k": net.k,
                "nbin": net.nbin, "cin": net.cin, "prior": True,
                "streaming_snr_db": snr,
                "ckpt": a.ckpt or "(arch-only)"})
    cfg = (f"d{net.dim}_L{net.layers}_ki{net.k_in}_k{net.k}"
           f"_nb{net.nbin}_cin{net.cin}_p1")
    out = ROOT / "results/z0" / f"future_inv_net_{cfg}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    ok = (fi["lookahead_ms"] == 0.0 and not fi["unbounded"]
          and not fi["inconclusive"] and snr >= 80.0)
    print(f"  未来不変性 {fi['lookahead_ms']} ms / unbounded={fi['unbounded']}"
          f" / inconclusive={fi['inconclusive']}  (probes {fi['probes']})")
    print(f"  streaming ≡ offline  SNR {snr} dB（合格線 80）")
    print(f"  -> {out.name}   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
