"""D693(論文 current/paper_mf.md の E-B1b・規則は results/conv_p0/d693/prereg.yaml): 声の低い実女声の入力(d688_lowself2・学習の外・170〜230 Hz)を
LX = 高い声の目標(dev2 の目標のうち B1R の表がある 28 目標・d487_uiro と同じ登録)と LL = 別の声の低い女声(lowself2 の別の話者)へ k0 で変換する組を作る(リンクだけ)。
入力 1 本につき 2 目標(LX: ti = 有る目標の (2·pi + j) mod 28 番目・LL: ti = (pi + 9·(j + 1)) mod 27)。同じ (pi, j) の LX と LL は同じ入力の音声。
    uv run python d693_build_lowx.py
"""
from __future__ import annotations

import json
from pathlib import Path

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
LOW = H / "d688_lowself2"
HI = H / "d487_uiro"


def link(dst: Path, src: Path) -> None:
    assert src.exists(), src
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src)


def main() -> int:
    low = json.loads((LOW / "devset.json").read_text())
    hi = json.loads((HI / "devset.json").read_text())
    n = len(low["pairs"])
    assert all(r["pi"] == r["ti"] for r in low["pairs"])
    av = sorted(int(d.name[3:]) for d in (HI / "rr_en_b1r").glob("en_*"))
    sets = {"d693_lx": (hi["targets"], HI, ("rr_en", "rr_en_b1r"), lambda pi, j: av[(2 * pi + j) % len(av)]),
            "d693_ll": (low["targets"], LOW, ("rr_en",), lambda pi, j: (pi + 9 * (j + 1)) % n)}
    for nm, (tg, TS, subs, pick) in sets.items():
        S = H / nm
        pairs = []
        for r in low["pairs"]:
            pi = r["pi"]
            link(S / "d56/_work" / f"op_{pi:04d}.wav", LOW / "d56/_work" / f"op_{pi:04d}.wav")
            link(S / "d85_work" / f"op_{pi:04d}", LOW / "d85_work" / f"op_{pi:04d}")
            for j in range(2):
                ti = pick(pi, j)
                assert nm != "d693_ll" or ti != pi
                k = f"p{pi:04d}_t{ti:03d}"
                link(S / "d56/_work" / f"pre_{k}.wav", LOW / "d56/_work" / f"pre_p{pi:04d}_t{pi:03d}.wav")
                link(S / "d56" / f"{k}_SRC.wav", LOW / "d56" / f"p{pi:04d}_t{pi:03d}_SRC.wav")
                for sub in subs:
                    link(S / sub / f"en_{ti:03d}", TS / sub / f"en_{ti:03d}")
                pairs.append({"pi": pi, "ti": ti, "j": j, "kind": nm[5:].upper(), "spk": r["spk"], "f0med_audio": r["f0med_audio"], "target": tg[ti]})
        (S / "devset.json").write_text(json.dumps({"targets": tg, "pairs": pairs, "kind": nm[5:].upper()}, indent=1))
        print(nm, len(pairs), "pairs", len({p["ti"] for p in pairs}), "targets", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
