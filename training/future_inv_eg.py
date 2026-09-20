"""E/G のネット込み未来不変性（出荷ゲート条件 1、実音声プローブ）。

入力 mel の t 以降を別発話の尾で書き換え、t 未満の出力フレームが変わらないことを見る。
E・G とも左パディングのみの構成だが、**構成ではなく実測で証明する**
（wavehax-groupnorm の教訓: ckpt を読まない検査は 22 巡素通りした）。

    uv run python future_inv_eg.py --e ../results/diag_e1/diag_e1_best.pt \
        --g ../results/diag_cipt/diag_cipt_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from train_vc_g import G1, load_item, feat_util, load_wav, wav_path_of
from train_vc_e import E1

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", required=True)
    ap.add_argument("--g", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cpu"                     # 決定性のため CPU で判定

    ek = torch.load(a.e, map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"],
              look=ek["args"].get("look", 0)).to(dev).eval()
    enet.load_state_dict(ek["net"])
    gk = torch.load(a.g, map_location=dev)
    gnet = G1(dim=gk["args"]["dim"], layers=gk["args"]["L"]).to(dev).eval()
    gnet.load_state_dict(gk["net"])

    fs = sorted((ROOT / "data/female_tts_feat").glob("*/*.pt"))
    d1, mel1 = load_item(fs[100])
    d2, mel2 = load_item(fs[500])

    t = min(mel1.shape[-1], mel2.shape[-1]) - 8
    t0 = t // 2
    mel_a = mel1[:, :t].clone()
    mel_b = mel1[:, :t].clone()
    mel_b[:, t0:] = mel2[:, t0:t]                  # 実音声で未来を書き換え

    bad = {}
    with torch.no_grad():
        ca = enet(mel_a[None])[0]
        cb = enet(mel_b[None])[0]
    look = ek["args"].get("look", 0)
    guard = t0 - look                              # look フレームは宣言済みの右文脈
    de = (ca[:, :guard] - cb[:, :guard]).abs().max()
    edit = (ca[:, guard:] - cb[:, guard:]).abs().max()
    bad["E"] = {"prefix_maxdiff": float(de), "edit_moved": float(edit)}

    xa = feat_util(d1, mel_a)
    xb = xa.clone()
    with torch.no_grad():
        xa2, xb2 = xa.clone(), xb.clone()
        xb2[:, t0:] = feat_util(d2, mel2[:, :t])[:, t0:]
        ga = gnet(xa2[None])[0]
        gb = gnet(xb2[None])[0]
    dg = (ga[:, :t0] - gb[:, :t0]).abs().max()
    editg = (ga[:, t0:] - gb[:, t0:]).abs().max()
    bad["G"] = {"prefix_maxdiff": float(dg), "edit_moved": float(editg)}

    ok = True
    for k, v in bad.items():
        inconc = v["edit_moved"] < 1e-6            # 編集が出力を動かさない=判定不能
        p = v["prefix_maxdiff"] == 0.0 and not inconc
        ok &= p
        print(f"  {k}: prefix max|diff| {v['prefix_maxdiff']:.2e}  "
              f"編集の影響 {v['edit_moved']:.2e}  "
              f"{'PASS' if p else ('INCONCLUSIVE' if inconc else 'FAIL')}")
    res = {"lookahead_ms": 0.0 if ok else None, "inconclusive": not ok and any(
        v["edit_moved"] < 1e-6 for v in bad.values()),
        "detail": bad, "e": a.e, "g": a.g,
        "e_arch": ek["args"], "g_arch": gk["args"]}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("  =>", "PASS（先読み 0）" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
