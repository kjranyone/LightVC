"""V の高域スパイクが in-domain 起因か OOD-mel 応答かの切り分け（tag diag_dom）。

    cs_dom: 女声 held GT mel + GT f0 -> V（V の in-domain copy-syn）
    gv_dom: 同コンテンツで G(ciptB) mel -> V（G の学習域での G→V）

男性 mp3 mel（OOD）の cs と突き合わせてスパイク率の帰属を切る。

    uv run python diag_dom.py --out ../results/diag_dom
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
import v2f as V2
from eval_g1 import render
from rddsp_gpu import mel_to_linear
from train_vc_g import G1, load_item, feat_util, FEATS

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/diag_dom"))
    ap.add_argument("--g", default=str(ROOT / "results/diag_ciptB/diag_ciptB_best.pt"))
    ap.add_argument("--vtag", default="v2f_prior_20260819_013905")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    gk = torch.load(a.g, map_location=dev)
    gnet = G1(dim=gk["args"]["dim"], layers=gk["args"]["L"]).to(dev).eval()
    gnet.load_state_dict(gk["net"])
    vk = torch.load(ROOT / f"results/{a.vtag}/{a.vtag}_best.pt", map_location=dev)
    vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                  norm=vk["args"]["norm"]).to(dev).eval()
    vnet.load_state_dict(vk["net"])
    W = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)

    files = []
    for root in FEATS:
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                files += sorted(spk.glob("*.pt"))
    spk_all = sorted({f.parent.name for f in files})
    held = set(spk_all[-24:])
    ev = [f for f in files if f.parent.name in held][: a.n]

    import librosa, soundfile
    for i, f in enumerate(ev):
        d, mel = load_item(f)
        wav_path = ROOT / str(d["path"]).lstrip("./")
        if not wav_path.exists():
            wav_path = Path(str(d["path"]))
        w, _ = librosa.load(str(wav_path), sr=44100, mono=True)
        gt = torch.from_numpy(w) * 32768.0
        n = gt.shape[-1]
        f0, _ = SF.causal_f0(gt)
        y_cs = render(mel, f0, vnet, W, n, dev)
        soundfile.write(out / f"cs_dom_{i}.wav", y_cs.clamp(-1, 1).numpy(), 44100)
        with torch.no_grad():
            x = feat_util(d, mel).to(dev)[None]
            mg = gnet(x)[0].cpu()
        y_gv = render(mg, f0, vnet, W, n, dev)
        soundfile.write(out / f"gv_dom_{i}.wav", y_gv.clamp(-1, 1).numpy(), 44100)
        print(f"  {i}: {f.parent.name[:10]} cs_dom/gv_dom 生成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
