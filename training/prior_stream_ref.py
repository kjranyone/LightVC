import math
import pathlib
import sys

import scipy.signal as ss
import soundfile as sf
import torch

import rddsp as R
import rddsp_gpu as RG
import causal_mel as CM
import ship_front as SF

# ⚠ 2.1 で ship_front / causal_mel に入れる front-end の頭の修正を、ここでは
#   monkey-patch で再現する（参照実装なので本体は触らない）。
#   `cstft` も `causal_mel._mel` も頭を **reflect** で埋めており、これは t=0 で
#   x[1:769] を読む＝**先読み 768 サンプル**。オフライン学習経路にも入っていた
#   未計測の非因果で、製品のストリーミングは再現できない。零詰めにすると
#   プライムも先読みも要らずに parity が取れる（実学習データ 20 本・最悪 +138.37 dB）。
def _cstft_zero_head(x, nfft=SF.NFFT_A, hop=SF.HOP_A):
    xp = torch.nn.functional.pad(x[..., None, :], (nfft - hop, 0),
                                 mode="constant")[..., 0, :]
    q = ((-x.shape[-1]) % hop) + (nfft - hop)
    xp = torch.nn.functional.pad(xp, (0, q))
    return torch.stft(xp, nfft, hop, nfft, SF._win(nfft, x.device), center=False,
                      return_complex=True, normalized=False)


def _mel_zero_head(y, n_fft, hop, num_mels, sr, fmin, fmax, left_pad, right_pad):
    if y.dim() == 1:
        y = y.unsqueeze(0)
    fb = CM._mel_fb(sr, n_fft, num_mels, fmin, fmax, y.device)
    w = CM._win(n_fft, y.device)
    yp = torch.nn.functional.pad(y.unsqueeze(1), (left_pad, right_pad),
                                 mode="constant").squeeze(1)
    sp = torch.stft(yp, n_fft, hop_length=hop, win_length=n_fft, window=w,
                    center=False, normalized=False, onesided=True,
                    return_complex=True)
    sp = torch.sqrt(torch.view_as_real(sp).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(fb, sp), min=1e-5))


SF.cstft = _cstft_zero_head
CM._mel = _mel_zero_head

K: int = 2
HS: int = SF.HOP_S
HA: int = SF.HOP_A
BLK: int = K * HS
IN_HIST: int = 1792
EXC_HIST: int = SF.NFFT_S - SF.HOP_S
GATE_DB: float = 80.0
MIN_PROBES: int = 16
# 励起リングの頭 (NFFT_S−HOP_S)/HOP_S フレームは、offline が持つ履歴を streaming が
# まだ持たない区間。除外しないと頭だけで SNR が 60〜80 dB に落ちる。
WARMUP_FRAMES: int = (SF.NFFT_S - SF.HOP_S) // SF.HOP_S


def _harmonics(f0u: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    n = f0u.shape[-1]
    imp = torch.zeros(n)
    cnt = torch.zeros(n)
    for s0 in range(0, SF.KMAX, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, SF.KMAX + 1),
                          dtype=torch.float32)[:, None]
        m = (kk * f0u[None] < R.SR / 2).float()
        imp = imp + (torch.cos(kk * phi[None]) * m).sum(0)
        cnt = cnt + m.sum(0)
    return imp / cnt.clamp(min=1.0).sqrt()


def full_prior(x: torch.Tensor, z: torch.Tensor):
    n = x.shape[-1]
    f0, _ = SF.causal_f0(x)
    fl = SF._fill(f0)
    f0u = SF.frame_upsample_causal(fl.double(), n, HA).clamp(min=0.0)
    acc = torch.cumsum(f0u, 0)
    phi = torch.remainder(2 * math.pi * acc / R.SR, 2 * math.pi).float()
    exc = 0.7 * _harmonics(f0u.float(), phi) + 0.3 * z
    E = SF.cstft(exc, SF.NFFT_S, HS)
    W = RG.mel_to_linear(x.device)
    mlin = SF.to_frames(W @ SF.mel(x), SF.n_frames(n))
    m = min(E.shape[-1], mlin.shape[-1])
    return E[:, :m] * (mlin[:, :m] - SF.MEL_REF).exp(), exc


def stream_prior(x: torch.Tensor, z: torch.Tensor):
    n = x.shape[-1]
    W = RG.mel_to_linear(x.device)
    fl_all: list[float] = []
    mel_all: list[torch.Tensor] = []
    acc = torch.zeros((), dtype=torch.float64)
    last_voiced: float | None = None
    # ⚠ プライムはしない。rev58 の reflect プライムは **t=0 で x[1:769] を読む＝
    #   先読み 768 サンプル（17.41 ms）**で、「起動コストだから台帳外」は誤りだった。
    #   頭を零詰めにすれば、ゼロ初期化のリングのままで一致する。
    xr = torch.zeros(IN_HIST)
    er = torch.zeros(EXC_HIST)
    outP, outE = [], []
    for st in range(0, n - BLK + 1, BLK):
        buf = torch.cat([xr, x[st:st + BLK]])
        xr = buf[-IN_HIST:]
        # 起動直後はリングに実サンプルが足りない。**ゼロを「信号」として解析しない**
        # ——offline はそのフレームを持たないので、持たせると `_causal_median` の
        # 窓の中身が食い違い、位相アキュムレータが恒久オフセットに積分する。
        b2 = buf[-min(st + BLK, IN_HIST + BLK):]
        ia = (len(b2) - BLK) // HA
        fb, _ = SF.causal_f0(b2)
        mb = SF.mel(b2)
        for c in range(BLK // HA):
            v = float(fb[ia + c])
            if v > 50 or last_voiced is None:
                last_voiced = max(v, 50.0)
            fl_all.append(last_voiced)
            mel_all.append(mb[:, ia + c])
        fl = torch.tensor(fl_all, dtype=torch.float64)
        p = torch.arange(st, st + BLK, dtype=torch.float64)
        t = (p + 1.0) / HA - 2.0
        i = t.floor().clamp(0, len(fl_all) - 2)
        fr = (t - i).clamp(0.0, 1.0)
        j = i.long()
        f0u = (fl[j] * (1 - fr) + fl[j + 1] * fr).clamp(min=0.0)
        a = torch.cumsum(f0u, 0) + acc
        acc = a[-1]
        phi = torch.remainder(2 * math.pi * a / R.SR, 2 * math.pi).float()
        exc = 0.7 * _harmonics(f0u.float(), phi) + 0.3 * z[st:st + BLK]
        outE.append(exc)
        eb = torch.cat([er, exc])
        er = eb[-EXC_HIST:]
        E = SF.cstft(eb, SF.NFFT_S, HS)[:, EXC_HIST // HS: EXC_HIST // HS + K]
        tt = torch.arange(st // HS, st // HS + K)
        jm = ((tt * HS + HS - HA) // HA).clamp(min=0).clamp(max=len(mel_all) - 1)
        mlin = torch.stack([mel_all[int(q)] for q in jm], -1)
        outP.append(E * (W @ mlin - SF.MEL_REF).exp())
    return torch.cat(outP, -1), torch.cat(outE)


def snr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    e = (a - b).abs().pow(2).sum()
    return float(10 * torch.log10(a.abs().pow(2).sum() / e.clamp_min(1e-30)))


def measure(src: str) -> tuple[float, float]:
    w, sr = sf.read(src)
    if w.ndim > 1:
        w = w.mean(1)
    w = ss.resample_poly(w, R.SR, sr)[:R.SR * 3]
    x = torch.tensor(w, dtype=torch.float32)
    z = torch.randn(x.shape[-1], generator=torch.Generator().manual_seed(999))
    Pf, ef = full_prior(x, z)
    Pb, eb = stream_prior(x, z)
    t = min(Pf.shape[-1], Pb.shape[-1])
    ne = min(len(ef), len(eb))
    k = WARMUP_FRAMES
    return snr_db(ef[:ne], eb[:ne]), snr_db(Pf[:, k:t], Pb[:, k:t])


def main() -> int:
    # ⚠ n=1 で PASS を宣言しない。rev57 まで 1 音源で +100.07 dB と記録していたが、
    #   実音声 9 本では中央値 +66.69 dB・8 本が 80 dB 未満だった。
    # ⚠ プローブは**学習コーパスの実音声**から採る。rev58 の既定は
    #   `male_tts_corpus` の先頭 8 本＝**1 話者・TTS 出力・2 テキスト × 5 スタイル**で、
    #   実質の独立標本は 2。分布も製品（実女性 2775 話者）と違った。
    if len(sys.argv) > 1:
        return _from_files(sys.argv[1:])
    import train_gvoc as TG
    items = []
    for f in sorted(TG.SHARDS.glob("sh_*.pt"))[:2]:
        items += torch.load(f, map_location="cpu", weights_only=False)[:10]
    items = items[:MIN_PROBES]
    if len(items) < MIN_PROBES:
        print(f"  ⚠ プローブが {len(items)} 本（{MIN_PROBES} 本必要）。少数で PASS を宣言しない")
        return 1
    print(f"  入力リング {IN_HIST}（プライム無し・先読み 0） / 励起リング {EXC_HIST} / "
          f"K={K} / 除外 {WARMUP_FRAMES} フレーム / プローブ {len(items)} 本（学習コーパス実音声）")
    worst = 1e9
    for n, it in enumerate(items):
        x = (it["w"].float() / 32767.0)[:R.SR * 3]
        z = torch.randn(x.shape[-1], generator=torch.Generator().manual_seed(999))
        Pf, ef = full_prior(x, z)
        Pb, eb = stream_prior(x, z)
        t = min(Pf.shape[-1], Pb.shape[-1])
        p = snr_db(Pf[:, WARMUP_FRAMES:t], Pb[:, WARMUP_FRAMES:t])
        worst = min(worst, p)
        print(f"  probe {n:2d}  prior {p:+8.2f} dB")
    ok = worst >= GATE_DB
    print(f"  最悪 prior parity = {worst:+.2f} dB（合格線 {GATE_DB}・n={len(items)}）")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _from_files(srcs: list[str]) -> int:
    if len(srcs) < 3:
        print("  ⚠ 音源が 3 本未満。n=1 で PASS を宣言しない")
        return 1
    print(f"  入力リング {IN_HIST}（プライム無し・先読み 0） / 励起リング {EXC_HIST} / "
          f"K={K} / 除外 {WARMUP_FRAMES} フレーム")
    worst = 1e9
    for q in srcs:
        e, p = measure(q)
        worst = min(worst, p)
        print(f"  {pathlib.Path(q).name[:24]:26s} 励起 {e:+8.2f} / prior {p:+8.2f} dB")
    ok = worst >= GATE_DB
    print(f"  最悪 prior parity = {worst:+.2f} dB（合格線 {GATE_DB}・n={len(srcs)}）")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
