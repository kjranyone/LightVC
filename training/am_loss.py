"""フレーム周期(200Hz)の包絡変調の線の、微分できる罰則(eval_nvoc.amline の線の定義に基づく)。
帯域(2–4k・4–8k・8–12k)を FFT マスクで取り出し、解析信号の包絡 → 1/12 に間引き(平均)→ 平均を引く → 区間全体に 1 つのハン窓 → パワースペクトル →
**バッチ全体でパワーを平均**(0.5 秒の 1 区間は雑音で ±1dB 振れるので 16 区間で平均する)。
線のパワー pk = 200/400/600Hz の ±3Hz の合計・近傍のパワー bg = ±40Hz(線を除く)の平均 × 線のビン数。
罰則 = 帯域平均で
  relu((pk_dB − bg_dB)(出力) − (pk_dB − bg_dB)(目標) − m_line)(two_sided では |·|・目標より低く行き過ぎる谷も罰する)  … 線の近傍に対する超過(評価 amline と同じ比・出力は目標より変調が全体に小さいので絶対パワーでは効かない)
  + relu(bg_dB(出力) − bg_dB(目標) − m_bg)                         … 近傍の変調の水増し・線の移動(190Hz などへ逃げる)の抜け道(水増しだけを片方向で罰する)
dB は目標の 100〜800Hz の変調パワーの eps_rel 倍の床つき(低レベルで勾配が 1/振幅に爆発しない)。
注意: proxy は評価の amline より感度が低い(約 0.6 倍・相関 0.8)ので昇格の根拠にしない。評価は eval_nvoc.amline。"""
from __future__ import annotations

import torch

BANDS = ((2000.0, 4000.0), (4000.0, 8000.0), (8000.0, 12000.0))
LINES = (200.0, 400.0, 600.0)


def band_mod_power(w: torch.Tensor, sr: int = 48000, dec: int = 12):
    """w [B, L] → (P [3, B, F], fm [F]): 帯域ごとの包絡の変調パワースペクトル(区間ごと・窓は 1 つ)。"""
    B, L = w.shape
    X = torch.fft.rfft(w.float(), dim=-1)
    f = torch.fft.rfftfreq(L, 1.0 / sr).to(w.device)
    Le = L // dec
    fm = torch.fft.rfftfreq(Le, dec / sr).to(w.device)
    win = torch.hann_window(Le, device=w.device)
    out = []
    for lo, hi in BANDS:
        m = ((f >= lo) & (f < hi)).float()
        full = torch.zeros(B, L, dtype=torch.complex64, device=w.device)
        full[:, :L // 2 + 1] = 2 * X * m
        env = torch.fft.ifft(full).abs()
        env = torch.nn.functional.avg_pool1d(env[:, None], dec, dec)[:, 0]
        env = (env - env.mean(-1, keepdim=True)) * win
        out.append(torch.fft.rfft(env, dim=-1).abs() ** 2)
    return torch.stack(out, 0), fm


def line_bg(P: torch.Tensor, fm: torch.Tensor, floor: torch.Tensor):
    """P [3, B, F]・floor [3] → (pk_dB [3], bg_dB [3], mod_total_dB [3]): バッチ平均のパワーから。"""
    Pm = P.mean(1)
    pk = bg = 0.0
    for h in LINES:
        on = ((fm >= h - 3) & (fm <= h + 3)).float()
        nb = ((fm >= h - 40) & (fm <= h + 40)).float() * (1 - on)
        pk = pk + (Pm * on).sum(-1)
        bg = bg + (Pm * nb).sum(-1) / nb.sum().clamp(min=1) * on.sum()
    tm = ((fm >= 100) & (fm <= 800)).float()
    tot = (Pm * tm).sum(-1)
    db = lambda v: 10 * torch.log10(v + floor)
    return db(pk), db(bg), 10 * torch.log10(tot.clamp(min=1e-20))


def am_penalty(y: torch.Tensor, t: torch.Tensor, m_line: float = 0.3, m_bg: float = 0.0, two_sided: bool = False, eps_rel: float = 1e-3) -> tuple:
    """戻り値 = (罰則, dict(line_excess_db・bg_dev_db・mod_out_db・mod_tgt_db: いずれも帯域平均・detach))。"""
    Py, fm = band_mod_power(y)
    with torch.no_grad():
        Pt, _ = band_mod_power(t)
        tm = ((fm >= 100) & (fm <= 800)).float()
        floor = eps_rel * (Pt.mean(1) * tm).sum(-1)
        pk_t, bg_t, mod_t = line_bg(Pt, fm, floor)
    pk_y, bg_y, mod_y = line_bg(Py, fm, floor)
    ex = (pk_y - bg_y) - (pk_t - bg_t)
    dev = bg_y - bg_t
    lin = torch.relu(ex.abs() - m_line) if two_sided else torch.relu(ex - m_line)
    pen = (lin + torch.relu(dev - m_bg)).mean()
    return pen, {"line_excess_db": ex.detach().mean(), "bg_dev_db": dev.detach().mean(), "mod_out_db": mod_y.detach().mean(), "mod_tgt_db": mod_t.mean()}
