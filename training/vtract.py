"""声道の物理の前向きモデル(決定的・微分可能・torch)。current/artic_inv.md の構音界面の基礎。

声門 → 口唇の N 区間の音響管(平面波)。区間 i は断面積 A_i・長さ L/N。区間ごとに損失つきの伝送行列(ABCD)
  [P_in, U_in]ᵀ = [[cosh γl, Z sinh γl], [sinh γl / Z, cosh γl]] [P_out, U_out]ᵀ,  Z = ρc / A,  γ = α(f) + jω/c
を掛け合わせ、口唇に放射インピーダンス Z_rad(円形ピストン・無限バッフルの低 ka 近似)、声門は理想体積速度源とする。
  体積速度の伝達 H_U(f) = U_lips / U_glottis = 1 / (M21·Z_rad + M22)
  放射音圧 ∝ jω·U_lips(+6dB/oct)。声門流の傾斜は別の関数(glottal_tilt)。
単位は SI(m・m²・Pa・m³/s)。周波数は任意の点で評価できる(倍音の位置・mel の中心)。声道長 L は連続値。
"""
from __future__ import annotations

import math

import torch

RHO = 1.14
C = 350.0


def wall_loss(f: torch.Tensor) -> torch.Tensor:
    """単位長さあたりの減衰 α(f)[Np/m]。粘性・熱伝導(√f に比例)と壁の振動(低域で大)の簡単な合成。帯域幅が実測の桁(F1 50〜80Hz・高次で増える)に入るように選んだ定数。"""
    return 0.02 * torch.sqrt(f.clamp(min=1.0)) + 0.35 * (300.0 / f.clamp(min=50.0)) ** 2


def radiation(f: torch.Tensor, area_lip: torch.Tensor) -> torch.Tensor:
    """口唇の放射インピーダンス(円形ピストン・無限バッフル・低 ka 近似)。area_lip [...]、f [F] → [..., F] 複素。"""
    k = 2 * math.pi * f / C
    a = torch.sqrt(area_lip / math.pi)[..., None]
    z0 = (RHO * C / area_lip)[..., None]
    ka = k * a
    return z0 * ((ka ** 2) / 2 + 1j * (8 * ka / (3 * math.pi)))


def transfer(areas: torch.Tensor, length: torch.Tensor, f: torch.Tensor, lossy: bool = True, open_end_ideal: bool = False) -> torch.Tensor:
    """areas [..., N](m²・声門側から口唇側)・length [...](m)・f [F](Hz)→ 体積速度の伝達 H_U [..., F](複素)。"""
    N = areas.shape[-1]
    l = (length / N)[..., None, None]
    w = 2 * math.pi * f
    alpha = wall_loss(f) if lossy else torch.zeros_like(f)
    gam = (alpha + 1j * w / C)
    Z = (RHO * C / areas)[..., :, None]
    gl = gam * l
    zero = torch.zeros_like(Z)
    ch, sh = torch.cosh(gl) + zero, torch.sinh(gl) + zero
    m11, m12, m21, m22 = ch, Z * sh, sh / Z, ch
    p11, p12, p21, p22 = m11[..., 0, :], m12[..., 0, :], m21[..., 0, :], m22[..., 0, :]
    for i in range(1, N):
        a11, a12, a21, a22 = m11[..., i, :], m12[..., i, :], m21[..., i, :], m22[..., i, :]
        p11, p12, p21, p22 = p11 * a11 + p12 * a21, p11 * a12 + p12 * a22, p21 * a11 + p22 * a21, p21 * a12 + p22 * a22
    zr = torch.zeros_like(p21) if open_end_ideal else radiation(f, areas[..., -1])
    return 1.0 / (p21 * zr + p22)


def envelope_db(areas: torch.Tensor, length: torch.Tensor, f: torch.Tensor, lossy: bool = True) -> torch.Tensor:
    """放射音圧の包絡(dB・声門流の傾斜なし)= 20 log10 |jω H_U|。"""
    h = transfer(areas, length, f, lossy=lossy)
    return 20 * torch.log10((2 * math.pi * f) * h.abs() + 1e-12)


def glottal_tilt_db(f: torch.Tensor, fc: torch.Tensor, slope_db_oct: torch.Tensor) -> torch.Tensor:
    """声門流の振幅スペクトルの傾斜(dB): fc より上で slope_db_oct の傾き(典型 −12)。fc・slope は [...] → [..., F]。"""
    r = f / fc[..., None]
    return slope_db_oct[..., None] / (20 * math.log10(2)) * 10 * torch.log10(1 + r ** 2)


def peaks(f: torch.Tensor, env_db: torch.Tensor, k: int = 4) -> list[float]:
    """包絡の極大の周波数(放物線補間)。env_db [F]。"""
    e = env_db.detach().cpu().double()
    ff = f.detach().cpu().double()
    out = []
    for i in range(1, len(e) - 1):
        if e[i] > e[i - 1] and e[i] >= e[i + 1]:
            a, b, c = e[i - 1], e[i], e[i + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if abs(float(den)) > 1e-12 else 0.0
            out.append(float(ff[i] + d * (ff[1] - ff[0])))
            if len(out) >= k:
                break
    return out


def _selftest() -> None:
    torch.set_default_dtype(torch.float64)
    f = torch.linspace(50, 5000, 4951)
    L = torch.tensor(0.175)
    uni = torch.full((20,), 3e-4)
    e0 = 20 * torch.log10(transfer(uni, L, f, lossy=False, open_end_ideal=True).abs())
    pk = peaks(f, e0, 3)
    want = [(2 * n - 1) * C / (4 * 0.175) for n in (1, 2, 3)]
    print("均一管 17.5cm(無損失・理想開端)のフォルマント", [round(p) for p in pk], "理論", [round(w) for w in want])
    assert all(abs(p - w) / w < 0.01 for p, w in zip(pk, want)), (pk, want)
    pk2 = peaks(f, 20 * torch.log10(transfer(uni, L / 1.2, f, lossy=False, open_end_ideal=True).abs()), 3)
    assert all(abs(b / a - 1.2) < 0.01 for a, b in zip(pk, pk2)), (pk, pk2)
    print("声道長 /1.2 → フォルマント比", [round(b / a, 3) for a, b in zip(pk, pk2)])
    er = envelope_db(uni, L, f)
    pkr = peaks(f, er, 3)
    print("損失 + 放射つき", [round(p) for p in pkr], "(放射の質量で少し下がる)")
    assert all(0.9 < p / w < 1.0 for p, w in zip(pkr, want)), pkr
    x = torch.linspace(0, 1, 20)
    a_tube = torch.where(x < 0.5, torch.tensor(0.6e-4), torch.tensor(6e-4))
    i_tube = torch.where(x < 0.5, torch.tensor(6e-4), torch.tensor(0.6e-4))
    fa, fi = peaks(f, envelope_db(a_tube, L, f), 2), peaks(f, envelope_db(i_tube, L, f), 2)
    print("/a/ 型(奥が狭い)F1, F2 =", [round(p) for p in fa], " /i/ 型(前が狭い)F1, F2 =", [round(p) for p in fi])
    assert fa[0] > fi[0] and fa[1] < fi[1], (fa, fi)
    ar = torch.full((20,), 3e-4, requires_grad=True)
    envelope_db(ar, L, torch.linspace(100, 4000, 50)).sum().backward()
    assert torch.isfinite(ar.grad).all() and ar.grad.abs().sum() > 0
    print("断面積への勾配: 有限・非ゼロ")
    bw = []
    for i, p in enumerate(pkr):
        j = int(torch.argmin((f - p).abs()))
        m = er[j]
        lo = j
        while lo > 0 and er[lo] > m - 3:
            lo -= 1
        hi = j
        while hi < len(f) - 1 and er[hi] > m - 3:
            hi += 1
        bw.append(float(f[hi] - f[lo]))
    print("帯域幅(−3dB)", [round(b) for b in bw], "Hz(実測の桁: F1 50〜80・F2 60〜120・F3 100〜200)")
    print("vtract selftest OK")


if __name__ == "__main__":
    _selftest()
