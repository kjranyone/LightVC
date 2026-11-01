"""D342: ppgvoc21(遅い揺らぎを条件に学習)の SR を z = 本物(×1 / ×1.5)/ ゼロで描いて v1⊥ を測る(乱数との差 = 韻律との結びつきの分)。診断。"""
import sys, subprocess, numpy as np, soundfile as sf, torch
sys.path.insert(0, '.')
import nvoc as N, train_ppgvoc as P
from pathlib import Path
O = Path('../results/conv_p0/d342'); dev = 'cuda'
st = torch.load('../results/diag_ppgvoc21/last_ema.pt', map_location='cpu', weights_only=False)
P.PPG = 'soft'; P.CTX.update(on=True, ch=96, adapt=False); P.slow_from_ckpt(st)
g = P.init_gen(dev, 'table'); g.load_state_dict(st['ema']); g.eval()
sr, _ = P.eval_items(P.e2e_held(), 'table', dev)
groups = []
with torch.no_grad():
    for name in ('ZREAL', 'ZREAL15', 'ZERO'):
        od = O / name; od.mkdir(parents=True, exist_ok=True)
        for nm, c, lg, spk in sr:
            p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
            ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
            s_ = P.spk_tensor(spk, dev)
            if name == 'ZERO':
                z = torch.zeros(1, 24, p.shape[-1], device=dev)
            else:
                z = P.slow_real(ct[:, 1:25], torch.einsum('bdk,bkt->bdt', s_[0], p) / 10) * (1.5 if name == 'ZREAL15' else 1.0)
            cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], s_, z)
            f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
            exc = P.TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
            y = g(cond, exc, s_)[0].cpu().numpy()[N.DELAY:]
            sf.write(od / f'{nm}.wav', (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        groups.append(f'{name}={od}/h*_SR.wav')
r = subprocess.run([sys.executable, 'eval_v1perp.py', '--groups', *groups, '--out', str(O / 'v1perp.json')], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-800:] if r.returncode else '')
