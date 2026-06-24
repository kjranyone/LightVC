"""
Collapse診断: モデルがtarget話者参照を無視しているか確認

固定source発話を10人の異なるtarget話者で変換し、
予測mcepの分散 / 正解target mcepの分散 を測る。

比が低い（例: <0.3）= モデルはrefを無視している（collapse）
比が高い（例: >0.7）= モデルはrefに従っている

複数の統計量を見る:
  Var_frame  : フレームごとの予測の、話者間分散 / 正解の話者間分散
  Mean_diff  : 話者平均mcepの、話者間分散 / 正解の話者間分散
  L1_inter   : 異なる話者予測間の平均L1距離
"""
import sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda")
MC_DIM = 25


def main():
    from train_sf_vc import EnvelopeConverter, load_speaker_embeddings

    print("=== Collapse診断 ===\n")

    spk_emb = load_speaker_embeddings()
    spk_ids = sorted(spk_emb.keys())[:10]

    ckpt = torch.load("checkpoints/sf_vc/latest.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = EnvelopeConverter(
        mc_dim=cfg["mc_dim"], spk_dim=cfg["spk_dim"],
        hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # sf_pairsから1つのsource発話を固定
    pair_data = np.load("data/sf_pairs/pair_000000.npz")
    mc_src = torch.from_numpy(pair_data["mc_src"]).float().unsqueeze(0).to(DEVICE).transpose(1, 2)
    T = mc_src.shape[2]

    print(f"Source: {pair_data['spk_src']} → {T} frames")
    print(f"Target speakers: {len(spk_ids)}\n")

    # 各target話者で予測
    predictions = []
    for spk_id in spk_ids:
        emb = spk_emb[spk_id].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(mc_src, emb)
        pred_np = pred.squeeze(0).cpu().numpy().T  # (T, mc_dim)
        predictions.append(pred_np)

    predictions = np.stack(predictions)  # (n_spk, T, mc_dim)

    # 正解: 各話者のmc_cacheから同じテキストのmcepを取得
    # pair_000000のtargetを基準に、全話者の同じテキストを探す
    text_id = None
    src_spk = str(pair_data["spk_src"])
    tgt_spk_orig = str(pair_data["spk_tgt"])

    # mc_cacheから各話者の多数の発話の平均mcepを計算（話者プロファイル代わり）
    cache_dir = Path("data/mc_cache")
    speaker_mean_mc = {}
    speaker_all_mc = {}
    for spk_id in spk_ids:
        spk_dir = cache_dir / spk_id
        if not spk_dir.exists():
            speaker_mean_mc[spk_id] = np.zeros(MC_DIM)
            speaker_all_mc[spk_id] = np.zeros((1, MC_DIM))
            continue
        spk_files = sorted(spk_dir.glob("*.npz"))[:20]
        if len(spk_files) == 0:
            speaker_mean_mc[spk_id] = np.zeros(MC_DIM)
            speaker_all_mc[spk_id] = np.zeros((1, MC_DIM))
            continue
        all_mc = []
        for f in spk_files:
            d = np.load(f)
            all_mc.append(d["mc"])
        all_mc = np.concatenate(all_mc, axis=0)
        speaker_mean_mc[spk_id] = all_mc.mean(axis=0)
        speaker_all_mc[spk_id] = all_mc

    # 統計量1: 予測の話者間分散
    # フレームごとに、10話者の予測の分散を計算 → フレーム平均
    var_pred = predictions.var(axis=0).mean()  # scalar

    # 統計量2: 正解mcepの話者間分散
    # 各話者の平均mcepの分散
    means_arr = np.stack([speaker_mean_mc[s] for s in spk_ids])  # (n_spk, mc_dim)
    var_truth = means_arr.var(axis=0).mean()

    ratio_var = var_pred / var_truth if var_truth > 0 else float('inf')

    # 統計量3: 話者平均mcepの比較
    pred_means = predictions.mean(axis=1)  # (n_spk, mc_dim)
    var_pred_means = pred_means.var(axis=0).mean()
    ratio_means = var_pred_means / var_truth if var_truth > 0 else float('inf')

    # 統計量4: 異なる話者予測間の平均L1距離
    l1_inter_pred = []
    for i in range(len(spk_ids)):
        for j in range(i+1, len(spk_ids)):
            l1 = np.abs(predictions[i] - predictions[j]).mean()
            l1_inter_pred.append(l1)
    l1_inter_pred = np.mean(l1_inter_pred)

    # 正解の話者間L1
    l1_inter_truth = []
    for i in range(len(spk_ids)):
        for j in range(i+1, len(spk_ids)):
            mc_i = speaker_all_mc[spk_ids[i]]
            mc_j = speaker_all_mc[spk_ids[j]]
            n = min(len(mc_i), len(mc_j), 500)
            l1 = np.abs(mc_i[:n] - mc_j[:n]).mean()
            l1_inter_truth.append(l1)
    l1_inter_truth = np.mean(l1_inter_truth)

    ratio_l1 = l1_inter_pred / l1_inter_truth if l1_inter_truth > 0 else float('inf')

    # 統計量5: 予測の有効rank (PCA explained variance ratio)
    from sklearn.decomposition import PCA
    pred_flat = predictions.reshape(len(spk_ids), -1)
    pca_pred = PCA(n_components=min(len(spk_ids)-1, 9))
    pca_pred.fit(pred_flat)
    eff_rank_pred = 1.0 / np.sum(pca_pred.explained_variance_ratio_ ** 2)

    print("--- 崩壊診断結果 ---")
    print(f"Var(予測)/Var(正解):                    {ratio_var:.4f}")
    print(f"Var(予測話者平均)/Var(正解話者平均):     {ratio_means:.4f}")
    print(f"L1(予測話者間)/L1(正解話者間):          {ratio_l1:.4f}")
    print(f"予測の有効rank (10話者中):               {eff_rank_pred:.2f}")
    print(f"  → 1.0に近い = 1パターンしか出ない (完全崩壊)")
    print(f"  → 10に近い = 話者ごとに違う予測")

    # coeff別の分散比
    print(f"\n--- coeff別 話者間分散比 ---")
    var_pred_per_coeff = predictions.var(axis=0).mean(axis=0)  # (mc_dim,)
    var_truth_per_coeff = means_arr.var(axis=0)  # (mc_dim,)
    ratio_per_coeff = var_pred_per_coeff / (var_truth_per_coeff + 1e-10)
    for c in range(min(10, MC_DIM)):
        print(f"  mc[{c:2d}]: pred_var={var_pred_per_coeff[c]:.6f}  truth_var={var_truth_per_coeff[c]:.6f}  ratio={ratio_per_coeff[c]:.4f}")

    # 結論
    print(f"\n--- 結論 ---")
    if ratio_means < 0.1:
        print("SEVERE COLLAPSE: モデルはtarget話者をほぼ無視 (ratio < 0.1)")
    elif ratio_means < 0.3:
        print("MODERATE COLLAPSE: target話者参照が弱い (ratio < 0.3)")
    elif ratio_means < 0.7:
        print("PARTIAL: target話者参照があるが不十分 (ratio < 0.7)")
    else:
        print("OK: target話者参照は機能している (ratio >= 0.7)")

    results = {
        "ratio_var": float(ratio_var),
        "ratio_means": float(ratio_means),
        "ratio_l1": float(ratio_l1),
        "eff_rank_pred": float(eff_rank_pred),
        "ratio_per_coeff": ratio_per_coeff.tolist(),
    }
    with open("results/collapse_diag.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n保存: results/collapse_diag.json")


if __name__ == "__main__":
    main()
