//! LightVC core inference library.
//!
//! paper_snv の製品経路(段 P126 の K3G-v4): 48 kHz・240 サンプルのブロック → 門と瞬時重みつきの前段 v4_dual(p_gate・p_front)→ 符号と回路の 3 組の平均と錨(k3g_model・p30_model・p_fixenv)→ 出力部 snv_p123_s(rvoc・DELAY 480)→ 出力の番(out_guard)。
//! 入口は `snv_stream::SnvStream`。他のモジュールはその依存。

pub mod affinity;
pub mod cheaptrick;
pub mod fft;
pub mod k3g_model;
pub mod nvoc;
pub mod out_guard;
pub mod p_fixenv;
pub mod p_front;
pub mod p_gate;
pub mod p_model;
pub(crate) mod prof;
pub mod p30_model;
pub mod rvoc;
pub mod snv_stream;
