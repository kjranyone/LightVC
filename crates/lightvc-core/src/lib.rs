//! LightVC core inference library.
//!
//! paper_snv の製品経路(段 P): 48 kHz・240 サンプルのブロック → 前段 v3(p_front)→ E / C(p_model)→ rvoc3hi(rvoc)。
//! 入口は `snv_stream::SnvStream`。他のモジュールはその依存。

pub mod affinity;
pub mod artic_render;
pub mod artic_vc;
pub mod c1;
pub mod enroll;
pub mod f0est;
pub mod lvl;
pub mod nvoc;
pub mod p_front;
pub mod p_model;
pub mod rr;
pub mod rvoc;
pub mod ship_front;
pub mod simd;
pub mod snv_stream;
