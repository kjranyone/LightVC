//! 話者登録の窓(モーダル)とリアルタイム画面の「話者登録」の容器。
//!
//! 窓の中で入力装置・波形・レベルの判定を見せ、30 秒の録音と登録の一覧(試聴・削除・WAV の追加)をまとめる。
//! キーボード: Enter = 録音を開始・Esc = 録音の中止 / 窓を閉じる。

use std::path::{Path, PathBuf};
use std::time::Duration;

use gpui::{
    canvas, div, fill, point, prelude::*, px, relative, size, Animation, AnimationExt as _, AnyElement, Bounds, Context, Div,
    KeyDownEvent, Rgba, SharedString, Window,
};

use super::{file_name, play_path, section_title, stop_player, supporting, LightVcUi, Tab};
use crate::enroll::{self, Live, Phase};
use crate::m3::{self, c, icon, shape, Btn, Typo};

/// 読み上げの例(何を話してもよい・普段の声で)。
const SCRIPT: &str = "今日はいい天気ですね。駅前に新しくできたお店で、お昼ごはんを食べてきました。\
明日は朝から友だちと出かける予定なので、今夜は早めに寝ようと思います。\
最近は、好きな音楽を聴きながら近所を散歩するのが楽しみです。";

/// 1 回の録音で、声が入っていてほしい秒数の目安。
const VOICED_MIN_S: f32 = 15.0;

fn scrim() -> Rgba {
    Rgba { r: 0.0, g: 0.0, b: 0.0, a: 0.32 }
}

/// 「n 本・計 x 秒」(長さが分からなければ本数だけ)。
fn count(n: usize, total: f32) -> String {
    if total > 0.5 { format!("{n} 本・計 {total:.0} 秒") } else { format!("{n} 本") }
}

/// 振幅を表示の高さ(0..1)へ。-60 dBFS を 0 にする対数の目盛り(小さな声でも形が見える)。
fn amp(v: f32) -> f32 {
    let a = v.abs();
    if a < 1e-6 { 0.0 } else { (((20.0 * a.log10()) + 60.0) / 60.0).clamp(0.0, 1.0) * v.signum() }
}

/// 列ごとの (最小, 最大) を棒で描く。total = 全体の列の数・offset = 最初の列の位置(見張りは右詰め)。
fn wave(cols: Vec<(f32, f32)>, total: usize, offset: usize, color: Rgba, clip: Rgba, axis: Rgba) -> impl IntoElement {
    canvas(
        |_, _, _| {},
        move |b, _, window, _| {
            let (w, h) = (f32::from(b.size.width), f32::from(b.size.height));
            let cy = h / 2.0;
            window.paint_quad(fill(Bounds::new(point(b.origin.x, b.origin.y + px(cy - 0.5)), size(b.size.width, px(1.))), axis));
            let total = total.max(1);
            let bars = ((w / 3.0).floor() as usize).clamp(1, total);
            let step = w / bars as f32;
            for i in 0..bars {
                let (c0, c1) = (i * total / bars, ((i + 1) * total / bars).max(i * total / bars + 1));
                let (lo, hi) = (c0.saturating_sub(offset), c1.saturating_sub(offset));
                if c1 <= offset || lo >= cols.len() {
                    continue;
                }
                let seg = &cols[lo..hi.min(cols.len())];
                let (mn, mx) = seg.iter().fold((0.0f32, 0.0f32), |(a, z), &(p, q)| (a.min(p), z.max(q)));
                let (top, bot) = (cy - amp(mx).max(0.0) * cy * 0.92, cy - amp(mn).min(0.0) * cy * 0.92);
                let col = if mn.abs().max(mx.abs()) >= 0.98 { clip } else { color };
                let x = b.origin.x + px(i as f32 * step);
                window.paint_quad(fill(Bounds::new(point(x, b.origin.y + px(top)), size(px((step - 1.0).max(1.0)), px((bot - top).max(1.0)))), col));
            }
        },
    )
    .w_full()
    .h_full()
}

/// 入力の大きさの判定(アイコン・文・色)。色だけに頼らず、必ず文で伝える。
fn judge(l: &Live) -> (&'static str, &'static str, Rgba) {
    if l.clip {
        ("error", "音が割れています。マイクから少し離れるか、入力の音量を下げてください。", c::error())
    } else if l.voice_db < -60.0 {
        ("mic_off", "声が届いていません。話しかけてみてください。反応しないときは、マイクの接続と入力装置を確かめてください。", c::error())
    } else if l.voice_db < -40.0 {
        ("info", "声が小さめです。マイクに近づくか、入力の音量を上げてください。", c::on_surface_variant())
    } else {
        ("check", "ちょうどよい大きさです。", c::tertiary())
    }
}

/// デモ表示用の見かけの入力(録音の途中)。
fn demo_live() -> Live {
    let t = |i: usize| (i as f32 * 0.37).sin().abs() * (0.25 + 0.2 * (i as f32 * 0.05).sin()) * if (i / 40) % 4 == 3 { 0.03 } else { 1.0 };
    let mut l = Live { phase: Phase::Recording, device: "マイク (USB Audio Device)".into(), host: "WASAPI".into(), sample_rate: 48_000, ..Default::default() };
    l.scope = (0..enroll::SCOPE_COLS).map(|i| (-t(i + 600), t(i + 600) * 0.9)).collect();
    l.take = (0..600).map(|i| (-t(i), t(i) * 0.9)).collect();
    (l.rms_db, l.peak_db, l.voice_db, l.recorded, l.voiced) = (-21.0, -8.0, -16.0, 12.0, 9.4);
    l
}

impl LightVcUi {
    pub(super) fn open_enroll(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.enroll_open = true;
        self.enroll_focus.focus(window);
        cx.notify();
    }

    pub(super) fn close_enroll(&mut self, cx: &mut Context<Self>) {
        self.enroll = None;
        self.enroll_open = false;
        stop_player(&mut self.enroll_player);
        self.enroll_playing = None;
        cx.notify();
    }

    fn reopen_enroll(&mut self, cx: &mut Context<Self>) {
        self.enroll = None;
        cx.notify();
    }

    fn enroll_live(&self) -> Option<Live> {
        if self.demo { Some(demo_live()) } else { self.enroll.as_ref().map(|s| s.live()) }
    }

    fn start_take(&mut self, cx: &mut Context<Self>) {
        if let Some(s) = &self.enroll
            && s.live().phase == Phase::Monitoring
        {
            stop_player(&mut self.enroll_player);
            self.enroll_playing = None;
            s.start_take();
        }
        cx.notify();
    }

    fn duration_of(&mut self, p: &Path) -> Option<f32> {
        *self.durations.entry(p.to_path_buf()).or_insert_with(|| enroll::duration_s(p))
    }

    /// 登録の合計の秒数(分からない物は数えない)。
    fn enrolled_seconds(&mut self, ops: &[PathBuf]) -> f32 {
        ops.iter().filter_map(|p| self.duration_of(p)).fold(0.0, |a, d| a + d)
    }

    /// リアルタイム画面の補助の列に置く「話者登録」の容器。
    pub(super) fn enroll_card(&mut self, ops: &[PathBuf], cx: &mut Context<Self>) -> Div {
        let total = self.enrolled_seconds(ops);
        let empty = ops.is_empty();
        let badge = if empty {
            div().label_m().text_color(c::error()).child("未登録")
        } else {
            div().flex().items_center().gap_1().child(icon("check", 16., c::tertiary())).child(div().label_m().text_color(c::on_surface_variant()).child("登録済み"))
        };
        let text = if empty {
            "変換の前に、あなたの普通の話し声を 30 秒ほど録音して登録します。".to_string()
        } else {
            format!("{}。録り直しや追加もできます。", count(ops.len(), total))
        };
        let (label, kind, ic) = if empty { ("登録する", Btn::Filled, "mic") } else { ("確認・追加", Btn::Tonal, "record_voice_over") };
        m3::card()
            .child(div().flex().items_center().gap_3().child(section_title("話者登録")).child(div().flex_1()).child(badge))
            .child(supporting(text))
            .child(div().flex().child(m3::button("enroll_open", Some(ic), label, kind, true).on_click(cx.listener(|this, _, window, cx| this.open_enroll(window, cx)))))
    }

    /// 窓の全体(暗幕 + 本体)。開いていなければ None。
    pub(super) fn render_enroll(&mut self, cx: &mut Context<Self>) -> Option<AnyElement> {
        if !self.enroll_open {
            return None;
        }
        if self.enroll.is_none() && !self.rt_running && !self.demo {
            self.enroll = Some(enroll::open(self.state.clone()));
        }
        let live = self.enroll_live();
        let phase = live.as_ref().map(|l| l.phase.clone());
        let recording = matches!(phase, Some(Phase::Recording));
        let ready = matches!(phase, Some(Phase::Monitoring));
        let ops = self.state.lock().unwrap().settings.operator.clone();

        let header = div().flex().items_start().gap_4()
            .child(m3::avatar("record_voice_over", c::primary_container(), c::on_primary_container()))
            .child(div().flex_1().min_w_0().flex().flex_col().gap_1()
                .child(div().headline_s().text_color(c::on_surface()).child("話者登録"))
                .child(div().body_m().text_color(c::on_surface_variant()).child(
                    "あなたの普通の話し声を 30 秒録音して、変換の基準にする声の特徴を測ります。モデルの学習ではないので、録音が終われば使えます。",
                )))
            .child(m3::icon_button("enroll_x", "close", false, c::on_surface_variant()).on_click(cx.listener(|this, _, _, cx| this.close_enroll(cx))));

        let body = div().id("enroll_body").flex_1().min_h_0().overflow_y_scroll().flex().flex_col().gap_5()
            .child(self.enroll_input(live.as_ref(), cx))
            .child(self.enroll_take(live.as_ref()))
            .child(self.enroll_list(&ops, recording, cx));

        let main_btn = if recording {
            m3::button("enroll_abort", Some("stop_fill1"), "録音を中止", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| {
                if let Some(s) = &this.enroll { s.abort_take(); }
                cx.notify();
            }))
        } else {
            let label = if ops.is_empty() { "録音を開始" } else { "もう 1 本録音" };
            m3::button("enroll_rec", Some("mic"), label, Btn::Filled, ready).on_click(cx.listener(|this, _, _, cx| this.start_take(cx)))
        };
        let footer = div().flex().flex_wrap().items_center().gap_2()
            .child(m3::button("enroll_wav", Some("folder_open"), "WAV から追加", Btn::Outlined, !recording).on_click(cx.listener(|this, _, _, cx| {
                if !this.enroll_live().is_some_and(|l| l.phase == Phase::Recording) { this.picks.operator.open(); }
                cx.notify();
            })))
            .child(div().flex_1())
            .child(div().body_s().text_color(c::on_surface_variant()).child(if recording { "Esc で中止" } else { "Enter で録音・Esc で閉じる" }))
            .child(m3::button("enroll_close", None, "閉じる", Btn::Text, !recording).on_click(cx.listener(|this, _, _, cx| {
                if !this.enroll_live().is_some_and(|l| l.phase == Phase::Recording) { this.close_enroll(cx); }
            })))
            .child(main_btn);

        let dialog = div().id("enroll_dialog").track_focus(&self.enroll_focus)
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _, cx| {
                let rec = this.enroll_live().is_some_and(|l| l.phase == Phase::Recording);
                match ev.keystroke.key.as_str() {
                    "escape" if rec => {
                        if let Some(s) = &this.enroll { s.abort_take(); }
                        cx.notify();
                    }
                    "escape" => this.close_enroll(cx),
                    "enter" if !rec => this.start_take(cx),
                    _ => {}
                }
            }))
            .w(px(640.)).max_w(relative(0.94)).max_h(relative(0.92)).flex().flex_col().gap_5().p_6()
            .rounded(px(shape::XL)).bg(c::surface_container_high()).shadow_lg()
            .child(header).child(body).child(footer)
            .with_animation("enroll_in", Animation::new(Duration::from_millis(220)), |el, d| {
                let e = 1.0 - (1.0 - d).powi(3);
                el.opacity(e).mt(px(16.0 * (1.0 - e)))
            });

        Some(div().id("enroll_scrim").absolute().inset_0().occlude().flex().items_center().justify_center().bg(scrim()).child(dialog).into_any_element())
    }

    /// 1. 入力の確認: 装置・波形・レベルと判定。
    fn enroll_input(&mut self, live: Option<&Live>, cx: &mut Context<Self>) -> Div {
        let sec = div().flex().flex_col().gap_3().child(div().title_m().text_color(c::on_surface()).child("1. 入力の確認"));
        let change = m3::button("enroll_dev", Some("settings"), "デバイスを変更", Btn::Text, true).on_click(cx.listener(|this, _, _, cx| {
            if this.enroll_live().is_some_and(|l| l.phase == Phase::Recording) { return; }
            this.close_enroll(cx);
            this.set_tab(Tab::Devices, cx);
        }));
        if self.rt_running && !self.demo {
            return sec
                .child(div().flex().items_start().gap_3()
                    .child(icon("info", 20., c::on_surface_variant()))
                    .child(div().flex_1().body_m().text_color(c::on_surface()).child("変換中は、マイクの確認と録音ができません(同じ入力装置を使うため)。変換を止めてから録音します。")))
                .child(div().flex().gap_2().child(m3::button("enroll_stop_rt", Some("stop_fill1"), "変換を止める", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| {
                    this.stop(cx);
                    this.reopen_enroll(cx);
                }))));
        }
        let Some(l) = live else {
            return sec.child(supporting("入力装置を開いています…"));
        };
        if let Phase::Failed(msg) = &l.phase {
            return sec
                .child(div().flex().items_start().gap_3()
                    .child(icon("error", 20., c::error()))
                    .child(div().flex_1().body_m().text_color(c::on_surface()).child(msg.clone())))
                .child(div().flex().gap_2()
                    .child(m3::button("enroll_retry", Some("refresh"), "もう一度試す", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| this.reopen_enroll(cx))))
                    .child(change));
        }
        let opening = l.phase == Phase::Opening;
        let device = div().flex().items_center().gap_3()
            .child(icon("mic", 20., c::on_surface_variant()))
            .child(div().flex_1().min_w_0().flex().flex_col()
                .child(div().body_l().text_color(c::on_surface()).truncate().child(if opening { "入力装置を開いています…".to_string() } else { l.device.clone() }))
                .when(!opening, |d| d.child(div().body_s().text_color(c::on_surface_variant()).child(format!("{}・{} kHz", l.host, l.sample_rate as f32 / 1000.0)))))
            .child(change);
        let (ic, text, col) = judge(l);
        let recording = l.phase == Phase::Recording;
        let axis = m3::alpha(c::on_surface(), 0.16);
        let panel = if recording {
            wave(l.take.clone(), (enroll::SECONDS as f32 / enroll::COL_S) as usize, 0, c::primary(), c::error(), axis).into_any_element()
        } else {
            wave(l.scope.iter().copied().collect(), enroll::SCOPE_COLS, enroll::SCOPE_COLS - l.scope.len(), c::primary(), c::error(), axis).into_any_element()
        };
        let caption = if recording {
            div().flex().items_center().gap_2()
                .child(m3::rec_dot())
                .child(div().label_l().text_color(c::on_surface()).child(format!("録音中 {:.0} / {} 秒", l.recorded.floor(), enroll::SECONDS)))
                .child(div().flex_1())
                .child(div().body_s().text_color(c::on_surface_variant()).child(format!("声が入った時間 {:.0} 秒", l.voiced)))
        } else {
            div().flex().items_center().gap_2()
                .child(div().label_m().text_color(c::on_surface_variant()).child("入力の波形(直近 3 秒)"))
        };
        let scope = div().flex().flex_col().gap_2().child(caption)
            .child(div().h(px(112.)).w_full().px_3().py_2().rounded(px(shape::LG)).bg(c::surface_container()).child(panel));
        let level = div().flex().items_center().gap_3()
            .child(div().label_m().flex_none().text_color(c::on_surface_variant()).child("レベル"))
            .child(div().flex_1().child(m3::meter((l.rms_db + 60.0) / 60.0, col, m3::alpha(c::on_surface(), 0.16))))
            .child(div().w(px(84.)).flex_none().label_l().text_color(c::on_surface()).child(if l.rms_db > -119.0 { format!("{:.0} dBFS", l.rms_db) } else { "—".into() }));
        let verdict = div().flex().items_start().gap_2()
            .child(icon(ic, 20., col))
            .child(div().flex_1().body_m().text_color(c::on_surface()).child(text));
        sec.child(device).child(scope).child(level).when(!opening, |s| s.child(verdict))
    }

    /// 2. 録音: 読み上げの例・進み・結果。
    fn enroll_take(&self, live: Option<&Live>) -> Div {
        let sec = div().flex().flex_col().gap_3().child(div().title_m().text_color(c::on_surface()).child("2. 録音"));
        let script = div().p_4().rounded(px(shape::LG)).bg(c::surface_container()).flex().flex_col().gap_2()
            .child(div().label_m().text_color(c::on_surface_variant()).child("読み上げの例(何を話してもかまいません。普段の声で)"))
            .child(div().body_l().text_color(c::on_surface()).child(SCRIPT));
        let Some(l) = live.filter(|_| !self.rt_running || self.demo) else {
            return sec.child(script);
        };
        let status: Div = match &l.phase {
            Phase::Recording => supporting("上の文を読むか、自由に話してください。30 秒で自動的に終わります。"),
            Phase::Saving => div().flex().items_center().gap_3().child(m3::loading_indicator(24.)).child(div().body_m().child("保存して登録しています…")),
            _ => match (&l.saved, &l.save_error) {
                (_, Some(e)) => div().flex().items_start().gap_2().child(icon("error", 20., c::error())).child(div().flex_1().body_m().text_color(c::on_surface()).child(e.clone())),
                (Some((p, voiced)), _) if *voiced < VOICED_MIN_S => div().flex().items_start().gap_2()
                    .child(icon("info", 20., c::on_surface_variant()))
                    .child(div().flex_1().body_m().text_color(c::on_surface()).child(format!(
                        "「{}」を登録しました。ただし声が入っていたのは {:.0} 秒で、少なめです。もう 1 本録ると変換が安定します。",
                        file_name(p), voiced
                    ))),
                (Some((p, voiced)), _) => div().flex().items_start().gap_2()
                    .child(icon("check", 20., c::tertiary()))
                    .child(div().flex_1().body_m().text_color(c::on_surface()).child(format!("「{}」を登録しました(声が入った時間 {:.0} 秒)。", file_name(p), voiced))),
                _ => supporting(format!("準備ができたら「録音を開始」を押し、上の判定が「ちょうどよい」になる声で {} 秒話してください。", enroll::SECONDS)),
            },
        };
        sec.child(script).child(status)
    }

    /// 登録済みの一覧(試聴・削除)。
    fn enroll_list(&mut self, ops: &[PathBuf], recording: bool, cx: &mut Context<Self>) -> Div {
        let total = self.enrolled_seconds(ops);
        let mut sec = div().flex().flex_col().gap_2().child(
            div().flex().items_center()
                .child(div().title_m().text_color(c::on_surface()).child("登録済み"))
                .child(div().flex_1())
                .child(div().body_s().text_color(c::on_surface_variant()).child(if ops.is_empty() { "なし".to_string() } else { count(ops.len(), total) })),
        );
        if ops.is_empty() {
            return sec.child(supporting("まだありません。録音するか、自分の声の WAV を追加します。"));
        }
        let mut list = m3::list();
        for (i, p) in ops.iter().enumerate() {
            let dur = self.duration_of(p).map_or("—".to_string(), |d| format!("{d:.0} 秒"));
            let playing = self.enroll_playing.as_deref() == Some(p.as_path()) && self.enroll_player.as_ref().is_some_and(|pl| pl.is_playing());
            let path = p.clone();
            list = list.child(
                m3::list_item(SharedString::from(format!("enroll_item_{i}")), false, i == 0, i == ops.len() - 1)
                    .child(icon("audio_file", 20., c::on_surface_variant()))
                    .child(div().flex_1().min_w_0().body_m().text_color(c::on_surface()).truncate().child(file_name(p)))
                    .child(div().label_m().flex_none().text_color(c::on_surface_variant()).child(dur))
                    .child(m3::icon_button(SharedString::from(format!("enroll_play_{i}")), if playing { "stop_fill1" } else { "play_arrow_fill1" }, false, c::on_surface_variant())
                        .on_click(cx.listener(move |this, _, _, cx| {
                            let was = this.enroll_playing.as_deref() == Some(path.as_path()) && this.enroll_player.as_ref().is_some_and(|pl| pl.is_playing());
                            stop_player(&mut this.enroll_player);
                            this.enroll_playing = None;
                            if !was && !this.enroll_live().is_some_and(|l| l.phase == Phase::Recording) {
                                match play_path(&mut this.enroll_player, &path) {
                                    Ok(()) => this.enroll_playing = Some(path.clone()),
                                    Err(e) => this.state.lock().unwrap().error = Some(format!("再生できません: {e}")),
                                }
                            }
                            cx.notify();
                        })))
                    .when(!recording, |row| row.child(m3::icon_button(SharedString::from(format!("enroll_del_{i}")), "delete", false, c::on_surface_variant())
                        .on_click(cx.listener(move |this, _, _, cx| {
                            stop_player(&mut this.enroll_player);
                            this.enroll_playing = None;
                            let mut s = this.state.lock().unwrap();
                            if i < s.settings.operator.len() { s.settings.operator.remove(i); }
                            cx.notify();
                        })))),
            );
        }
        sec = sec.child(list);
        sec
    }
}
