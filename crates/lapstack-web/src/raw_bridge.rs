// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! The engine's way to the raw decoder, which is a wasm module of its own
//! (`web/pkg-raw`, `crates/lapstack-raw`, LGPL) that the worker loads: three
//! functions the worker puts on the global object. `lapstackRawDevelop` returns
//! `{w, h, channels, bits, turns, flip, color, data, free}` with `data` a typed
//! array viewing the module's memory; the pixels are copied from it into ours
//! here and `free()` gives the module's copy back. A thrown error is the
//! decoder's message. Installed as `lapstack_core::raw`'s backend when this
//! module starts.

use js_sys::{Float32Array, Function, Reflect, Uint8Array, Uint16Array};
use lapstack_core::raw::{Pixels, RawBackend, RawColor, RawImage, RawMeta};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = globalThis, js_name = lapstackRawDevelop, catch)]
    fn js_develop(bytes: &[u8], linear: bool) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(js_namespace = globalThis, js_name = lapstackRawPreview, catch)]
    fn js_preview(bytes: &[u8]) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(js_namespace = globalThis, js_name = lapstackRawMetadata, catch)]
    fn js_metadata(bytes: &[u8]) -> Result<JsValue, JsValue>;
}

pub struct JsBackend;

fn message(e: JsValue) -> String {
    if let Some(s) = e.as_string() {
        return s;
    }
    if e.is_instance_of::<js_sys::Error>() {
        return String::from(js_sys::Error::from(e).message());
    }
    format!("{e:?}")
}

fn num(o: &JsValue, k: &str) -> Result<usize, String> {
    Reflect::get(o, &JsValue::from_str(k)).ok().and_then(|v| v.as_f64()).map(|v| v as usize).ok_or_else(|| format!("raw: the decoder's answer has no {k}"))
}

/// The image out of the worker's answer, its pixels copied into our memory.
fn take(o: JsValue) -> Result<RawImage, String> {
    if o.is_null() || o.is_undefined() {
        return Err("raw: the decoder returned nothing".into());
    }
    let (w, h, channels, bits) = (num(&o, "w")?, num(&o, "h")?, num(&o, "channels")?, num(&o, "bits")?);
    let turns = num(&o, "turns").unwrap_or(0) as u8;
    let flip = Reflect::get(&o, &JsValue::from_str("flip")).ok().and_then(|v| v.as_bool()).unwrap_or(false);
    let data = Reflect::get(&o, &JsValue::from_str("data")).map_err(|_| "raw: the decoder's answer has no data".to_string())?;
    let px = match (bits, channels) {
        (16, 1) => Pixels::Gray16(Uint16Array::from(data).to_vec()),
        (16, 3) => Pixels::Rgb16(Uint16Array::from(data).to_vec()),
        (8, 3) => Pixels::Rgb8(Uint8Array::from(data).to_vec()),
        (32, 3) => {
            let a = Float32Array::from(data);
            let n = (w * h) as u32;
            if a.length() != 3 * n {
                return Err("raw: the decoder returned planes of the wrong size".into());
            }
            Pixels::Planes3([a.subarray(0, n).to_vec(), a.subarray(n, 2 * n).to_vec(), a.subarray(2 * n, 3 * n).to_vec()])
        }
        (b, c) => return Err(format!("raw: the decoder returned {b}-bit, {c}-channel pixels")),
    };
    let color = match Reflect::get(&o, &JsValue::from_str("color")).ok().and_then(|v| v.as_string()) {
        Some(s) => Some(serde_json::from_str::<RawColor>(&s).map_err(|e| format!("raw: the decoder's color description does not parse: {e}"))?),
        None => None,
    };
    if let Ok(f) = Reflect::get(&o, &JsValue::from_str("free")) {
        if let Ok(f) = f.dyn_into::<Function>() {
            let _ = f.call0(&JsValue::NULL);
        }
    }
    Ok(RawImage { w, h, px, turns, flip, color })
}

impl RawBackend for JsBackend {
    fn develop(&self, bytes: &[u8], linear: bool) -> Result<RawImage, String> {
        take(js_develop(bytes, linear).map_err(message)?)
    }
    fn preview(&self, bytes: &[u8]) -> Option<RawImage> {
        let o = js_preview(bytes).ok()?;
        if o.is_null() || o.is_undefined() {
            return None;
        }
        take(o).ok()
    }
    fn metadata(&self, bytes: &[u8]) -> Option<RawMeta> {
        let s = js_metadata(bytes).ok()?.as_string()?;
        serde_json::from_str(&s).ok()
    }
    fn describe(&self) -> String {
        "raw decoder: pkg-raw (the worker's module)".into()
    }
}

#[wasm_bindgen(start)]
fn start() {
    lapstack_core::raw::set_backend(Box::new(JsBackend));
}
