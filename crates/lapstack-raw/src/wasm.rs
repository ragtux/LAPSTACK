// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LGPL-2.1-only

//! The wasm module (`web/pkg-raw`). A developed image stays in this module's
//! memory: `RawImage` hands out its pointer and length, the worker makes a
//! typed-array view on the module's memory with them, the engine copies from
//! that view into its own memory, and then the `RawImage` is freed — one copy,
//! no intermediate. The view is only good until this module's memory grows,
//! which nothing does between the call and the copy.

use wasm_bindgen::prelude::*;

use crate::{Image, Pixels};

#[wasm_bindgen]
pub struct RawImage {
    img: Image,
}

#[wasm_bindgen]
impl RawImage {
    #[wasm_bindgen(getter)]
    pub fn w(&self) -> u32 {
        self.img.w as u32
    }
    #[wasm_bindgen(getter)]
    pub fn h(&self) -> u32 {
        self.img.h as u32
    }
    /// 1 (gray) or 3.
    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u32 {
        self.img.px.channels()
    }
    /// 8 (RGB8), 16 (gray or RGB u16) or 32 (three f32 planes).
    #[wasm_bindgen(getter)]
    pub fn bits(&self) -> u32 {
        self.img.px.bits()
    }
    #[wasm_bindgen(getter)]
    pub fn turns(&self) -> u8 {
        self.img.turns
    }
    #[wasm_bindgen(getter)]
    pub fn flip(&self) -> bool {
        self.img.flip
    }
    /// The camera's color as JSON (`develop_linear`), else undefined.
    #[wasm_bindgen(getter)]
    pub fn color(&self) -> Option<String> {
        self.img.color.as_ref().and_then(|c| serde_json::to_string(c).ok())
    }
    /// The pixels' address in this module's memory (a byte offset, aligned to
    /// the element).
    pub fn ptr(&self) -> u32 {
        match &self.img.px {
            Pixels::Gray16(v) | Pixels::Rgb16(v) => v.as_ptr() as u32,
            Pixels::Rgb8(v) => v.as_ptr() as u32,
            Pixels::Planes3(v) => v.as_ptr() as u32,
        }
    }
    /// The pixels' length in elements (u8, u16 or f32 as `bits` says).
    pub fn len(&self) -> u32 {
        match &self.img.px {
            Pixels::Gray16(v) | Pixels::Rgb16(v) => v.len() as u32,
            Pixels::Rgb8(v) => v.len() as u32,
            Pixels::Planes3(v) => v.len() as u32,
        }
    }
}

#[wasm_bindgen]
pub fn abi() -> u32 {
    crate::ABI
}

#[wasm_bindgen]
pub fn develop(bytes: &[u8]) -> Result<RawImage, JsError> {
    crate::develop(bytes).map(|img| RawImage { img }).map_err(|e| JsError::new(&e))
}

#[wasm_bindgen]
pub fn develop_linear(bytes: &[u8]) -> Result<RawImage, JsError> {
    crate::develop_linear(bytes).map(|img| RawImage { img }).map_err(|e| JsError::new(&e))
}

#[wasm_bindgen]
pub fn preview(bytes: &[u8]) -> Option<RawImage> {
    crate::preview(bytes).map(|img| RawImage { img })
}

/// The file's metadata as JSON (`Metadata`), or undefined.
#[wasm_bindgen]
pub fn metadata(bytes: &[u8]) -> Option<String> {
    crate::metadata(bytes).and_then(|m| serde_json::to_string(&m).ok())
}
