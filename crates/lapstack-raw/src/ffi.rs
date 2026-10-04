// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LGPL-2.1-only

//! The C ABI of the shared library (`liblapstack_raw.so`, `.dylib`,
//! `lapstack_raw.dll`), which lapstack loads at run time. Every call takes the
//! raw file's bytes; an image comes back in an `LrImage` the caller must give
//! back to `lapstack_raw_free_image`, a string in memory the caller must give
//! back to `lapstack_raw_free_str`. Nothing here unwinds into the caller: a
//! panic inside rawler is caught and reported as an error string.

use std::ffi::{CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::{Image, Pixels};

/// A developed image. `data` is `len` bytes of pixels in the layout `bits` and
/// `channels` say (see the crate doc: 16-bit gray or RGB interleaved, 8-bit RGB
/// interleaved, or three f32 planes when `bits` is 32); `color` is the JSON of
/// the camera's color (`develop_linear`) or null. `handle` is the library's own.
#[repr(C)]
pub struct LrImage {
    pub w: u32,
    pub h: u32,
    pub channels: u32,
    pub bits: u32,
    pub turns: u8,
    pub flip: u8,
    pub _pad: [u8; 6],
    pub data: *const u8,
    pub len: usize,
    pub color: *const c_char,
    handle: *mut c_void,
}

struct Owned {
    #[allow(dead_code)]
    px: Pixels,
    #[allow(dead_code)]
    color: Option<CString>,
}

fn c_string(s: String) -> *mut c_char {
    CString::new(s.replace('\0', " ")).map(CString::into_raw).unwrap_or(std::ptr::null_mut())
}

unsafe fn fill(out: *mut LrImage, img: Image) {
    let color = img.color.as_ref().and_then(|c| serde_json::to_string(c).ok()).and_then(|s| CString::new(s).ok());
    let owned = Box::new(Owned { px: img.px, color });
    let bytes = owned.px.bytes();
    let o = LrImage {
        w: img.w as u32,
        h: img.h as u32,
        channels: owned.px.channels(),
        bits: owned.px.bits(),
        turns: img.turns,
        flip: img.flip as u8,
        _pad: [0; 6],
        data: bytes.as_ptr(),
        len: bytes.len(),
        color: owned.color.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
        handle: Box::into_raw(owned) as *mut c_void,
    };
    unsafe { out.write(o) };
}

/// The contract's revision (`crate::ABI`): a loader refuses a library whose
/// number is not the one it was written against.
#[unsafe(no_mangle)]
pub extern "C" fn lapstack_raw_abi() -> u32 {
    crate::ABI
}

/// `develop` (`linear` = 0) or `develop_linear` (`linear` = 1) of the file in
/// `bytes[..len]` into `*out`. Returns null on success, else an error string.
///
/// # Safety
/// `bytes` must point to `len` readable bytes and `out` to writable space for
/// an `LrImage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapstack_raw_develop(bytes: *const u8, len: usize, linear: u32, out: *mut LrImage) -> *mut c_char {
    let input = unsafe { std::slice::from_raw_parts(bytes, len) };
    let r = catch_unwind(AssertUnwindSafe(|| if linear != 0 { crate::develop_linear(input) } else { crate::develop(input) }));
    match r {
        Ok(Ok(img)) => {
            unsafe { fill(out, img) };
            std::ptr::null_mut()
        }
        Ok(Err(e)) => c_string(e),
        Err(_) => c_string("raw: the decoder panicked on this file".into()),
    }
}

/// The camera's preview of the file into `*out`. Returns 1 when there is one,
/// 0 when there is none (`*out` untouched).
///
/// # Safety
/// As `lapstack_raw_develop`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapstack_raw_preview(bytes: *const u8, len: usize, out: *mut LrImage) -> u32 {
    let input = unsafe { std::slice::from_raw_parts(bytes, len) };
    match catch_unwind(AssertUnwindSafe(|| crate::preview(input))) {
        Ok(Some(img)) => {
            unsafe { fill(out, img) };
            1
        }
        _ => 0,
    }
}

/// The file's metadata as JSON (`Metadata`), or null when the file is not a
/// raw rawler reads.
///
/// # Safety
/// `bytes` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapstack_raw_metadata(bytes: *const u8, len: usize) -> *mut c_char {
    let input = unsafe { std::slice::from_raw_parts(bytes, len) };
    match catch_unwind(AssertUnwindSafe(|| crate::metadata(input))) {
        Ok(Some(m)) => serde_json::to_string(&m).map(c_string).unwrap_or(std::ptr::null_mut()),
        _ => std::ptr::null_mut(),
    }
}

/// Give an image back.
///
/// # Safety
/// `img` must have been filled by this library and not freed since.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapstack_raw_free_image(img: *mut LrImage) {
    if img.is_null() {
        return;
    }
    let i = unsafe { &mut *img };
    if !i.handle.is_null() {
        drop(unsafe { Box::from_raw(i.handle as *mut Owned) });
    }
    i.handle = std::ptr::null_mut();
    i.data = std::ptr::null();
    i.color = std::ptr::null();
    i.len = 0;
}

/// Give a string back.
///
/// # Safety
/// `s` must have come from this library and not been freed since.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapstack_raw_free_str(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}
