// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! A global allocator that keeps large blocks for reuse.
//!
//! Every stage of the fold makes fresh planes — the pyramid's levels, the
//! warp, the luma, the energies, the focus slice, the decoder's buffers —
//! and glibc hands a block above its mmap threshold (32 MB at most) back to
//! the kernel on free, so the next frame maps it afresh and takes the page
//! faults again: with 128 threads first-touching a 45 MB plane at once, that
//! is ~150 ms per plane where the arithmetic is 20 ms (`depth.rs` measured it
//! on the box filter). This allocator wraps the system one and, for blocks
//! of `BIG` bytes or more, keeps the freed ones in a small table and serves a
//! later request of the same size class from there; the pages stay mapped
//! and warm. Zeroed requests on a recycled block are cleared with `memset`,
//! ~4 ms per 45 MB. The table holds at most `SLOTS` blocks and `CAP` bytes;
//! beyond that, blocks go back to the system as before.
//!
//! Installed by the CLI (`#[global_allocator]`); the browser build does not
//! use it (wasm has no page faults to save).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// The pool can be switched off (`LAPSTACK_NO_POOL=1` in the CLI), for a
/// comparison; blocks already pooled are still served.
pub static DISABLED: AtomicBool = AtomicBool::new(false);

/// Blocks at least this large are pooled.
const BIG: usize = 4 << 20;
const SLOTS: usize = 64;
/// Bytes the pool may hold idle.
const CAP: usize = 4 << 30;

struct Table {
    n: usize,
    bytes: usize,
    blocks: [(usize, usize, usize); SLOTS], // (size, align, ptr)
}

pub struct PoolAlloc {
    table: Mutex<Table>,
}

impl PoolAlloc {
    pub const fn new() -> PoolAlloc {
        PoolAlloc { table: Mutex::new(Table { n: 0, bytes: 0, blocks: [(0, 0, 0); SLOTS] }) }
    }

    /// A pooled block that fits `layout` (same alignment, its size and at
    /// most a quarter more, so a plane never squats on a much larger block).
    fn take(&self, layout: Layout) -> Option<(*mut u8, usize)> {
        let mut t = self.table.lock().ok()?;
        let want = layout.size();
        let mut best: Option<usize> = None;
        for i in 0..t.n {
            let (sz, al, _) = t.blocks[i];
            if al == layout.align() && sz >= want && sz <= want + want / 4 && best.is_none_or(|b| sz < t.blocks[b].0) {
                best = Some(i);
            }
        }
        let i = best?;
        let (sz, _, p) = t.blocks[i];
        t.n -= 1;
        t.blocks[i] = t.blocks[t.n];
        t.bytes -= sz;
        Some((p as *mut u8, sz))
    }

    /// Keep a freed block; `false` when the table is full.
    fn put(&self, ptr: *mut u8, layout: Layout) -> bool {
        let Ok(mut t) = self.table.lock() else { return false };
        if t.n == SLOTS || t.bytes + layout.size() > CAP {
            return false;
        }
        let n = t.n;
        t.blocks[n] = (layout.size(), layout.align(), ptr as usize);
        t.n += 1;
        t.bytes += layout.size();
        true
    }
}

impl Default for PoolAlloc {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for PoolAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= BIG && let Some((p, _)) = self.take(layout) {
            return p;
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= BIG && let Some((p, _)) = self.take(layout) {
            unsafe { std::ptr::write_bytes(p, 0, layout.size()) };
            return p;
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() >= BIG && !DISABLED.load(Ordering::Relaxed) && self.put(ptr, layout) {
            return;
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.size() < BIG && new_size < BIG {
            return unsafe { System.realloc(ptr, layout, new_size) };
        }
        // through the pool: a new block, the bytes copied, the old one kept
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let p = unsafe { self.alloc(new_layout) };
        if !p.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, p, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_blocks_come_back_from_the_pool() {
        let a = PoolAlloc::new();
        let l = Layout::from_size_align(BIG, 4).unwrap();
        let p = unsafe { a.alloc(l) };
        assert!(!p.is_null());
        unsafe { std::ptr::write_bytes(p, 7, BIG) };
        unsafe { a.dealloc(p, l) };
        assert_eq!(a.table.lock().unwrap().n, 1);
        // the same size class again: the same block, and zeroed when asked
        let q = unsafe { a.alloc_zeroed(l) };
        assert_eq!(q, p);
        assert!((0..BIG).step_by(4093).all(|i| unsafe { *q.add(i) } == 0));
        assert_eq!(a.table.lock().unwrap().n, 0);
        unsafe { a.dealloc(q, l) };
        // a different alignment or a much smaller request does not take it
        let l2 = Layout::from_size_align(BIG, 8).unwrap();
        let r = unsafe { a.alloc(l2) };
        assert_ne!(r, p);
        unsafe { a.dealloc(r, l2) };
        let l3 = Layout::from_size_align(BIG / 2, 4).unwrap();
        assert!(a.take(l3).is_none());
        // a block up to a quarter larger serves a request, the smallest fit first
        let l5 = Layout::from_size_align(BIG + BIG / 5, 4).unwrap();
        let big = unsafe { a.alloc(l5) };
        assert_ne!(big, p);
        unsafe { a.dealloc(big, l5) };
        let s = unsafe { a.alloc(l) };
        assert_eq!(s, p);
        let s3 = unsafe { a.alloc(l) };
        assert_eq!(s3, big);
        unsafe { a.dealloc(s3, l5) };
        // realloc goes through the pool and keeps the bytes
        unsafe { *s = 42 };
        let s2 = unsafe { a.realloc(s, l, BIG + BIG / 2) };
        assert_eq!(unsafe { *s2 }, 42);
        unsafe { a.dealloc(s2, Layout::from_size_align(BIG + BIG / 2, 4).unwrap()) };
        // small blocks never enter the table
        let ls = Layout::from_size_align(1024, 8).unwrap();
        let t = unsafe { a.alloc(ls) };
        unsafe { a.dealloc(t, ls) };
        assert!(a.table.lock().unwrap().n <= 4);
        // drain what the test left
        let mut tb = a.table.lock().unwrap();
        for i in 0..tb.n {
            let (sz, al, ptr) = tb.blocks[i];
            unsafe { System.dealloc(ptr as *mut u8, Layout::from_size_align(sz, al).unwrap()) };
        }
        tb.n = 0;
    }
}
