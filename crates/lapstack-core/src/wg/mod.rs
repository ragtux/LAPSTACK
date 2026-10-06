// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! The wgpu engine: lapstack's fusion, alignment and depth pass as WGSL
//! compute kernels (`shaders.wgsl`) over one small wgpu layer (`gpu.rs`).
//! The browser app runs them on WebGPU (`lapstack-web` builds its engine on
//! these modules); natively the same kernels run on Vulkan, Metal or DX12
//! for the CLI's `--gpu` where there is no CUDA (`engine.rs`). One kernel set,
//! so the browser and the native GPU path cannot drift apart.

pub mod align;
pub mod depth;
pub mod fold;
pub mod gpu;
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
