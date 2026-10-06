// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! lapstack-core — focus stacking on the Laplacian pyramid, written from the
//! papers:
//!
//! * P. Burt & E. Adelson, "The Laplacian Pyramid as a Compact Image Code"
//!   (1983) and E. Adelson et al., "Pyramid methods in image processing"
//!   (RCA Engineer, 1984): REDUCE/EXPAND, the band-pass decomposition and the
//!   "multifocus composite" (pick the node with the larger magnitude, then
//!   expand-and-add; blending happens in the reconstruction).
//! * W. Wang & F. Chang, "A Multi-focus Image Fusion Method Based on
//!   Laplacian Pyramid" (J. Computers 6(12), 2011): the 5×5 binomial
//!   generating kernel, *maximum region energy* selection for the band-pass
//!   levels and a local deviation + entropy rule for the residual.
//!
//! `pyramid` is the transform, `fuse` the rules and the N-frame accumulator,
//! `depth` the depth-from-focus pass (ring difference filter, guided-filter
//! aggregation, sub-frame peaks, edge-aware WLS), `align` the 4-DOF similarity
//! registration, `dust` the dust map, `io` bit-depth-preserving TIFF/PNG/JPEG I/O, `view` synthetic
//! stereo, `mesh` the textured 3D model, `overlay` the scale bar and caption
//! burned into the saved images, `stack::run` the decode → align →
//! fuse → depth pipeline, and `batch` the rules that cut a list of frames into
//! stacks for a batch of runs.

pub mod align;
pub mod batch;
pub mod brightness;
pub mod depth;
pub mod dng;
pub mod dust;
pub mod fuse;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod pool;
pub mod project;
#[cfg(feature = "wgpu")]
pub mod wg;
pub mod prep;
pub mod io;
pub mod mesh;
pub mod meta;
pub mod overlay;
pub mod raw;
pub mod pyramid;
pub mod stack;
pub mod view;
pub mod wav;

pub use align::{AlignParams, Interp, Sim};
pub use batch::{Split, Stack};
pub use depth::{DepthMap, DepthParams, FocusMeasure, Upsample};
pub use dng::DngInfo;
pub use dust::{DustMap, DustMode, DustParams};
pub use fuse::{FuseParams, Fuser, TopRule};
pub use io::Depth;
pub use mesh::{Mesh, MeshParams, TexFormat};
pub use overlay::{Overlay, OverlayParams};
pub use pyramid::Img3;
pub use stack::{FrameSource, Output, Params, Slab, run, run_with, slab_ranges};
pub use view::{Layout, View};
