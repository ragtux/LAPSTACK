// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

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
//! registration, `io` bit-depth-preserving TIFF/PNG/JPEG I/O, `view` synthetic
//! stereo, `mesh` the textured 3D model, and `stack::run` the decode → align →
//! fuse → depth pipeline.

pub mod align;
pub mod brightness;
pub mod depth;
pub mod fuse;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod io;
pub mod mesh;
pub mod meta;
pub mod pyramid;
pub mod stack;
pub mod view;

pub use align::{AlignParams, CancelToken, Cancelled, Sim};
pub use depth::{DepthMap, DepthParams, FocusMeasure, Upsample};
pub use fuse::{FuseParams, Fuser, TopRule};
pub use io::Depth;
pub use mesh::{Mesh, MeshParams, TexFormat};
pub use pyramid::Img3;
pub use stack::{FrameSource, Output, Params, Slab, run, run_with, slab_ranges};
pub use view::{Layout, View};
