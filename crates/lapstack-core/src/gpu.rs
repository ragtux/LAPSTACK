// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! CUDA fusion path (feature `gpu`). Same maths as `pyramid.rs` + `fuse.rs`
//! — the binomial taps, reflect-101 borders, luma energy, binomial window and
//! winner-take-all select are transcribed kernel for kernel — so the output
//! matches the CPU path to float rounding. Per frame only the three RGB
//! planes cross the bus up and the residual level (a few KB) comes back; the
//! accumulator pyramid, best-energy planes and winner map live on the device.
//! The residual rule (deviation + entropy over N tiny planes) stays on the CPU.
//! With halo control (`fuse.rs`) the guide's weights are made, REDUCEd and
//! folded on the device too (`wgtk`, `wacck`, `wnormk`), the residual included.
//!
//! The depth pass has its twin here too (`depth_from_slices`, the kernels
//! of `DEPTH_SRC`): the guided-filter aggregation of the slices, the peak
//! tracker, the WLS solve with its multigrid hierarchy and the upsampling
//! all run on the device, the slices uploaded one at a time.
//!
//! cudarc with `dynamic-loading` (libcuda + libnvrtc found at runtime, no
//! toolkit at build time), kernels compiled by NVRTC once per run (each
//! module — fusion + frames, depth — its own source).
//!
//! `GpuFrames` keeps the aligned stream's frames on the device as well
//! (`--gpu-align`): luma, the registration pyramids, the Nelder-Mead cost
//! search (one `warp_cost` launch per evaluation, on the CPU search's
//! per-level schedule, `align::level_steps`), the warp, the brightness
//! gains and the depth pass's focus slice, the warped planes handed to
//! `GpuFuser::push_device` in place. Only the Nelder-Mead control flow and
//! the decode stay on the host.

use crate::align::{Interp, Sim, inverse, level_steps, nelder_mead};
use crate::depth::{DepthParams, FocusMeasure, blocks, rdf_taps};
use crate::fuse::{FuseParams, HALO_FLOOR, HALO_REF, binomial, fuse_residuals, halo_guide, upsample_index};
use crate::pyramid::{Img3, auto_levels, half};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Helpers both modules use: reflect-101 and the block mean.
const COMMON: &str = r#"
__device__ __forceinline__ int refl(int i, int n){
    if(n==1) return 0;
    while(i<0 || i>=n){ if(i<0) i=-i; else i=2*(n-1)-i; }
    return i;
}
// block mean to the depth pass's working grid (depth::block_mean)
extern "C" __global__ void bmeank(const float* in,float* out,int w,int h,int k,int dw,int dh){
    int ox=blockIdx.x*blockDim.x+threadIdx.x, oy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ox>=dw||oy>=dh) return; int x0=ox*k, x1=min(x0+k,w), y0=oy*k, y1=min(y0+k,h); float a=0.f;
    for(int y=y0;y<y1;y++) for(int x=x0;x<x1;x++) a+=in[(size_t)y*w+x];
    out[(size_t)oy*dw+ox]=a/(float)((y1-y0)*(x1-x0));
}
"#;

const SRC: &str = r#"
#define K0 (1.0f/16.0f)
#define K1 (4.0f/16.0f)
#define K2 (6.0f/16.0f)
#define E0 (2.0f*K0)
#define E1 (2.0f*K2)
#define OD (2.0f*K1)
// REDUCE, horizontal: tmp (h x ow) = 5-tap at even columns
extern "C" __global__ void redH(const float* in,float* tmp,int w,int h,int ow){
    int oj=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(oj>=ow||y>=h) return; int c=2*oj; const float* r=in+(size_t)y*w;
    tmp[(size_t)y*ow+oj]= K0*r[refl(c-2,w)]+K1*r[refl(c-1,w)]+K2*r[refl(c,w)]+K1*r[refl(c+1,w)]+K0*r[refl(c+2,w)];
}
// REDUCE, vertical: out (oh x ow)
extern "C" __global__ void redV(const float* tmp,float* out,int h,int ow,int oh){
    int oj=blockIdx.x*blockDim.x+threadIdx.x, oi=blockIdx.y*blockDim.y+threadIdx.y;
    if(oj>=ow||oi>=oh) return; int c=2*oi;
    out[(size_t)oi*ow+oj]= K0*tmp[(size_t)refl(c-2,h)*ow+oj]+K1*tmp[(size_t)refl(c-1,h)*ow+oj]
        +K2*tmp[(size_t)refl(c,h)*ow+oj]+K1*tmp[(size_t)refl(c+1,h)*ow+oj]+K0*tmp[(size_t)refl(c+2,h)*ow+oj];
}
// EXPAND, horizontal: tmp (ch x ow) from coarse (ch x cw)
extern "C" __global__ void expH(const float* c,float* tmp,int cw,int ch,int ow){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=ow||y>=ch) return; const float* r=c+(size_t)y*cw; int i=x/2; float v;
    if((x&1)==0) v=E0*r[refl(i-1,cw)]+E1*r[refl(i,cw)]+E0*r[refl(i+1,cw)];
    else v=OD*(r[refl(i,cw)]+r[refl(i+1,cw)]);
    tmp[(size_t)y*ow+x]=v;
}
// EXPAND, vertical: out (oh x ow) from tmp (ch x ow)
extern "C" __global__ void expV(const float* tmp,float* out,int ch,int ow,int oh){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=ow||y>=oh) return; int i=y/2; float v;
    if((y&1)==0) v=E0*tmp[(size_t)refl(i-1,ch)*ow+x]+E1*tmp[(size_t)refl(i,ch)*ow+x]+E0*tmp[(size_t)refl(i+1,ch)*ow+x];
    else v=OD*(tmp[(size_t)refl(i,ch)*ow+x]+tmp[(size_t)refl(i+1,ch)*ow+x]);
    out[(size_t)y*ow+x]=v;
}
extern "C" __global__ void subk(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]-=b[i]; }
extern "C" __global__ void addk(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
extern "C" __global__ void clampk(float* a,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]=fminf(fmaxf(a[i],0.f),1.f); }
extern "C" __global__ void energy_y(const float* r,const float* g,const float* b,float* e,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float y=0.299f*r[i]+0.587f*g[i]+0.114f*b[i]; e[i]=y*y;
}
extern "C" __global__ void energy_rgb(const float* r,const float* g,const float* b,float* e,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    e[i]=r[i]*r[i]+g[i]*g[i]+b[i]*b[i];
}
// separable window sum with weights wt[klen] (klen odd), reflect-101
extern "C" __global__ void winH(const float* in,float* out,int w,int h,const float* wt,int klen){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; int r=klen/2; const float* row=in+(size_t)y*w; float a=0.f;
    for(int t=0;t<klen;t++) a+=wt[t]*row[refl(x+t-r,w)];
    out[(size_t)y*w+x]=a;
}
extern "C" __global__ void winV(const float* in,float* out,int w,int h,const float* wt,int klen){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; int r=klen/2; float a=0.f;
    for(int t=0;t<klen;t++) a+=wt[t]*in[(size_t)refl(y+t-r,h)*w+x];
    out[(size_t)y*w+x]=a;
}
// winner-take-all: where en > best, take the new coefficients (+ record idx)
extern "C" __global__ void selk(const float* en,float* best,float* a0,float* a1,float* a2,
        const float* n0,const float* n1,const float* n2,float* win,float idx,int rec,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    if(en[i]>best[i]){ best[i]=en[i]; a0[i]=n0[i]; a1[i]=n1[i]; a2[i]=n2[i]; if(rec) win[i]=idx; }
}
// halo control: the guide's weights ((en + floor) / ref)^p, exponent clamped to +-60
extern "C" __global__ void wgtk(const float* en,float* w,float p,float floor_,float ref,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    w[i]=expf(fminf(fmaxf(p*logf((en[i]+floor_)/ref),-60.f),60.f));
}
// weighted fold: acc += w * new, wsum += w
extern "C" __global__ void wacck(const float* w,float* a0,float* a1,float* a2,
        const float* n0,const float* n1,const float* n2,float* ws,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float k=w[i]; a0[i]+=k*n0[i]; a1[i]+=k*n1[i]; a2[i]+=k*n2[i]; ws[i]+=k;
}
// the weighted mean: acc /= wsum
extern "C" __global__ void wnormk(float* a0,float* a1,float* a2,const float* ws,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float k=1.f/ws[i]; a0[i]*=k; a1[i]*=k; a2[i]*=k;
}
// ---- alignment cost (fused Spline4x4 warp + DC-removed-RMS reduction) ----
// Warp `tgt` into `ref`'s frame with the inverse affine (ia..ity), and accumulate,
// over valid (in-bounds) pixels, sum(d), sum(d^2), count of d=ref-warp. The host
// turns these into rms = sqrt((Sd2 - Sd^2/n)/n) — one kernel + one readback per
// Nelder-Mead eval. Matches align.rs warp_plane + dc_removed_rms.
// FP32 per-pixel (consumer FP64 is ~1/32 rate; FP32 coords give ~0.001 px at 8K).
__device__ __forceinline__ void spl4f(float t,float* w){
    w[0]=((-1.f/3.f*t+0.8f)*t-0.46666667f)*t;
    w[1]=((t-1.8f)*t-0.2f)*t+1.f;
    w[2]=((1.2f-t)*t+0.8f)*t;
    w[3]=((1.f/3.f*t-0.2f)*t-0.13333334f)*t;
}
extern "C" __global__ void warp_cost(const float* tgt,int tw,int th,
        const float* ref,int aw,int ah,
        float ia,float ib,float itx,float id_,float ie,float ity,float ig0,float ig1,double* out){
    // float block reduction (d in [-1,1], so sums are small & exact enough), then
    // one double atomicAdd per block keeps the global sum accurate over millions of px.
    __shared__ float sd[256], sd2[256], sn[256];
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    int t=threadIdx.y*blockDim.x+threadIdx.x;
    float ld=0.f, ld2=0.f, ln=0.f;
    if(x<aw && y<ah){
        float den=ig0*x+ig1*y+1.f;   // the homography's divisor (exactly 1 for an affine transform)
        float sx=(ia*x+ib*y+itx)/den, sy=(id_*x+ie*y+ity)/den;
        if(sx>=0.f && sx<=(float)(tw-1) && sy>=0.f && sy<=(float)(th-1)){
            int x0=(int)floorf(sx), y0=(int)floorf(sy);
            float wx[4],wy[4]; spl4f(sx-x0,wx); spl4f(sy-y0,wy);
            float acc=0.f;
            for(int j=0;j<4;j++){ int yy=min(max(y0+j-1,0),th-1); float r=0.f;
                for(int i=0;i<4;i++){ int xx=min(max(x0+i-1,0),tw-1); r+=wx[i]*tgt[yy*tw+xx]; }
                acc+=wy[j]*r; }
            float d=ref[y*aw+x]-acc;
            ld=d; ld2=d*d; ln=1.f;
        }
    }
    sd[t]=ld; sd2[t]=ld2; sn[t]=ln; __syncthreads();
    for(int s=(blockDim.x*blockDim.y)>>1; s>0; s>>=1){
        if(t<s){ sd[t]+=sd[t+s]; sd2[t]+=sd2[t+s]; sn[t]+=sn[t+s]; } __syncthreads();
    }
    if(t==0){ atomicAdd(&out[0],(double)sd[0]); atomicAdd(&out[1],(double)sd2[0]); atomicAdd(&out[2],(double)sn[0]); }
}
// ---- device-resident frames (GpuFrames): luma, the registration pyramid, the warp,
// the brightness gains and the focus slice of a frame, all on the device ----
extern "C" __global__ void lumak(const float* r,const float* g,const float* b,float* y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=0.299f*r[i]+0.587f*g[i]+0.114f*b[i];
}
// Burt's generating kernel with a = 0.33 (align::burt_kernel), border-renormalised by the
// sum of the in-bounds taps (align::reduce_burt): H pass at even columns, V pass at even rows
#define BA 0.33f
#define BB 0.25f
#define BC (0.25f-BA/2.f)
__device__ __forceinline__ float btap(int t){ return t==2?BA:((t==1||t==3)?BB:BC); }
__device__ __forceinline__ float bnorm(int i,int n){ float s=0.f; for(int t=0;t<5;t++){ int x=i+t-2; if(x>=0&&x<n) s+=btap(t); } return s; }
extern "C" __global__ void burtH(const float* in,float* tmp,int w,int h,int ow){
    int oj=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(oj>=ow||y>=h) return; int c=2*oj; const float* r=in+(size_t)y*w; float a=0.f;
    for(int t=0;t<5;t++){ int x=c+t-2; if(x>=0&&x<w) a+=btap(t)*r[x]; }
    tmp[(size_t)y*ow+oj]=a;
}
extern "C" __global__ void burtV(const float* tmp,float* out,int w,int h,int ow,int oh){
    int oj=blockIdx.x*blockDim.x+threadIdx.x, oi=blockIdx.y*blockDim.y+threadIdx.y;
    if(oj>=ow||oi>=oh) return; int c=2*oi; float a=0.f;
    for(int t=0;t<5;t++){ int y=c+t-2; if(y>=0&&y<h) a+=btap(t)*tmp[(size_t)y*ow+oj]; }
    out[(size_t)oi*ow+oj]=a/(bnorm(c,h)*bnorm(2*oj,w));
}
// the warp's kernels (align::Interp::id): 0 nearest, 1 bilinear, 2 bicubic, 3 spline4x4,
// 4 spline6x6, 5 lanczos3; `ktaps` taps at 1 - taps/2 .. from the floor of the source point,
// the distance-form kernels normalised to sum 1 (align::taps_of)
__device__ __forceinline__ int ktaps(int k){ return k<2?2:(k<4?4:6); }
__device__ __forceinline__ float keysf(float d){ if(d<1.f) return (1.5f*d-2.5f)*d*d+1.f; if(d<2.f) return ((-0.5f*d+2.5f)*d-4.f)*d+2.f; return 0.f; }
__device__ __forceinline__ float spline36f(float d){
    if(d<1.f) return ((13.f/11.f*d-453.f/209.f)*d-3.f/209.f)*d+1.f;
    if(d<2.f){ float u=d-1.f; return ((-6.f/11.f*u+270.f/209.f)*u-156.f/209.f)*u; }
    if(d<3.f){ float u=d-2.f; return ((1.f/11.f*u-45.f/209.f)*u+26.f/209.f)*u; }
    return 0.f;
}
__device__ __forceinline__ float lanczos3f(float d){ if(d<1e-9f) return 1.f; if(d>=3.f) return 0.f; float a=3.14159265f*d, b=a/3.f; return sinf(a)/a*(sinf(b)/b); }
__device__ __forceinline__ void kweights(int k,float t,float* w){
    for(int i=0;i<6;i++) w[i]=0.f;
    if(k==0){ float r=t>=0.5f?1.f:0.f; w[0]=1.f-r; w[1]=r; }
    else if(k==1){ w[0]=1.f-t; w[1]=t; }
    else if(k==3){ spl4f(t,w); }
    else { int n=ktaps(k), start=1-n/2; float s=0.f;
        for(int i=0;i<n;i++){ float d=fabsf(t-(float)(i+start)); float v=k==2?keysf(d):(k==4?spline36f(d):lanczos3f(d)); w[i]=v; s+=v; }
        for(int i=0;i<n;i++) w[i]/=s; }
}
// one plane warped by the inverse homography (in double, as the CPU's warp_with); where the
// source point falls outside the frame, the pixel as shot (stack::AlignedFrames's fallback)
extern "C" __global__ void warpk(const float* src,float* out,int w,int h,
        double ia,double ib,double itx,double id_,double ie,double ity,double ig0,double ig1,int k){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; size_t idx=(size_t)y*w+x;
    double den=ig0*x+ig1*y+1.0; double sx=(ia*x+ib*y+itx)/den, sy=(id_*x+ie*y+ity)/den;
    if(!(sx>=0.0&&sx<=(double)(w-1)&&sy>=0.0&&sy<=(double)(h-1))){ out[idx]=src[idx]; return; }
    int x0=(int)floor(sx), y0=(int)floor(sy);
    float wx[6],wy[6]; kweights(k,(float)(sx-x0),wx); kweights(k,(float)(sy-y0),wy);
    int n=ktaps(k), start=1-n/2; float acc=0.f;
    for(int j=0;j<n;j++){ int yy=min(max(y0+j+start,0),h-1); float r=0.f;
        for(int i=0;i<n;i++){ int xx=min(max(x0+i+start,0),w-1); r+=wx[i]*src[(size_t)yy*w+xx]; }
        acc+=wy[j]*r; }
    out[idx]=acc;
}
// frame 0 on the brightness sampling grid (brightness::Reference: every step-th pixel)
extern "C" __global__ void gridk(const float* src,float* g,int w,int gw,int gh,int step){
    int gx=blockIdx.x*blockDim.x+threadIdx.x, gy=blockIdx.y*blockDim.y+threadIdx.y;
    if(gx>=gw||gy>=gh) return; g[(size_t)gy*gw+gx]=src[(size_t)(gy*step)*w+gx*step];
}
// brightness sums (brightness::sums): over the grid pixels the warp covers (all, at the
// identity), the frame's three channel sums, the reference's, and the count -> out[7]
extern "C" __global__ void bsumk(const float* f0,const float* f1,const float* f2,const float* r0,const float* r1,const float* r2,
        int w,int h,int gw,int gh,int step,int ident,double ia,double ib,double itx,double id_,double ie,double ity,double ig0,double ig1,double* out){
    __shared__ float s[7][256];
    int gx=blockIdx.x*blockDim.x+threadIdx.x, gy=blockIdx.y*blockDim.y+threadIdx.y;
    int t=threadIdx.y*blockDim.x+threadIdx.x; float v[7]={0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    if(gx<gw&&gy<gh){ int x=gx*step, y=gy*step; bool ok=true;
        if(!ident){ double den=ig0*x+ig1*y+1.0; double sx=(ia*x+ib*y+itx)/den, sy=(id_*x+ie*y+ity)/den;
            ok= sx>=0.0&&sx<=(double)(w-1)&&sy>=0.0&&sy<=(double)(h-1); }
        if(ok){ size_t i=(size_t)y*w+x, j=(size_t)gy*gw+gx; v[0]=f0[i]; v[1]=f1[i]; v[2]=f2[i]; v[3]=r0[j]; v[4]=r1[j]; v[5]=r2[j]; v[6]=1.f; } }
    for(int k=0;k<7;k++) s[k][t]=v[k]; __syncthreads();
    for(int st=(blockDim.x*blockDim.y)>>1; st>0; st>>=1){ if(t<st) for(int k=0;k<7;k++) s[k][t]+=s[k][t+st]; __syncthreads(); }
    if(t==0) for(int k=0;k<7;k++) atomicAdd(&out[k],(double)s[k][0]);
}
extern "C" __global__ void gaink(float* a,float g,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]*=g; }
// the ring difference filter: a sparse convolution with reflect-101 borders (depth::conv_sparse
// over depth::rdf_taps, (dy, dx, weight) triples), then |.|
extern "C" __global__ void rdfk(const float* y,float* out,int w,int h,const float* taps,int ntaps){
    int x=blockIdx.x*blockDim.x+threadIdx.x, yy=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||yy>=h) return; float a=0.f;
    for(int t=0;t<ntaps;t++){ int dy=(int)taps[3*t], dx=(int)taps[3*t+1]; a+=taps[3*t+2]*y[(size_t)refl(yy+dy,h)*w+refl(x+dx,w)]; }
    out[(size_t)yy*w+x]=fabsf(a);
}
// the sum-modified Laplacian with sample step s (depth::focus_measure, Sml)
extern "C" __global__ void smlk(const float* y,float* out,int w,int h,int s){
    int x=blockIdx.x*blockDim.x+threadIdx.x, yy=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||yy>=h) return; const float* row=y+(size_t)yy*w; float c=row[x];
    float l=row[refl(x-s,w)], r=row[refl(x+s,w)], u=y[(size_t)refl(yy-s,h)*w+x], d=y[(size_t)refl(yy+s,h)*w+x];
    out[(size_t)yy*w+x]=fabsf(2.f*c-l-r)+fabsf(2.f*c-u-d);
}
"#;

const KERNELS: [&str; 15] = [
    "redH", "redV", "expH", "expV", "subk", "addk", "clampk", "energy_y", "energy_rgb", "winH", "winV", "selk", "wgtk", "wacck", "wnormk",
];

fn cfg2(w: usize, h: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((w as u32).div_ceil(32), (h as u32).div_ceil(8), 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 }
}
fn cfg1(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

struct Lvl {
    p: [CudaSlice<f32>; 3],
    w: usize,
    h: usize,
}

/// Stream + loaded kernels, kept apart from the buffers so a launch can borrow
/// the kernels immutably while buffers are borrowed mutably.
struct G {
    s: Arc<cudarc::driver::CudaStream>,
    k: std::collections::HashMap<&'static str, CudaFunction>,
}

impl G {
    fn f(&self, n: &'static str) -> &CudaFunction {
        &self.k[n]
    }
}

/// Init a CUDA context, compile `SRC` with NVRTC and load `names`.
fn init_gpu(src: &str, names: &[&'static str]) -> Result<(Arc<CudaContext>, G), String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed (is nvidia_uvm loaded?): {e:?}"))?;
    let s = ctx.default_stream();
    // arch compute_86 so warp_cost can use double atomicAdd.
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_86"), ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts([COMMON, src].concat(), opts).map_err(|e| format!("nvrtc: {e:?}"))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("cuda module: {e:?}"))?;
    let mut k = std::collections::HashMap::new();
    for &nm in names {
        k.insert(nm, module.load_function(nm).map_err(|e| format!("kernel {nm}: {e:?}"))?);
    }
    Ok((ctx, G { s, k }))
}

/// GPU twin of `fuse::Fuser` (same `new` / `push` / `finish` contract).
pub struct GpuFuser {
    _ctx: Arc<CudaContext>,
    g: G,
    pub w: usize,
    pub h: usize,
    pub levels: usize,
    params: FuseParams,
    depth_level: usize,
    acc: Vec<Lvl>,
    /// Working pyramid of the frame being folded (reused): levels 1..=levels;
    /// level 0 is the frame's own planes (`push_device`) or `cur0`.
    cur: Vec<Lvl>,
    /// Level 0 for frames that come from the host (`push`), made on first use.
    cur0: Option<Lvl>,
    /// Best energy per band-pass level; with halo control the weight sum at
    /// the levels coarser than the guide, the residual's included.
    best: Vec<CudaSlice<f32>>,
    /// Halo control: (guide level, hardness).
    halo: Option<(usize, f32)>,
    win: CudaSlice<f32>,
    wt: CudaSlice<f32>,
    klen: i32,
    /// Scratch: half-size pass buffer, full-size expand output, two energy planes.
    tmp_half: CudaSlice<f32>,
    tmp_full: CudaSlice<f32>,
    en: CudaSlice<f32>,
    en2: CudaSlice<f32>,
    tops: Vec<Img3>,
    count: usize,
}

impl GpuFuser {
    pub fn new(w: usize, h: usize, params: FuseParams) -> Result<GpuFuser, String> {
        let t = std::time::Instant::now();
        let (ctx, g) = init_gpu(SRC, &KERNELS)?;
        let s = g.s.clone();
        let levels = params.levels.unwrap_or_else(|| auto_levels(w, h, 32)).max(1);
        let depth_level = params.depth_level.min(levels - 1);
        let halo = halo_guide(&params, levels);
        let al = |n: usize| s.alloc_zeros::<f32>(n).map_err(|e| format!("cuda alloc: {e:?}"));
        let mut dims = vec![(w, h)];
        for _ in 0..levels {
            let (cw, ch) = *dims.last().unwrap();
            dims.push((half(cw), half(ch)));
        }
        let mk = |dims: &[(usize, usize)]| -> Result<Vec<Lvl>, String> {
            dims.iter().map(|&(lw, lh)| Ok(Lvl { p: [al(lw * lh)?, al(lw * lh)?, al(lw * lh)?], w: lw, h: lh })).collect()
        };
        let acc = mk(&dims)?;
        let cur = mk(&dims[1..])?;
        let mut best = Vec::new();
        for (l, &(lw, lh)) in dims.iter().enumerate() {
            let mut b = al(lw * lh)?;
            if halo.is_none_or(|(g, _)| l <= g) {
                // any energy (>= 0) beats -1, so the first frame fills the accumulator
                let neg = vec![-1.0f32; lw * lh];
                s.memcpy_htod(&neg, &mut b).map_err(|e| format!("{e:?}"))?;
            }
            // else a weight sum, from zero
            best.push(b);
        }
        let (dw, dh) = dims[depth_level];
        let win = al(dw * dh)?;
        let wtv = binomial(params.energy_radius);
        let klen = wtv.len() as i32;
        let wt = s.memcpy_stod(&wtv).map_err(|e| format!("{e:?}"))?;
        let tmp_half = al((half(w) * h).max(half(h) * w))?;
        let tmp_full = al(w * h)?;
        let en = al(w * h)?;
        let en2 = al(w * h)?;
        eprintln!("[gpu] context + nvrtc {:.2}s", t.elapsed().as_secs_f64());
        Ok(GpuFuser {
            _ctx: ctx, g, w, h, levels, params, depth_level, acc, cur, cur0: None, best, halo, win, wt, klen,
            tmp_half, tmp_full, en, en2, tops: Vec::new(), count: 0,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Build the Laplacian pyramid of the frame whose level 0 is `p0` (in
    /// place: `p0` becomes the finest band-pass level).
    fn build(&mut self, p0: &mut [CudaSlice<f32>; 3]) {
        for li in 0..self.levels {
            let (fw, fh) = if li == 0 { (self.w, self.h) } else { (self.cur[li - 1].w, self.cur[li - 1].h) };
            let (cw, ch) = (self.cur[li].w, self.cur[li].h);
            let (fwi, fhi, cwi, chi) = (fw as i32, fh as i32, cw as i32, ch as i32);
            for c in 0..3 {
                let (lo, hi) = self.cur.split_at_mut(li);
                let fp: &mut CudaSlice<f32> = if li == 0 { &mut p0[c] } else { &mut lo[li - 1].p[c] };
                let cp = &mut hi[0].p[c];
                // reduce
                { let mut b = self.g.s.launch_builder(self.g.f("redH")); b.arg(&*fp).arg(&mut self.tmp_half).arg(&fwi).arg(&fhi).arg(&cwi); unsafe { b.launch(cfg2(cw, fh)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("redV")); b.arg(&self.tmp_half).arg(&mut *cp).arg(&fhi).arg(&cwi).arg(&chi); unsafe { b.launch(cfg2(cw, ch)).unwrap() }; }
                // expand + subtract in place
                { let mut b = self.g.s.launch_builder(self.g.f("expH")); b.arg(&*cp).arg(&mut self.tmp_half).arg(&cwi).arg(&chi).arg(&fwi); unsafe { b.launch(cfg2(fw, ch)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("expV")); b.arg(&self.tmp_half).arg(&mut self.tmp_full).arg(&chi).arg(&fwi).arg(&fhi); unsafe { b.launch(cfg2(fw, fh)).unwrap() }; }
                let n = (fw * fh) as i32;
                { let mut b = self.g.s.launch_builder(self.g.f("subk")); b.arg(&mut *fp).arg(&self.tmp_full).arg(&n); unsafe { b.launch(cfg1(fw * fh)).unwrap() }; }
            }
        }
    }

    /// Fold a frame from the host: uploaded into `cur0`, then `push_device`.
    pub fn push(&mut self, frame: &Img3) -> Result<(), String> {
        assert!(frame.w == self.w && frame.h == self.h, "frame size mismatch");
        let mut p0 = match self.cur0.take() {
            Some(l) => l,
            None => {
                let al = |n: usize| self.g.s.alloc_zeros::<f32>(n).map_err(|e| format!("cuda alloc: {e:?}"));
                Lvl { p: [al(self.w * self.h)?, al(self.w * self.h)?, al(self.w * self.h)?], w: self.w, h: self.h }
            }
        };
        for c in 0..3 {
            self.g.s.memcpy_htod(&frame.p[c], &mut p0.p[c]).map_err(|e| format!("{e:?}"))?;
        }
        let r = self.push_device(&mut p0.p);
        self.cur0 = Some(p0);
        r
    }

    /// Fold a frame already on the device (`GpuFrames`): its three planes
    /// (w × h) are consumed — they become the finest band-pass level.
    pub fn push_device(&mut self, p0: &mut [CudaSlice<f32>; 3]) -> Result<(), String> {
        self.build(p0);
        let idx = self.count as f32;
        let nsel = self.halo.map_or(self.levels, |(g, _)| g + 1);
        for li in 0..nsel {
            let (lw, lh) = if li == 0 { (self.w, self.h) } else { (self.cur[li - 1].w, self.cur[li - 1].h) };
            let (n, ni, wi, hi) = (lw * lh, (lw * lh) as i32, lw as i32, lh as i32);
            let [n0, n1, n2] = if li == 0 { &*p0 } else { &self.cur[li - 1].p };
            let ek = if self.params.use_chroma { "energy_rgb" } else { "energy_y" };
            { let mut b = self.g.s.launch_builder(self.g.f(ek)); b.arg(n0).arg(n1).arg(n2).arg(&mut self.en).arg(&ni); unsafe { b.launch(cfg1(n)).unwrap() }; }
            if self.klen > 1 {
                { let mut b = self.g.s.launch_builder(self.g.f("winH")); b.arg(&self.en).arg(&mut self.en2).arg(&wi).arg(&hi).arg(&self.wt).arg(&self.klen); unsafe { b.launch(cfg2(lw, lh)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("winV")); b.arg(&self.en2).arg(&mut self.en).arg(&wi).arg(&hi).arg(&self.wt).arg(&self.klen); unsafe { b.launch(cfg2(lw, lh)).unwrap() }; }
            }
            let rec = (li == self.depth_level) as i32;
            let [a0, a1, a2] = &mut self.acc[li].p;
            let mut b = self.g.s.launch_builder(self.g.f("selk"));
            b.arg(&self.en).arg(&mut self.best[li]).arg(a0).arg(a1).arg(a2).arg(n0).arg(n1).arg(n2).arg(&mut self.win).arg(&idx).arg(&rec).arg(&ni);
            unsafe { b.launch(cfg1(n)).unwrap() };
        }
        if let Some((guide, p)) = self.halo {
            // the guide's weights (from its region energy, still in `en`), REDUCEd
            // level by level into `en2`, fold the coarser levels and the residual
            let (mut lw, mut lh) = if guide == 0 { (self.w, self.h) } else { (self.cur[guide - 1].w, self.cur[guide - 1].h) };
            let ni = (lw * lh) as i32;
            { let mut b = self.g.s.launch_builder(self.g.f("wgtk")); b.arg(&self.en).arg(&mut self.en2).arg(&p).arg(&HALO_FLOOR).arg(&HALO_REF).arg(&ni); unsafe { b.launch(cfg1(lw * lh)).unwrap() }; }
            for li in guide + 1..=self.levels {
                let (cw, ch) = (self.cur[li - 1].w, self.cur[li - 1].h);
                let (lwi, lhi, cwi, chi) = (lw as i32, lh as i32, cw as i32, ch as i32);
                { let mut b = self.g.s.launch_builder(self.g.f("redH")); b.arg(&self.en2).arg(&mut self.tmp_half).arg(&lwi).arg(&lhi).arg(&cwi); unsafe { b.launch(cfg2(cw, lh)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("redV")); b.arg(&self.tmp_half).arg(&mut self.en2).arg(&lhi).arg(&cwi).arg(&chi); unsafe { b.launch(cfg2(cw, ch)).unwrap() }; }
                let n = cw * ch;
                let ni = n as i32;
                let [n0, n1, n2] = &self.cur[li - 1].p;
                let [a0, a1, a2] = &mut self.acc[li].p;
                let mut b = self.g.s.launch_builder(self.g.f("wacck"));
                b.arg(&self.en2).arg(a0).arg(a1).arg(a2).arg(n0).arg(n1).arg(n2).arg(&mut self.best[li]).arg(&ni);
                unsafe { b.launch(cfg1(n)).unwrap() };
                (lw, lh) = (cw, ch);
            }
        } else {
            // residual level back to the host (tiny)
            let top = &self.cur[self.levels - 1];
            let mut t = Img3::zeros(top.w, top.h);
            for c in 0..3 {
                t.p[c] = self.g.s.memcpy_dtov(&top.p[c]).map_err(|e| format!("{e:?}"))?;
            }
            self.tops.push(t);
        }
        self.count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<(Img3, Vec<f32>), String> {
        assert!(self.count > 0, "no frames pushed");
        match self.halo {
            Some((guide, _)) => {
                // the weighted means
                for li in guide + 1..=self.levels {
                    let n = self.acc[li].w * self.acc[li].h;
                    let ni = n as i32;
                    let [a0, a1, a2] = &mut self.acc[li].p;
                    let mut b = self.g.s.launch_builder(self.g.f("wnormk"));
                    b.arg(a0).arg(a1).arg(a2).arg(&self.best[li]).arg(&ni);
                    unsafe { b.launch(cfg1(n)).unwrap() };
                }
            }
            None => {
                let top = fuse_residuals(&self.tops, &self.params);
                for c in 0..3 {
                    self.g.s.memcpy_htod(&top.p[c], &mut self.acc[self.levels].p[c]).map_err(|e| format!("{e:?}"))?;
                }
            }
        }
        // collapse: G_l = L_l + EXPAND(G_{l+1})
        for li in (0..self.levels).rev() {
            let (fw, fh) = (self.acc[li].w, self.acc[li].h);
            let (cw, ch) = (self.acc[li + 1].w, self.acc[li + 1].h);
            let (fwi, fhi, cwi, chi) = (fw as i32, fh as i32, cw as i32, ch as i32);
            for c in 0..3 {
                let (fine, coarse) = self.acc.split_at_mut(li + 1);
                let (fp, cp) = (&mut fine[li].p[c], &coarse[0].p[c]);
                { let mut b = self.g.s.launch_builder(self.g.f("expH")); b.arg(cp).arg(&mut self.tmp_half).arg(&cwi).arg(&chi).arg(&fwi); unsafe { b.launch(cfg2(fw, ch)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("expV")); b.arg(&self.tmp_half).arg(&mut self.tmp_full).arg(&chi).arg(&fwi).arg(&fhi); unsafe { b.launch(cfg2(fw, fh)).unwrap() }; }
                let n = (fw * fh) as i32;
                { let mut b = self.g.s.launch_builder(self.g.f("addk")); b.arg(&mut *fp).arg(&self.tmp_full).arg(&n); unsafe { b.launch(cfg1(fw * fh)).unwrap() }; }
            }
        }
        let n = self.w * self.h;
        let ni = n as i32;
        let mut img = Img3::zeros(self.w, self.h);
        for c in 0..3 {
            { let mut b = self.g.s.launch_builder(self.g.f("clampk")); b.arg(&mut self.acc[0].p[c]).arg(&ni); unsafe { b.launch(cfg1(n)).unwrap() }; }
            img.p[c] = self.g.s.memcpy_dtov(&self.acc[0].p[c]).map_err(|e| format!("{e:?}"))?;
        }
        let (dw, dh) = (self.acc[self.depth_level].w, self.acc[self.depth_level].h);
        let win = self.g.s.memcpy_dtov(&self.win).map_err(|e| format!("{e:?}"))?;
        let depth = upsample_index(&win, dw, dh, self.w, self.h, self.depth_level);
        Ok((img, depth))
    }
}

/// A frame on the device, as `GpuFrames` hands it to the fuser: its three
/// warped, equalised planes (consumed by `GpuFuser::push_device`) and, when
/// the depth pass asked for it, its focus slice (`depth::focus_slice`).
pub struct DeviceFrame<'a> {
    pub planes: &'a mut [CudaSlice<f32>; 3],
    pub focus: Option<Vec<f32>>,
}

/// One level of a registration pyramid on the device.
type DLvl = (CudaSlice<f32>, usize, usize);

/// The levels above full resolution, by the rule of `align::gauss_levels`.
fn pyr_dims(w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut d = Vec::new();
    let (mut cw, mut ch) = (w, h);
    while ch > 64 && cw > 8 {
        cw = half(cw);
        ch = half(ch);
        d.push((cw, ch));
    }
    d
}

/// The device side of the aligned stream (`stack::AlignedFrames` with
/// `--gpu-align`): a decoded frame is uploaded once, and everything the
/// stream does with it happens here — its luma and registration pyramid
/// (`lumak`, `burtH`/`burtV`), the Nelder-Mead cost search against the
/// previous warped frame's pyramid (`warp_cost`), the warp with the pixel
/// as shot where the source point leaves the frame (`warpk`, the kernel of
/// the user's choice), the brightness gains against frame 0 on the sampling
/// grid (`gridk`, `bsumk`, `gaink`) and the depth pass's focus slice
/// (`rdfk`/`smlk`, `bmeank`). The warped planes go to the fuser as they are
/// (`DeviceFrame`); the host gets back the transform, the gains, the slice,
/// and the frame itself only when something on the host asks (`download`).
/// Coordinates are computed in double like the CPU's; taps and sums in FP32,
/// so a frame differs from the CPU's by float rounding.
pub struct GpuFrames {
    _ctx: Arc<CudaContext>,
    g: G,
    w: usize,
    h: usize,
    interp: Interp,
    /// The frame as decoded, and as warped and equalised.
    src: [CudaSlice<f32>; 3],
    out: [CudaSlice<f32>; 3],
    /// The frame's luma (the search's target; later the equalised frame's, for
    /// the focus measure), the warped frame's luma (the next search's
    /// reference), and a full-size scratch plane.
    yt: CudaSlice<f32>,
    yr: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    /// The pyramids above `yt` and `yr`.
    pt: Vec<DLvl>,
    pr: Vec<DLvl>,
    have_ref: bool,
    /// Frame 0 on the brightness sampling grid, once seen.
    grid: Option<[CudaSlice<f32>; 3]>,
    gw: usize,
    gh: usize,
    sums: std::cell::RefCell<CudaSlice<f64>>,
    cost: std::cell::RefCell<CudaSlice<f64>>,
    /// The ring difference filter's taps, for the measure they were made for.
    taps: Option<(FocusMeasure, CudaSlice<f32>, i32)>,
    slice: Option<(CudaSlice<f32>, usize, usize)>,
}

/// The brightness sampling step (`brightness::STEP`).
const BSTEP: usize = 4;

impl GpuFrames {
    pub fn new(w: usize, h: usize, interp: Interp) -> Result<GpuFrames, String> {
        let (ctx, g) = init_gpu(SRC, &["warp_cost", "lumak", "burtH", "burtV", "warpk", "gridk", "bsumk", "gaink", "rdfk", "smlk", "bmeank"])?;
        let s = g.s.clone();
        let al = |n: usize| s.alloc_zeros::<f32>(n).map_err(|e| format!("cuda alloc: {e:?}"));
        let n = w * h;
        let mk = || -> Result<Vec<DLvl>, String> { pyr_dims(w, h).into_iter().map(|(lw, lh)| Ok((al(lw * lh)?, lw, lh))).collect() };
        Ok(GpuFrames {
            src: [al(n)?, al(n)?, al(n)?],
            out: [al(n)?, al(n)?, al(n)?],
            yt: al(n)?,
            yr: al(n)?,
            tmp: al(n)?,
            pt: mk()?,
            pr: mk()?,
            have_ref: false,
            grid: None,
            gw: w.div_ceil(BSTEP),
            gh: h.div_ceil(BSTEP),
            sums: std::cell::RefCell::new(s.alloc_zeros::<f64>(7).map_err(|e| format!("cuda alloc: {e:?}"))?),
            cost: std::cell::RefCell::new(s.alloc_zeros::<f64>(3).map_err(|e| format!("cuda alloc: {e:?}"))?),
            taps: None,
            slice: None,
            _ctx: ctx,
            g,
            w,
            h,
            interp,
        })
    }

    /// The registration pyramid above a full-resolution luma plane.
    fn reduce_chain(g: &G, tmp: &mut CudaSlice<f32>, y: &CudaSlice<f32>, w: usize, h: usize, levels: &mut [DLvl]) {
        let (mut fw, mut fh) = (w, h);
        for l in 0..levels.len() {
            let (lo, hi) = levels.split_at_mut(l);
            let fine: &CudaSlice<f32> = if l == 0 { y } else { &lo[l - 1].0 };
            let (coarse, cw, ch) = (&mut hi[0].0, hi[0].1, hi[0].2);
            let (fwi, fhi, cwi, chi) = (fw as i32, fh as i32, cw as i32, ch as i32);
            { let mut b = g.s.launch_builder(g.f("burtH")); b.arg(fine).arg(&mut *tmp).arg(&fwi).arg(&fhi).arg(&cwi); unsafe { b.launch(cfg2(cw, fh)).unwrap() }; }
            { let mut b = g.s.launch_builder(g.f("burtV")); b.arg(&*tmp).arg(coarse).arg(&fwi).arg(&fhi).arg(&cwi).arg(&chi); unsafe { b.launch(cfg2(cw, ch)).unwrap() }; }
            (fw, fh) = (cw, ch);
        }
    }

    /// Luma of three planes into `y`.
    fn luma(g: &G, p: &[CudaSlice<f32>; 3], y: &mut CudaSlice<f32>, n: usize) {
        let ni = n as i32;
        let mut b = g.s.launch_builder(g.f("lumak"));
        b.arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(y).arg(&ni);
        unsafe { b.launch(cfg1(n)).unwrap() };
    }

    /// The transform that takes the frame's luma (`yt`, its pyramid built)
    /// onto the previous warped frame's (`yr`), from `guess` over the `free`
    /// parameters, `coarsen` levels short of full resolution: the CPU's
    /// `multiscale_align` with the cost on the device.
    fn search(&self, guess: Sim, free: [bool; Sim::N], coarsen: usize) -> Sim {
        let (w, h) = (self.w, self.h);
        let stream = self.g.s.clone();
        let free_idx: Vec<usize> = (0..Sim::N).filter(|&k| free[k]).collect();
        let span = Sim::SPAN;
        let nlv = self.pt.len().min(self.pr.len()) + 1;
        let finest = coarsen.min(nlv.saturating_sub(1));
        fn lvl_of<'a>(p: &'a [DLvl], y: &'a CudaSlice<f32>, w: usize, h: usize, l: usize) -> (&'a CudaSlice<f32>, usize, usize) {
            if l == 0 { (y, w, h) } else { (&p[l - 1].0, p[l - 1].1, p[l - 1].2) }
        }
        let iv = guess.as_vec();
        let mut cur = iv;
        let lo_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] - span[k]).collect();
        let hi_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] + span[k]).collect();
        for lvl in (finest..nlv).rev() {
            let (rd, aw, ah) = lvl_of(&self.pr, &self.yr, w, h, lvl);
            let (td, tw, th) = lvl_of(&self.pt, &self.yt, w, h, lvl);
            let cur_snap = cur;
            let cost = |xf: &[f64]| -> f64 {
                let mut v = cur_snap;
                for (k, &idx) in free_idx.iter().enumerate() {
                    v[idx] = xf[k];
                }
                let inv = inverse(Sim::from_vec(&v).matrix(tw, th));
                let (twi, thi, awi, ahi) = (tw as i32, th as i32, aw as i32, ah as i32);
                let (ia, ib, itx) = (inv[0][0] as f32, inv[0][1] as f32, inv[0][2] as f32);
                let (id_, ie, ity) = (inv[1][0] as f32, inv[1][1] as f32, inv[1][2] as f32);
                let (ig0, ig1) = (inv[2][0] as f32, inv[2][1] as f32);
                let mut ob = self.cost.borrow_mut();
                stream.memset_zeros(&mut *ob).unwrap();
                let mut b = stream.launch_builder(self.g.f("warp_cost"));
                b.arg(td).arg(&twi).arg(&thi).arg(rd).arg(&awi).arg(&ahi)
                    .arg(&ia).arg(&ib).arg(&itx).arg(&id_).arg(&ie).arg(&ity).arg(&ig0).arg(&ig1).arg(&mut *ob);
                unsafe { b.launch(cfg2(aw, ah)).unwrap() };
                let r = stream.memcpy_dtov(&*ob).unwrap();
                let (sd, sd2, cnt) = (r[0], r[1], r[2]);
                if cnt < 16.0 {
                    return 1e9;
                }
                ((sd2 - sd * sd / cnt) / cnt).max(0.0).sqrt()
            };
            let x0: Vec<f64> = free_idx.iter().map(|&k| cur[k]).collect();
            let (step, tol) = level_steps(&free_idx, lvl, w, h);
            let best = nelder_mead(&cost, &x0, &lo_f, &hi_f, &step, &tol);
            for (k, &idx) in free_idx.iter().enumerate() {
                cur[idx] = best[k];
            }
        }
        Sim::from_vec(&cur)
    }

    /// Frame `img` onto the device, registered, warped and equalised:
    /// searched from `guess` (the previous frame's transform) unless
    /// `known` gives its transform and gains already (a later pass), or it
    /// is the first frame, which sits at the identity and becomes the
    /// brightness reference. Returns the transform, the gains and, for
    /// `measure`, the focus slice. Afterwards `planes` holds the frame and
    /// `yr` its luma for the next frame's search.
    #[allow(clippy::too_many_arguments)]
    pub fn process(
        &mut self,
        img: &Img3,
        guess: Sim,
        free: [bool; Sim::N],
        coarsen: usize,
        known: Option<(Sim, [f32; 3])>,
        brightness: bool,
        measure: Option<&DepthParams>,
    ) -> Result<(Sim, [f32; 3], Option<Vec<f32>>), String> {
        assert!(img.w == self.w && img.h == self.h, "frame size mismatch");
        let (w, h, n) = (self.w, self.h, self.w * self.h);
        let e = |r: Result<(), cudarc::driver::DriverError>| r.map_err(|e| format!("{e:?}"));
        for c in 0..3 {
            e(self.g.s.memcpy_htod(&img.p[c], &mut self.src[c]))?;
        }
        let first = !self.have_ref && known.is_none();
        let searching = known.is_none() && !first;
        // the target's luma and pyramid, for the search; nothing else needs them
        if searching {
            Self::luma(&self.g, &self.src, &mut self.yt, n);
            Self::reduce_chain(&self.g, &mut self.tmp, &self.yt, w, h, &mut self.pt);
        }
        let sim = match known {
            Some((s, _)) => s,
            None if first => Sim::id(),
            None => self.search(guess, free, coarsen),
        };
        // the warp
        if sim == Sim::id() {
            for c in 0..3 {
                e(self.g.s.memcpy_dtod(&self.src[c], &mut self.out[c]))?;
            }
        } else {
            let inv = inverse(sim.matrix(w, h));
            let (wi, hi, k) = (w as i32, h as i32, self.interp.id() as i32);
            for c in 0..3 {
                let mut b = self.g.s.launch_builder(self.g.f("warpk"));
                b.arg(&self.src[c]).arg(&mut self.out[c]).arg(&wi).arg(&hi)
                    .arg(&inv[0][0]).arg(&inv[0][1]).arg(&inv[0][2]).arg(&inv[1][0]).arg(&inv[1][1]).arg(&inv[1][2]).arg(&inv[2][0]).arg(&inv[2][1]).arg(&k);
                unsafe { b.launch(cfg2(w, h)).unwrap() };
            }
        }
        // the next search's reference: this frame as warped, before the gains
        if known.is_none() {
            Self::luma(&self.g, &self.out, &mut self.yr, n);
            Self::reduce_chain(&self.g, &mut self.tmp, &self.yr, w, h, &mut self.pr);
            self.have_ref = true;
        }
        // the brightness gains against frame 0 on the sampling grid
        let (gw, gh) = (self.gw, self.gh);
        let gains = match known {
            Some((_, g)) => g,
            None if !brightness => [1.0; 3],
            None if first => {
                let mut grid = [0, 1, 2].map(|_| self.g.s.alloc_zeros::<f32>(gw * gh).map_err(|e| format!("cuda alloc: {e:?}")));
                let (wi, gwi, ghi, st) = (w as i32, gw as i32, gh as i32, BSTEP as i32);
                for c in 0..3 {
                    let gc = grid[c].as_mut().map_err(|e| e.clone())?;
                    let mut b = self.g.s.launch_builder(self.g.f("gridk"));
                    b.arg(&self.out[c]).arg(gc).arg(&wi).arg(&gwi).arg(&ghi).arg(&st);
                    unsafe { b.launch(cfg2(gw, gh)).unwrap() };
                }
                self.grid = Some([grid[0].as_ref().unwrap().clone(), grid[1].as_ref().unwrap().clone(), grid[2].as_ref().unwrap().clone()]);
                [1.0; 3]
            }
            None => {
                let grid = self.grid.as_ref().ok_or("brightness reference missing")?;
                let inv = inverse(sim.matrix(w, h));
                let (wi, hi, gwi, ghi, st, ident) = (w as i32, h as i32, gw as i32, gh as i32, BSTEP as i32, (sim == Sim::id()) as i32);
                let mut sums = self.sums.borrow_mut();
                e(self.g.s.memset_zeros(&mut *sums))?;
                {
                    let mut b = self.g.s.launch_builder(self.g.f("bsumk"));
                    b.arg(&self.out[0]).arg(&self.out[1]).arg(&self.out[2]).arg(&grid[0]).arg(&grid[1]).arg(&grid[2])
                        .arg(&wi).arg(&hi).arg(&gwi).arg(&ghi).arg(&st).arg(&ident)
                        .arg(&inv[0][0]).arg(&inv[0][1]).arg(&inv[0][2]).arg(&inv[1][0]).arg(&inv[1][1]).arg(&inv[1][2]).arg(&inv[2][0]).arg(&inv[2][1])
                        .arg(&mut *sums);
                    unsafe { b.launch(cfg2(gw, gh)).unwrap() };
                }
                let r = self.g.s.memcpy_dtov(&*sums).map_err(|e| format!("{e:?}"))?;
                crate::brightness::ratio([r[3], r[4], r[5]], [r[0], r[1], r[2]], r[6] as usize)
            }
        };
        if !crate::brightness::is_unity(gains) {
            let ni = n as i32;
            for c in 0..3 {
                if (gains[c] - 1.0).abs() > 1e-6 {
                    let mut b = self.g.s.launch_builder(self.g.f("gaink"));
                    b.arg(&mut self.out[c]).arg(&gains[c]).arg(&ni);
                    unsafe { b.launch(cfg1(n)).unwrap() };
                }
            }
        }
        // the focus slice of the frame as fused
        let focus = match measure {
            None => None,
            Some(dp) => {
                Self::luma(&self.g, &self.out, &mut self.yt, n);
                let (wi, hi) = (w as i32, h as i32);
                match dp.focus {
                    FocusMeasure::Rdf { r_in, r_out } => {
                        if self.taps.as_ref().is_none_or(|(m, _, _)| *m != dp.focus) {
                            let t: Vec<f32> = rdf_taps(r_in, r_out).iter().flat_map(|&(dy, dx, wt)| [dy as f32, dx as f32, wt]).collect();
                            let nt = (t.len() / 3) as i32;
                            self.taps = Some((dp.focus, self.g.s.memcpy_stod(&t).map_err(|e| format!("{e:?}"))?, nt));
                        }
                        let (_, taps, nt) = self.taps.as_ref().unwrap();
                        let mut b = self.g.s.launch_builder(self.g.f("rdfk"));
                        b.arg(&self.yt).arg(&mut self.tmp).arg(&wi).arg(&hi).arg(taps).arg(nt);
                        unsafe { b.launch(cfg2(w, h)).unwrap() };
                    }
                    FocusMeasure::Sml { step } => {
                        let st = step.max(1) as i32;
                        let mut b = self.g.s.launch_builder(self.g.f("smlk"));
                        b.arg(&self.yt).arg(&mut self.tmp).arg(&wi).arg(&hi).arg(&st);
                        unsafe { b.launch(cfg2(w, h)).unwrap() };
                    }
                }
                let k = 1usize << dp.scale;
                let (dw, dh) = (blocks(w, k), blocks(h, k));
                if self.slice.as_ref().is_none_or(|(_, a, b)| (*a, *b) != (dw, dh)) {
                    self.slice = Some((self.g.s.alloc_zeros::<f32>(dw * dh).map_err(|e| format!("cuda alloc: {e:?}"))?, dw, dh));
                }
                let (sl, _, _) = self.slice.as_mut().unwrap();
                let (ki, dwi, dhi) = (k as i32, dw as i32, dh as i32);
                {
                    let mut b = self.g.s.launch_builder(self.g.f("bmeank"));
                    b.arg(&self.tmp).arg(&mut *sl).arg(&wi).arg(&hi).arg(&ki).arg(&dwi).arg(&dhi);
                    unsafe { b.launch(cfg2(dw, dh)).unwrap() };
                }
                Some(self.g.s.memcpy_dtov(&*sl).map_err(|e| format!("{e:?}"))?)
            }
        };
        Ok((sim, gains, focus))
    }

    /// The processed frame's planes, for the fuser.
    pub fn planes(&mut self) -> &mut [CudaSlice<f32>; 3] {
        &mut self.out
    }

    /// The processed frame, back on the host.
    pub fn download(&self) -> Result<Img3, String> {
        let mut img = Img3::zeros(self.w, self.h);
        for c in 0..3 {
            img.p[c] = self.g.s.memcpy_dtov(&self.out[c]).map_err(|e| format!("{e:?}"))?;
        }
        Ok(img)
    }
}

// ---------------------------------------------------------------- depth pass

/// The depth pass's kernels (`depth.rs` transcribed): the box and guided
/// filters, the streamed peak tracker, the median, the WLS system with its
/// separable sweeps and multigrid hierarchy, the conjugate gradient's
/// vector operations, and the upsampling.
const DEPTH_SRC: &str = r#"
// box filter, border-clipped: boxH row sums into tmp, boxV column sums / count
extern "C" __global__ void boxH(const float* in,float* tmp,int w,int h,int r){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; int x0=max(x-r,0), x1=min(x+r+1,w); const float* row=in+(size_t)y*w; float s=0.f;
    for(int i=x0;i<x1;i++) s+=row[i];
    tmp[(size_t)y*w+x]=s;
}
extern "C" __global__ void boxV(const float* tmp,float* out,int w,int h,int r){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; int y0=max(y-r,0), y1=min(y+r+1,h); float s=0.f;
    for(int i=y0;i<y1;i++) s+=tmp[(size_t)i*w+x];
    float cx=(float)(min(x+r+1,w)-max(x-r,0));
    out[(size_t)y*w+x]=s/(cx*(float)(y1-y0));
}
extern "C" __global__ void mulk(const float* a,const float* b,float* o,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]*b[i]; }
// guided filter: the guide's variance, the coefficients a, b, and q = mean_a * I + mean_b
extern "C" __global__ void gfvar(const float* mi,const float* mii,float* vi,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) vi[i]=fmaxf(mii[i]-mi[i]*mi[i],0.f); }
extern "C" __global__ void gfab(const float* mp,const float* corr,const float* mi,const float* vi,float* a,float* b,float eps,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float A=(corr[i]-mi[i]*mp[i])/(vi[i]+eps); a[i]=A; b[i]=mp[i]-A*mi[i];
}
extern "C" __global__ void gfapply(const float* ma,const float* mb,const float* g,float* o,int n,int clampz){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float q=ma[i]*g[i]+mb[i]; o[i]=clampz?fmaxf(q,0.f):q;
}
// streamed peak tracker (depth::PeakTracker): state s of 9 planes,
// 0 c1, 1 l1, 2 r1, 3 c2, 4 prev, 5 prev2, 6 sum, 7 cmin, 8 i1
__device__ __forceinline__ void peak_register(float* s,int n,int i,float val,float idx,float l,float r){
    if(val>s[i]){ if(s[i]>=0.f) s[3*n+i]=s[i]; s[i]=val; s[8*n+i]=idx; s[n+i]=l; s[2*n+i]=r; }
    else if(val>s[3*n+i]) s[3*n+i]=val;
}
extern "C" __global__ void peak_init(float* s,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    s[i]=-1.f; s[n+i]=-1.f; s[2*n+i]=-1.f; s[3*n+i]=-1.f; s[4*n+i]=0.f; s[5*n+i]=0.f; s[6*n+i]=0.f; s[7*n+i]=3.4e38f; s[8*n+i]=0.f;
}
extern "C" __global__ void peak_push(const float* q,float* s,int n,int m){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float v=q[i], prev=s[4*n+i], prev2=s[5*n+i];
    if(m>=1){
        bool is_peak=(m==1||prev>=prev2)&&prev>v;
        if(is_peak) peak_register(s,n,i,prev,(float)(m-1),m>=2?prev2:-1.f,v);
    }
    s[6*n+i]+=v; s[7*n+i]=fminf(s[7*n+i],v); s[5*n+i]=prev; s[4*n+i]=v;
}
extern "C" __global__ void peak_finish(float* s,float* depth,float* conf,int n,int nf,float floor_,float gate){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float prev=s[4*n+i], prev2=s[5*n+i];
    if(nf==1||prev>=prev2) peak_register(s,n,i,prev,(float)(nf-1),nf>=2?prev2:-1.f,-1.f);
    float c1=s[i], l=s[n+i], r=s[2*n+i], c2=s[3*n+i], delta=0.f;
    if(l>=0.f&&r>=0.f&&c1>0.f){
        float ll=logf(fmaxf(l,1e-12f)), lc=logf(c1), lr=logf(fmaxf(r,1e-12f)), den=ll-2.f*lc+lr;
        if(den<0.f) delta=fminf(fmaxf(0.5f*(ll-lr)/den,-0.5f),0.5f);
    }
    depth[i]=s[8*n+i]+delta;
    float c=0.f;
    if(c1>0.f){
        float mean=s[6*n+i]/(float)nf;
        float prom=fminf(fmaxf(1.f-mean/c1,0.f),1.f);
        float pkr=c2>=0.f?fminf(fmaxf(1.f-c2/c1,0.f),1.f):1.f;
        float g=(gate>0.f&&floor_>0.f)?fminf(fmaxf((c1-floor_)/(gate*floor_),0.f),1.f):1.f;
        c=prom*pkr*g;
    }
    conf[i]=c;
}
// 3x3 median, reflect-101
extern "C" __global__ void median3k(const float* a,float* o,int w,int h){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; float v[9]; int k=0;
    for(int dy=-1;dy<=1;dy++){ int yy=refl(y+dy,h); for(int dx=-1;dx<=1;dx++) v[k++]=a[(size_t)yy*w+refl(x+dx,w)]; }
    for(int i=0;i<5;i++) for(int j=8;j>i;j--) if(v[j]<v[j-1]){ float t=v[j]; v[j]=v[j-1]; v[j-1]=t; }
    o[(size_t)y*w+x]=v[4];
}
// data weights: wd = min(max(c,0) * inv_p90, 1) + eps; and the Huber reweight
extern "C" __global__ void wdk(const float* c,float* wd,int n,float inv,float eps){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) wd[i]=fminf(fmaxf(c[i],0.f)*inv,1.f)+eps; }
extern "C" __global__ void robustk(const float* cw,const float* d,const float* u,float* wd,int n,float tau,float eps){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    wd[i]=fmaxf(cw[i]*fminf(tau/fabsf(d[i]-u[i]),1.f),0.f)+eps;
}
// edge weights lambda * exp(-|dI|/sigma) between (x,y)-(x+1,y) -> ax and (x,y)-(x,y+1) -> ay
extern "C" __global__ void edgew(const float* g,float* ax,float* ay,int w,int h,float inv,float lam){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; size_t i=(size_t)y*w+x;
    ax[i]=x+1<w?lam*expf(fabsf(g[i]-g[i+1])*inv):0.f;
    ay[i]=y+1<h?lam*expf(fabsf(g[i]-g[i+w])*inv):0.f;
}
// 1-D WLS along rows / columns by the Thomas algorithm (depth::solve_rows / solve_cols):
// (wd_i + lam(a_{i-1} + a_i)) u_i - lam a_{i-1} u_{i-1} - lam a_i u_{i+1} = wd_i f_i,
// a = the edge weights (with lambda in them: lam is lambda_t / lambda), wd = 1 when unit.
// One thread per row / column; cp, dp two scratch planes.
extern "C" __global__ void fgs_rows(const float* f,float* u,const float* wd,const float* a,float* cp,float* dp,int w,int h,float lam,int unit){
    int y=blockIdx.x*blockDim.x+threadIdx.x; if(y>=h) return; size_t o=(size_t)y*w; float prev_a=0.f;
    for(int i=0;i<w;i++){
        size_t k=o+i; float ai=i+1<w?a[k]:0.f, wk=unit?1.f:wd[k];
        float diag=wk+lam*(prev_a+ai), lower=-lam*prev_a, upper=-lam*ai;
        float c0=i>0?cp[k-1]:0.f, d0=i>0?dp[k-1]:0.f;
        float m=diag-lower*c0; if(fabsf(m)<1e-12f) m=1e-12f;
        cp[k]=upper/m; dp[k]=(wk*f[k]-lower*d0)/m; prev_a=ai;
    }
    u[o+w-1]=dp[o+w-1];
    for(int i=w-2;i>=0;i--){ size_t k=o+i; u[k]=dp[k]-cp[k]*u[k+1]; }
}
extern "C" __global__ void fgs_cols(const float* f,float* u,const float* wd,const float* a,float* cp,float* dp,int w,int h,float lam,int unit){
    int x=blockIdx.x*blockDim.x+threadIdx.x; if(x>=w) return; float prev_a=0.f;
    for(int y=0;y<h;y++){
        size_t k=(size_t)y*w+x; float ai=y+1<h?a[k]:0.f, wk=unit?1.f:wd[k];
        float diag=wk+lam*(prev_a+ai), lower=-lam*prev_a, upper=-lam*ai;
        float c0=y>0?cp[k-w]:0.f, d0=y>0?dp[k-w]:0.f;
        float m=diag-lower*c0; if(fabsf(m)<1e-12f) m=1e-12f;
        cp[k]=upper/m; dp[k]=(wk*f[k]-lower*d0)/m; prev_a=ai;
    }
    size_t last=(size_t)(h-1)*w+x; u[last]=dp[last];
    for(int y=h-2;y>=0;y--){ size_t k=(size_t)y*w+x; u[k]=dp[k]-cp[k]*u[k+w]; }
}
// the multigrid hierarchy (depth::MgLevel): 2x2 aggregation of the data weights and of the
// edges a block boundary cuts; 1/diag; the stencil A x = wd x + sum a (x - x_nb) in three uses
extern "C" __global__ void coarse_wd(const float* wd,float* c,int w,int h,int cw,int ch){
    int X=blockIdx.x*blockDim.x+threadIdx.x, Y=blockIdx.y*blockDim.y+threadIdx.y;
    if(X>=cw||Y>=ch) return; float s=0.f;
    for(int y=2*Y;y<min(2*Y+2,h);y++) for(int x=2*X;x<min(2*X+2,w);x++) s+=wd[(size_t)y*w+x];
    c[(size_t)Y*cw+X]=s;
}
extern "C" __global__ void coarse_ax(const float* ax,float* c,int w,int h,int cw,int ch){
    int X=blockIdx.x*blockDim.x+threadIdx.x, Y=blockIdx.y*blockDim.y+threadIdx.y;
    if(X>=cw||Y>=ch) return; float s=0.f;
    if(X+1<cw) for(int y=2*Y;y<min(2*Y+2,h);y++) s+=ax[(size_t)y*w+2*X+1];
    c[(size_t)Y*cw+X]=s;
}
extern "C" __global__ void coarse_ay(const float* ay,float* c,int w,int h,int cw,int ch){
    int X=blockIdx.x*blockDim.x+threadIdx.x, Y=blockIdx.y*blockDim.y+threadIdx.y;
    if(X>=cw||Y>=ch) return; float s=0.f;
    if(Y+1<ch){ int y=2*Y+1; for(int x=2*X;x<min(2*X+2,w);x++) s+=ay[(size_t)y*w+x]; }
    c[(size_t)Y*cw+X]=s;
}
extern "C" __global__ void dinvk(const float* wd,const float* ax,const float* ay,float* dinv,int w,int h){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; size_t k=(size_t)y*w+x; float v=wd[k];
    if(x>0) v+=ax[k-1]; if(x+1<w) v+=ax[k]; if(y>0) v+=ay[k-w]; if(y+1<h) v+=ay[k];
    dinv[k]=1.f/fmaxf(v,1e-12f);
}
// mode 0: o = A x; 1: o = b - A x; 2: o = x + omega dinv (b - A x)
extern "C" __global__ void stencil(const float* x,const float* b,const float* wd,const float* ax,const float* ay,const float* dinv,float* o,int w,int h,int mode,float omega){
    int i=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=w||y>=h) return; size_t k=(size_t)y*w+i; float xk=x[k], v=wd[k]*xk;
    if(i>0) v+=ax[k-1]*(xk-x[k-1]); if(i+1<w) v+=ax[k]*(xk-x[k+1]);
    if(y>0) v+=ay[k-w]*(xk-x[k-w]); if(y+1<h) v+=ay[k]*(xk-x[k+w]);
    o[k]=mode==0?v:(mode==1?b[k]-v:xk+omega*dinv[k]*(b[k]-v));
}
extern "C" __global__ void jac0(const float* b,const float* dinv,float* x,int n,float omega){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=omega*dinv[i]*b[i]; }
extern "C" __global__ void restrictk(const float* r,float* bc,int w,int h,int cw,int ch){
    int X=blockIdx.x*blockDim.x+threadIdx.x, Y=blockIdx.y*blockDim.y+threadIdx.y;
    if(X>=cw||Y>=ch) return; float s=0.f;
    for(int y=2*Y;y<min(2*Y+2,h);y++) for(int x=2*X;x<min(2*X+2,w);x++) s+=r[(size_t)y*w+x];
    bc[(size_t)Y*cw+X]=s;
}
extern "C" __global__ void prolongk(float* x,const float* xc,int w,int h,int cw){
    int i=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=w||y>=h) return; x[(size_t)y*w+i]+=xc[(size_t)(y/2)*cw+i/2];
}
// dot product in double: block reduction, one atomic per block into acc
extern "C" __global__ void dotk(const float* a,const float* b,double* acc,int n){
    __shared__ double sh[256]; int t=threadIdx.x; int i=blockIdx.x*blockDim.x+t; double s=0.0;
    for(;i<n;i+=gridDim.x*blockDim.x) s+=(double)a[i]*(double)b[i];
    sh[t]=s; __syncthreads();
    for(int k=128;k>0;k>>=1){ if(t<k) sh[t]+=sh[t+k]; __syncthreads(); }
    if(t==0) atomicAdd(acc,sh[0]);
}
// scal = [rz, alpha, beta, pap, rr]; mode 0: rz = acc; 1: pap = acc, alpha = rz/pap;
// 2: beta = acc/rz, rz = acc; 3: rr = acc. Clears acc.
extern "C" __global__ void scalk(double* acc,double* scal,int mode){
    double v=acc[0]; acc[0]=0.0;
    if(mode==0) scal[0]=v;
    else if(mode==1){ scal[3]=v; scal[1]=v>0.0?scal[0]/v:0.0; }
    else if(mode==2){ scal[2]=scal[0]>0.0?v/scal[0]:0.0; scal[0]=v; }
    else scal[4]=v;
}
extern "C" __global__ void axpyk(float* y,const float* x,const double* scal,int n,float sign){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=sign*(float)scal[1]*x[i]; }
extern "C" __global__ void pupdk(float* p,const float* z,const double* scal,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=z[i]+(float)scal[2]*p[i]; }
// bilinear from the dw x dh grid of k-pixel blocks (samples at block centres) to w x h
__device__ __forceinline__ float bil(const float* g,int dw,int dh,int k,int x,int y){
    float inv=1.f/(float)k;
    float fy=fminf(fmaxf(((float)y+0.5f)*inv-0.5f,0.f),(float)(dh-1)), fx=fminf(fmaxf(((float)x+0.5f)*inv-0.5f,0.f),(float)(dw-1));
    int y0=(int)fy, x0=(int)fx, y1=min(y0+1,dh-1), x1=min(x0+1,dw-1); float ty=fy-(float)y0, tx=fx-(float)x0;
    const float* r0=g+(size_t)y0*dw; const float* r1=g+(size_t)y1*dw;
    float top=r0[x0]+(r0[x1]-r0[x0])*tx, bot=r1[x0]+(r1[x1]-r1[x0])*tx;
    return top+(bot-top)*ty;
}
extern "C" __global__ void upbil(const float* g,float* o,int w,int h,int dw,int dh,int k){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; o[(size_t)y*w+x]=bil(g,dw,dh,k,x,y);
}
// guided upsampling: (bilinear mean_a) * luma + (bilinear mean_b), clamped to [0, maxd]
extern "C" __global__ void upapply(const float* ma,const float* mb,const float* yf,float* o,int w,int h,int dw,int dh,int k,float maxd){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; size_t i=(size_t)y*w+x;
    o[i]=fminf(fmaxf(bil(ma,dw,dh,k,x,y)*yf[i]+bil(mb,dw,dh,k,x,y),0.f),maxd);
}
"#;

const DEPTH_KERNELS: [&str; 30] = [
    "bmeank", "boxH", "boxV", "mulk", "gfvar", "gfab", "gfapply", "peak_init", "peak_push", "peak_finish", "median3k", "wdk", "robustk",
    "edgew", "fgs_rows", "fgs_cols", "coarse_wd", "coarse_ax", "coarse_ay", "dinvk", "stencil", "jac0", "restrictk", "prolongk", "dotk",
    "scalk", "axpyk", "pupdk", "upbil", "upapply",
];

/// A kernel argument: a plane, a double buffer, an int or a float. The
/// depth pass's launches read and write planes of one pool, so they take
/// every plane by shared reference (on one stream cudarc records nothing
/// for a mutable one either).
#[derive(Clone, Copy)]
enum A<'a> {
    B(&'a CudaSlice<f32>),
    D(&'a CudaSlice<f64>),
    I(i32),
    F(f32),
}

fn launch(g: &G, name: &'static str, cfg: LaunchConfig, args: &[A]) {
    let mut b = g.s.launch_builder(g.f(name));
    for a in args {
        match a {
            A::B(x) => b.arg(*x),
            A::D(x) => b.arg(*x),
            A::I(v) => b.arg(v),
            A::F(v) => b.arg(v),
        };
    }
    unsafe { b.launch(cfg).unwrap() };
}

/// One grid of the WLS multigrid hierarchy on the device (`depth::MgLevel`).
struct DLevel {
    w: usize,
    h: usize,
    wd: CudaSlice<f32>,
    ax: CudaSlice<f32>,
    ay: CudaSlice<f32>,
    dinv: CudaSlice<f32>,
    x: CudaSlice<f32>,
    b: CudaSlice<f32>,
    r: CudaSlice<f32>,
}

/// The WLS solver on the device (`depth::WlsSolver`): the hierarchy, the
/// sweeps' scratch and the CG vectors, kept for the reweighted second solve.
struct DWls {
    levels: Vec<DLevel>,
    cp: CudaSlice<f32>,
    dp: CudaSlice<f32>,
    f: CudaSlice<f32>,
    r: CudaSlice<f32>,
    z: CudaSlice<f32>,
    p: CudaSlice<f32>,
    ap: CudaSlice<f32>,
    acc: CudaSlice<f64>,
    scal: CudaSlice<f64>,
}

impl DWls {
    fn new(g: &G, guide: &CudaSlice<f32>, w: usize, h: usize, lambda: f32, sigma_c: f32) -> Result<DWls, String> {
        let al = |n: usize| g.s.alloc_zeros::<f32>(n).map_err(|e| format!("cuda alloc: {e:?}"));
        let mk = |w: usize, h: usize| -> Result<DLevel, String> {
            let n = w * h;
            Ok(DLevel { w, h, wd: al(n)?, ax: al(n)?, ay: al(n)?, dinv: al(n)?, x: al(n)?, b: al(n)?, r: al(n)? })
        };
        let mut levels = vec![mk(w, h)?];
        launch(g, "edgew", cfg2(w, h), &[A::B(guide), A::B(&levels[0].ax), A::B(&levels[0].ay), A::I(w as i32), A::I(h as i32), A::F(-1.0 / sigma_c.max(1e-6)), A::F(lambda)]);
        while levels.last().unwrap().w.max(levels.last().unwrap().h) > crate::depth::MG_MIN {
            let f = levels.last().unwrap();
            let (cw, ch) = (half(f.w), half(f.h));
            let c = mk(cw, ch)?;
            let geo = [A::I(f.w as i32), A::I(f.h as i32), A::I(cw as i32), A::I(ch as i32)];
            launch(g, "coarse_ax", cfg2(cw, ch), &[A::B(&f.ax), A::B(&c.ax), geo[0], geo[1], geo[2], geo[3]]);
            launch(g, "coarse_ay", cfg2(cw, ch), &[A::B(&f.ay), A::B(&c.ay), geo[0], geo[1], geo[2], geo[3]]);
            levels.push(c);
        }
        let n = w * h;
        Ok(DWls {
            levels, cp: al(n)?, dp: al(n)?, f: al(n)?, r: al(n)?, z: al(n)?, p: al(n)?, ap: al(n)?,
            acc: g.s.alloc_zeros::<f64>(1).map_err(|e| format!("cuda alloc: {e:?}"))?,
            scal: g.s.alloc_zeros::<f64>(5).map_err(|e| format!("cuda alloc: {e:?}"))?,
        })
    }

    /// The data weights are in `levels[0].wd`: aggregate them down the
    /// hierarchy and take every grid's diagonal.
    fn set_weights(&self, g: &G) {
        for l in 0..self.levels.len() {
            let c = &self.levels[l];
            if l > 0 {
                let f = &self.levels[l - 1];
                launch(g, "coarse_wd", cfg2(c.w, c.h), &[A::B(&f.wd), A::B(&c.wd), A::I(f.w as i32), A::I(f.h as i32), A::I(c.w as i32), A::I(c.h as i32)]);
            }
            launch(g, "dinvk", cfg2(c.w, c.h), &[A::B(&c.wd), A::B(&c.ax), A::B(&c.ay), A::B(&c.dinv), A::I(c.w as i32), A::I(c.h as i32)]);
        }
    }

    /// `o = f(A x)` on grid `l` by `stencil`'s mode.
    fn stencil(&self, g: &G, l: usize, x: &CudaSlice<f32>, b: &CudaSlice<f32>, o: &CudaSlice<f32>, mode: i32) {
        let c = &self.levels[l];
        launch(g, "stencil", cfg2(c.w, c.h), &[A::B(x), A::B(b), A::B(&c.wd), A::B(&c.ax), A::B(&c.ay), A::B(&c.dinv), A::B(o), A::I(c.w as i32), A::I(c.h as i32), A::I(mode), A::F(crate::depth::MG_OMEGA)]);
    }

    /// One damped-Jacobi sweep on grid `l` (`x` from zero when `zero_start`).
    fn jacobi(&mut self, g: &G, l: usize, zero_start: bool) {
        let c = &self.levels[l];
        let n = c.w * c.h;
        if zero_start {
            launch(g, "jac0", cfg1(n), &[A::B(&c.b), A::B(&c.dinv), A::B(&c.x), A::I(n as i32), A::F(crate::depth::MG_OMEGA)]);
        } else {
            self.stencil(g, l, &c.x, &c.b, &c.r, 2);
            let c = &mut self.levels[l];
            std::mem::swap(&mut c.x, &mut c.r);
        }
    }

    /// One V-cycle from grid `l` (its `b` set) into its `x`.
    fn vcycle(&mut self, g: &G, l: usize) {
        use crate::depth::{MG_COARSE, MG_POST, MG_PRE};
        if l + 1 == self.levels.len() {
            for k in 0..MG_COARSE {
                self.jacobi(g, l, k == 0);
            }
            return;
        }
        for k in 0..MG_PRE {
            self.jacobi(g, l, k == 0);
        }
        {
            let c = &self.levels[l];
            self.stencil(g, l, &c.x, &c.b, &c.r, 1);
            let d = &self.levels[l + 1];
            launch(g, "restrictk", cfg2(d.w, d.h), &[A::B(&c.r), A::B(&d.b), A::I(c.w as i32), A::I(c.h as i32), A::I(d.w as i32), A::I(d.h as i32)]);
        }
        self.vcycle(g, l + 1);
        {
            let (c, d) = (&self.levels[l], &self.levels[l + 1]);
            launch(g, "prolongk", cfg2(c.w, c.h), &[A::B(&c.x), A::B(&d.x), A::I(c.w as i32), A::I(c.h as i32), A::I(d.w as i32)]);
        }
        for _ in 0..MG_POST {
            self.jacobi(g, l, false);
        }
    }

    fn dot(&self, g: &G, a: &CudaSlice<f32>, b: &CudaSlice<f32>, n: usize, mode: i32) {
        let blocks = (n.div_ceil(256)).min(1024) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        launch(g, "dotk", cfg, &[A::B(a), A::B(b), A::D(&self.acc), A::I(n as i32)]);
        launch(g, "scalk", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, &[A::D(&self.acc), A::D(&self.scal), A::I(mode)]);
    }

    /// Solve for the data `d` into `u` (the FGS guess, then the
    /// multigrid-preconditioned CG); the final relative residual and the
    /// iterations taken.
    fn solve(&mut self, g: &G, d: &CudaSlice<f32>, u: &CudaSlice<f32>, max_iters: usize) -> Result<(f32, usize), String> {
        let (w, h) = (self.levels[0].w, self.levels[0].h);
        let n = w * h;
        let (wi, hi, ni) = (w as i32, h as i32, n as i32);
        let e = |r: Result<(), cudarc::driver::DriverError>| r.map_err(|e| format!("{e:?}"));
        // --- the FGS guess: the CPU's schedule, lambda_t as a ratio of the
        // lambda folded into the edge weights
        const T: usize = 3;
        e(g.s.memcpy_dtod(d, &mut self.f))?;
        for t in 1..=T {
            let lam_t = 1.5 * 4f32.powi((T - t) as i32) / (4f32.powi(T as i32) - 1.0);
            let l0 = &self.levels[0];
            let unit = (t != 1) as i32;
            launch(g, "fgs_rows", cfg1(h), &[A::B(&self.f), A::B(u), A::B(&l0.wd), A::B(&l0.ax), A::B(&self.cp), A::B(&self.dp), A::I(wi), A::I(hi), A::F(lam_t), A::I(unit)]);
            e(g.s.memcpy_dtod(u, &mut self.f))?;
            launch(g, "fgs_cols", cfg1(w), &[A::B(&self.f), A::B(u), A::B(&l0.wd), A::B(&l0.ay), A::B(&self.cp), A::B(&self.dp), A::I(wi), A::I(hi), A::F(lam_t), A::I(1)]);
            e(g.s.memcpy_dtod(u, &mut self.f))?;
        }
        // --- CG on (W + lambda L) u = W d, preconditioned by a V-cycle
        launch(g, "mulk", cfg1(n), &[A::B(&self.levels[0].wd), A::B(d), A::B(&self.f), A::I(ni)]); // b
        self.dot(g, &self.f, &self.f, n, 3);
        let bnorm = (g.s.memcpy_dtov(&self.scal).map_err(|e| format!("{e:?}"))?[4]).sqrt().max(1e-30);
        self.stencil(g, 0, u, &self.f, &self.r, 1);
        let precond = |s: &mut DWls, g: &G| -> Result<(), String> {
            e(g.s.memcpy_dtod(&s.r, &mut s.levels[0].b))?;
            s.vcycle(g, 0);
            e(g.s.memcpy_dtod(&s.levels[0].x, &mut s.z))
        };
        precond(self, g)?;
        e(g.s.memcpy_dtod(&self.z, &mut self.p))?;
        self.dot(g, &self.r, &self.z, n, 0);
        self.dot(g, &self.r, &self.r, n, 3);
        let rel = |s: &DWls| -> Result<f32, String> { Ok((g.s.memcpy_dtov(&s.scal).map_err(|e| format!("{e:?}"))?[4].sqrt() / bnorm) as f32) };
        let mut res = rel(self)?;
        let mut iters = 0;
        for _ in 0..max_iters {
            if res < 1e-5 {
                break;
            }
            iters += 1;
            self.stencil(g, 0, &self.p, &self.f, &self.ap, 0);
            self.dot(g, &self.p, &self.ap, n, 1);
            launch(g, "axpyk", cfg1(n), &[A::B(u), A::B(&self.p), A::D(&self.scal), A::I(ni), A::F(1.0)]);
            launch(g, "axpyk", cfg1(n), &[A::B(&self.r), A::B(&self.ap), A::D(&self.scal), A::I(ni), A::F(-1.0)]);
            self.dot(g, &self.r, &self.r, n, 3);
            res = rel(self)?;
            if res < 1e-5 {
                break;
            }
            precond(self, g)?;
            self.dot(g, &self.r, &self.z, n, 2);
            launch(g, "pupdk", cfg1(n), &[A::B(&self.p), A::B(&self.z), A::D(&self.scal), A::I(ni)]);
        }
        Ok((res, iters))
    }
}

/// The depth pass on the device (`depth::depth_from_slices`): the slices,
/// each uploaded as it comes, are aggregated with the guided filter and
/// folded into the peak tracker; the sub-frame depth is regularised by the
/// WLS solve and upsampled on the fused luma, all on the device. Only the
/// noise floor and the confidence's 90th percentile (medians of a
/// subsample) are taken on the host, and the depth and confidence maps
/// come back.
pub fn depth_from_slices(
    slices: &mut dyn Iterator<Item = Result<Vec<f32>, String>>,
    n: usize,
    fused: &Img3,
    p: &DepthParams,
    log: &mut dyn FnMut(String),
) -> Result<crate::depth::DepthMap, String> {
    use crate::depth::{Upsample, luma, normalize_conf};
    let t = std::time::Instant::now();
    let (ctx, g) = init_gpu(DEPTH_SRC, &DEPTH_KERNELS)?;
    let _keep = ctx;
    let (w, h) = (fused.w, fused.h);
    let k = 1usize << p.scale;
    let (dw, dh) = (blocks(w, k), blocks(h, k));
    let m = dw * dh;
    let (wi, hi, dwi, dhi, mi) = (w as i32, h as i32, dw as i32, dh as i32, m as i32);
    let e = |r: Result<(), cudarc::driver::DriverError>| r.map_err(|e| format!("{e:?}"));
    let al = |n: usize| g.s.alloc_zeros::<f32>(n).map_err(|e| format!("cuda alloc: {e:?}"));
    // the guide: the fused luma, block-averaged to the working grid
    let yfull = g.s.memcpy_stod(&luma(fused)).map_err(|e| format!("{e:?}"))?;
    let guide = al(m)?;
    launch(&g, "bmeank", cfg2(dw, dh), &[A::B(&yfull), A::B(&guide), A::I(wi), A::I(hi), A::I(k as i32), A::I(dwi), A::I(dhi)]);
    let bt = al(m)?;
    let boxf = |src: &CudaSlice<f32>, dst: &CudaSlice<f32>, r: usize| {
        launch(&g, "boxH", cfg2(dw, dh), &[A::B(src), A::B(&bt), A::I(dwi), A::I(dhi), A::I(r as i32)]);
        launch(&g, "boxV", cfg2(dw, dh), &[A::B(&bt), A::B(dst), A::I(dwi), A::I(dhi), A::I(r as i32)]);
    };
    // the guided filter: the guide's mean and variance at radius r, then
    // per plane the coefficients' means into (ma, mb)
    let (mean_i, var_i) = (al(m)?, al(m)?);
    let [t0, t1, t2, t3] = [al(m)?, al(m)?, al(m)?, al(m)?];
    let stats = |r: usize| {
        boxf(&guide, &mean_i, r);
        launch(&g, "mulk", cfg1(m), &[A::B(&guide), A::B(&guide), A::B(&t0), A::I(mi)]);
        boxf(&t0, &t1, r);
        launch(&g, "gfvar", cfg1(m), &[A::B(&mean_i), A::B(&t1), A::B(&var_i), A::I(mi)]);
    };
    // ma -> t1, mb -> t0
    let coeffs = |src: &CudaSlice<f32>, r: usize, eps: f32| {
        boxf(src, &t0, r); // mean_p
        launch(&g, "mulk", cfg1(m), &[A::B(&guide), A::B(src), A::B(&t1), A::I(mi)]);
        boxf(&t1, &t2, r); // corr_Ip
        launch(&g, "gfab", cfg1(m), &[A::B(&t0), A::B(&t2), A::B(&mean_i), A::B(&var_i), A::B(&t2), A::B(&t3), A::F(eps), A::I(mi)]);
        boxf(&t2, &t1, r); // mean a
        boxf(&t3, &t0, r); // mean b
    };
    log(format!(
        "depth: {n} frames, focus {:?}, working grid {dw}x{dh} (1/{k}), aggregation r={} eps={} (GPU, {:.2}s to start)",
        p.focus, p.agg_radius, p.agg_eps, t.elapsed().as_secs_f64()
    ));
    if p.agg_radius > 0 {
        stats(p.agg_radius);
    }
    let state = al(9 * m)?;
    launch(&g, "peak_init", cfg1(m), &[A::B(&state), A::I(mi)]);
    let mut sl = al(m)?;
    let agg = al(m)?;
    for i in 0..n {
        let c = slices.next().ok_or_else(|| format!("depth: slice {i} of {n} missing"))??;
        if c.len() != m {
            return Err(format!("depth: slice {i} has {} cells, the working grid {dw}x{dh}", c.len()));
        }
        e(g.s.memcpy_htod(&c, &mut sl))?;
        drop(c);
        if p.agg_radius > 0 {
            coeffs(&sl, p.agg_radius, p.agg_eps);
            // the guided filter can undershoot; the profile statistics assume >= 0
            launch(&g, "gfapply", cfg1(m), &[A::B(&t1), A::B(&t0), A::B(&guide), A::B(&agg), A::I(mi), A::I(1)]);
            launch(&g, "peak_push", cfg1(m), &[A::B(&agg), A::B(&state), A::I(mi), A::I(i as i32)]);
        } else {
            launch(&g, "peak_push", cfg1(m), &[A::B(&sl), A::B(&state), A::I(mi), A::I(i as i32)]);
        }
    }
    e(g.s.synchronize())?;
    log(format!("depth: {n} slices aggregated  ({:.1}s)", t.elapsed().as_secs_f64()));
    // noise floor: median of the per-pixel profile minimum (subsampled)
    let cmin = g.s.memcpy_dtov(&state.slice(7 * m..8 * m)).map_err(|e| format!("{e:?}"))?;
    let mut mins: Vec<f32> = cmin.iter().step_by(7).copied().collect();
    let mid = mins.len() / 2;
    let floor = *mins.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
    let (mut d, mut conf) = (sl, agg); // reused: the raw depth and confidence
    launch(&g, "peak_finish", cfg1(m), &[A::B(&state), A::B(&t2), A::B(&conf), A::I(mi), A::I(n as i32), A::F(floor), A::F(p.gate)]);
    if p.median {
        launch(&g, "median3k", cfg2(dw, dh), &[A::B(&t2), A::B(&d), A::I(dwi), A::I(dhi)]);
    } else {
        e(g.s.memcpy_dtod(&t2, &mut d))?;
    }
    drop(state);
    let mut conf_h = g.s.memcpy_dtov(&conf).map_err(|e| format!("{e:?}"))?;
    let p90 = normalize_conf(&mut conf_h);
    let mean_conf = conf_h.iter().map(|&c| c as f64).sum::<f64>() / conf_h.len() as f64;
    log(format!("depth: peaks found, confidence p90 {p90:.3}, mean (normalised) {mean_conf:.3}  ({:.1}s)", t.elapsed().as_secs_f64()));
    // the normalised confidence, the data weight and the map that comes back
    e(g.s.memcpy_htod(&conf_h, &mut conf))?;
    let mut u = al(m)?;
    let (rel, iters) = if p.lambda > 0.0 {
        const EPS_DATA: f32 = 1e-4;
        let mut wls = DWls::new(&g, &guide, dw, dh, p.lambda, p.sigma_c)?;
        launch(&g, "wdk", cfg1(m), &[A::B(&conf), A::B(&wls.levels[0].wd), A::I(mi), A::F(1.0), A::F(EPS_DATA)]);
        wls.set_weights(&g);
        let (rel, iters) = wls.solve(&g, &d, &u, p.cg_iters)?;
        if p.robust > 0.0 {
            // one IRLS step with a Huber loss on the data residual
            launch(&g, "robustk", cfg1(m), &[A::B(&conf), A::B(&d), A::B(&u), A::B(&wls.levels[0].wd), A::I(mi), A::F(p.robust), A::F(EPS_DATA)]);
            wls.set_weights(&g);
            let (rel2, iters2) = wls.solve(&g, &d, &u, p.cg_iters)?;
            log(format!("depth: robust reweighting (tau={} frames), first solve {iters} iterations, residual {rel:.1e}", p.robust));
            (rel2, iters2)
        } else {
            (rel, iters)
        }
    } else {
        e(g.s.memcpy_dtod(&d, &mut u))?;
        (0.0, 0)
    };
    e(g.s.synchronize())?;
    log(format!("depth: WLS lambda={} sigma_c={} solved in {iters} iterations, residual {rel:.1e}  ({:.1}s)", p.lambda, p.sigma_c, t.elapsed().as_secs_f64()));
    // upsampling to full resolution
    let max_d = (n - 1) as f32;
    let full = g.s.alloc_zeros::<f32>(w * h).map_err(|e| format!("cuda alloc: {e:?}"))?;
    match p.upsample {
        Upsample::Guided { radius, eps } => {
            stats(radius);
            coeffs(&u, radius, eps);
            launch(&g, "upapply", cfg2(w, h), &[A::B(&t1), A::B(&t0), A::B(&yfull), A::B(&full), A::I(wi), A::I(hi), A::I(dwi), A::I(dhi), A::I(k as i32), A::F(max_d)]);
        }
        Upsample::Bilinear => {
            // a = 0, b = u: the bilinear map clamped like the guided one
            launch(&g, "wdk", cfg1(m), &[A::B(&u), A::B(&t1), A::I(mi), A::F(0.0), A::F(0.0)]);
            launch(&g, "upapply", cfg2(w, h), &[A::B(&t1), A::B(&u), A::B(&yfull), A::B(&full), A::I(wi), A::I(hi), A::I(dwi), A::I(dhi), A::I(k as i32), A::F(max_d)]);
        }
    }
    let depth = g.s.memcpy_dtov(&full).map_err(|e| format!("{e:?}"))?;
    let conf_full = if k == 1 {
        conf_h
    } else {
        launch(&g, "upbil", cfg2(w, h), &[A::B(&conf), A::B(&full), A::I(wi), A::I(hi), A::I(dwi), A::I(dhi), A::I(k as i32)]);
        g.s.memcpy_dtov(&full).map_err(|e| format!("{e:?}"))?
    };
    log(format!("depth: upsampled to {w}x{h} ({:?})  ({:.1}s)", p.upsample, t.elapsed().as_secs_f64()));
    Ok(crate::depth::DepthMap { depth, conf: conf_full, w, h, dw, dh, floor: cmin })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::{DepthParams, focus_slice};

    /// The device depth pass against the CPU's on a synthetic stack: two
    /// frames sharp on either half of a blurred-vs-sharp checker, the maps
    /// agreeing to float rounding except where a near-tie in the peak
    /// search falls the other way. Needs a CUDA device (`--features gpu`).
    #[test]
    fn gpu_depth_matches_cpu() {
        let (w, h) = (160, 120);
        let mut img = Img3::zeros(w, h);
        let mut s = 7u64;
        for i in 0..w * h {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (x, y) = (i % w, i / w);
            let v = (((x / 6 + y / 6) % 2) as f32) * 0.6 + 0.2 + ((s >> 33) % 100) as f32 * 0.001;
            for c in 0..3 {
                img.p[c][i] = v;
            }
        }
        let blur = |src: &Img3| {
            let mut out = src.clone();
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        let mut a = 0.0;
                        for dy in -4isize..=4 {
                            for dx in -4isize..=4 {
                                a += src.p[c][(y as isize + dy).clamp(0, h as isize - 1) as usize * w + (x as isize + dx).clamp(0, w as isize - 1) as usize];
                            }
                        }
                        out.p[c][y * w + x] = a / 81.0;
                    }
                }
            }
            out
        };
        let soft = blur(&img);
        let (mut a, mut b) = (img.clone(), img.clone());
        for i in 0..w * h {
            for c in 0..3 {
                if i % w >= w / 2 { a.p[c][i] = soft.p[c][i] } else { b.p[c][i] = soft.p[c][i] }
            }
        }
        let p = DepthParams { lambda: 5.0, ..Default::default() };
        let frames = [a, b];
        let slices: Vec<Vec<f32>> = frames.iter().map(|f| focus_slice(f, &p)).collect();
        let n = slices.len();
        let cpu = crate::depth::depth_from_slices(&mut slices.clone().into_iter().map(Ok), n, &img, &p, &mut |_| {}).unwrap();
        let gpu = depth_from_slices(&mut slices.into_iter().map(Ok), n, &img, &p, &mut |_| {}).unwrap();
        assert_eq!((gpu.w, gpu.h, gpu.dw, gpu.dh), (cpu.w, cpu.h, cpu.dw, cpu.dh));
        let far = cpu.depth.iter().zip(&gpu.depth).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
        assert!(far <= w * h / 1000, "{far} depth pixels differ by more than 1e-3 frames");
        let far = cpu.conf.iter().zip(&gpu.conf).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
        assert!(far <= w * h / 1000, "{far} confidence pixels differ by more than 1e-3");
        // and the map means something: near frame 0 on the left, frame 1 on the right
        let left = (0..h).map(|y| gpu.depth[y * w + w / 4]).sum::<f32>() / h as f32;
        let right = (0..h).map(|y| gpu.depth[y * w + 3 * w / 4]).sum::<f32>() / h as f32;
        assert!(left < 0.25 && right > 0.75, "left {left} right {right}");
    }
}
