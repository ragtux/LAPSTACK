// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

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
//! cudarc with `dynamic-loading` (libcuda + libnvrtc found at runtime, no
//! toolkit at build time), kernels compiled by NVRTC once per run.
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

const SRC: &str = r#"
#define K0 (1.0f/16.0f)
#define K1 (4.0f/16.0f)
#define K2 (6.0f/16.0f)
#define E0 (2.0f*K0)
#define E1 (2.0f*K2)
#define OD (2.0f*K1)
__device__ __forceinline__ int refl(int i, int n){
    if(n==1) return 0;
    while(i<0 || i>=n){ if(i<0) i=-i; else i=2*(n-1)-i; }
    return i;
}
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
// block mean to the depth pass's working grid (depth::block_mean)
extern "C" __global__ void bmeank(const float* in,float* out,int w,int h,int k,int dw,int dh){
    int ox=blockIdx.x*blockDim.x+threadIdx.x, oy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ox>=dw||oy>=dh) return; int x0=ox*k, x1=min(x0+k,w), y0=oy*k, y1=min(y0+k,h); float a=0.f;
    for(int y=y0;y<y1;y++) for(int x=x0;x<x1;x++) a+=in[(size_t)y*w+x];
    out[(size_t)oy*dw+ox]=a/(float)((y1-y0)*(x1-x0));
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
fn init_gpu(names: &[&'static str]) -> Result<(Arc<CudaContext>, G), String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed (is nvidia_uvm loaded?): {e:?}"))?;
    let s = ctx.default_stream();
    // arch compute_86 so warp_cost can use double atomicAdd.
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_86"), ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).map_err(|e| format!("nvrtc: {e:?}"))?;
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
        let (ctx, g) = init_gpu(&KERNELS)?;
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
        let (ctx, g) = init_gpu(&["warp_cost", "lumak", "burtH", "burtV", "warpk", "gridk", "bsumk", "gaink", "rdfk", "smlk", "bmeank"])?;
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
