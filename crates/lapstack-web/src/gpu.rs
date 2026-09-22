//! wgpu layer: device setup, one bind-group layout shared by every kernel, a
//! dynamic-offset uniform ring, a command recorder that auto-flushes, and
//! async buffer readbacks.

use futures_channel::oneshot;
use std::cell::RefCell;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Kernel parameters (mirrors `struct P` in shaders.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct P {
    pub w: u32,
    pub h: u32,
    pub ow: u32,
    pub oh: u32,
    pub off_in: u32,
    pub off_out: u32,
    pub klen: u32,
    pub flag: u32,
    pub f0: f32,
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
}

const SLOT: u64 = 256;
/// Frame uploads and readbacks move through the browser in slices of this
/// size (see `Gpu::upload` and `Gpu::read_at`); the upload ring has this many slots.
const XFER_SLICE: usize = 16 << 20;
const UPLOAD_SLOTS: usize = 4;
const SLOTS: usize = 2048;
const KERNELS: [&str; 57] = [
    "red_h", "red_v", "exp_h", "exp_v", "energy", "win_h", "win_v", "sel", "wgt", "wacc", "wnorm", "fill", "clamp01", "copy_plane",
    "warp", "cost", "to_rgba8", "to_rgb16", "proxy", "luma_u16", "luma_f32", "down1", "blk_mean", "bright", "gain3",
    // depth from focus (depth.rs)
    "conv_taps", "box_h", "box_v", "mul", "gf_ab", "gf_apply", "gf_var", "unpack_u16", "pack_u16",
    "peak_init", "peak_push", "peak_finish", "median3", "scale_clamp", "edge_w", "fgs_rows", "fgs_cols",
    "cg_matvec", "cg_resid", "cg_zp", "dot_partial", "reduce_scal", "cg_axpy_u", "cg_update_rz",
    "cg_update_p", "robust_w", "up_apply",
    // depth-map rendering, In focus
    "dmap_acc", "dmap_norm", "wav_acc", "wav_norm", "focus_out",
];

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    pub info: wgpu::AdapterInfo,
    bgl: wgpu::BindGroupLayout,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    uni: wgpu::Buffer,
    dummies: Vec<wgpu::Buffer>,
    /// upload staging ring (see `upload`), and each slot's pending map, if any
    ring: RefCell<Vec<wgpu::Buffer>>,
    ring_maps: RefCell<Vec<Option<oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>>>>,
}

/// Yield to the event loop: a task, not a microtask, since the browser flushes
/// queued GPU commands and runs other work only between tasks. A message port
/// rather than setTimeout(0), which is clamped to 4 ms once timers nest.
pub async fn yield_now() {
    thread_local! { static CHAN: web_sys::MessageChannel = web_sys::MessageChannel::new().unwrap(); }
    let p = js_sys::Promise::new(&mut |resolve, _| {
        CHAN.with(|c| {
            c.port1().set_onmessage(Some(&resolve));
            let _ = c.port2().post_message(&wasm_bindgen::JsValue::NULL);
        });
    });
    let _ = wasm_bindgen_futures::JsFuture::from(p).await;
}

pub fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Workgroup grid for a 1-D kernel over `n` elements (256 per group).
pub fn grid1(n: usize) -> (u32, u32) {
    let groups = ceil_div(n as u32, 256).max(1);
    let gx = groups.min(32768);
    (gx, ceil_div(groups, gx))
}
pub fn grid2(w: usize, h: usize) -> (u32, u32) {
    (ceil_div(w as u32, 16).max(1), ceil_div(h as u32, 16).max(1))
}

impl Gpu {
    pub async fn new() -> Result<Gpu, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(desc);
        // Chrome can answer the very first requestAdapter() of a process with
        // null while its GPU process is still coming up; retry a few times.
        let mut adapter = None;
        let mut last = String::new();
        for _ in 0..4 {
            match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
            {
                Ok(a) => {
                    adapter = Some(a);
                    break;
                }
                Err(e) => {
                    last = format!("{e:?}");
                    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&wasm_bindgen::JsValue::NULL)).await.ok();
                }
            }
        }
        let adapter = adapter.ok_or_else(|| format!("no WebGPU adapter: {last}"))?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lapstack"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device: {e:?}"))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lapstack kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lapstack bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(48),
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, true),
                storage(6, true),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mut pipes = HashMap::new();
        for &k in &KERNELS {
            if pipes.contains_key(k) {
                continue;
            }
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(k),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.insert(k, p);
        }
        let uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform ring"),
            size: SLOT * SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummies = (0..6)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("dummy{i}")),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let ring = (0..UPLOAD_SLOTS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upload ring"),
                    size: XFER_SLICE as u64,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                })
            })
            .collect();
        Ok(Gpu {
            device, queue, limits, info, bgl, pipes, uni, dummies,
            ring: RefCell::new(ring), ring_maps: RefCell::new((0..UPLOAD_SLOTS).map(|_| None).collect()),
        })
    }

    pub fn buffer(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16).div_ceil(4) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    pub fn buffer_f32(&self, label: &str, n: usize) -> wgpu::Buffer {
        self.buffer(label, n as u64 * 4)
    }
    pub fn buffer_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }
    pub fn staging(&self, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes.max(16).div_ceil(4) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn rec(&self) -> Rec<'_> {
        Rec { gpu: self, enc: Some(self.device.create_command_encoder(&Default::default())), n: 0, keep: Vec::new() }
    }

    /// Upload a whole frame. One writeBuffer of a full 16-bit frame (274 MB at
    /// 45 MP) is copied by Chrome's GPU process on the thread that also composites
    /// the page, and every animation and scroll stalls for the duration (~150 ms);
    /// smaller writeBuffer calls each pay for a fresh shared-memory block. So the
    /// frame goes through a ring of persistently mapped staging buffers: each slice
    /// is written into a mapped slot, unmapped, copied to `dst` on the GPU, and the
    /// slot mapped again for its next turn. The copies are short, and waiting for
    /// a slot's map yields the event loop.
    pub async fn upload(&self, dst: &wgpu::Buffer, data: &[u8]) -> Result<(), String> {
        let ring = self.ring.borrow();
        let mut maps = self.ring_maps.borrow_mut();
        for (i, chunk) in data.chunks(XFER_SLICE).enumerate() {
            let slot = i % ring.len();
            if let Some(rx) = maps[slot].take() {
                rx.await.map_err(|_| "map callback dropped".to_string())?.map_err(|e| format!("map: {e:?}"))?;
            }
            let len = chunk.len();
            let st = &ring[slot];
            {
                let mut view = st.slice(..(len.div_ceil(8) * 8) as u64).get_mapped_range_mut().map_err(|e| format!("{e:?}"))?;
                view.slice(..len).copy_from_slice(chunk);
            }
            st.unmap();
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(st, 0, dst, (i * XFER_SLICE) as u64, (len.div_ceil(4) * 4) as u64);
            self.queue.submit([enc.finish()]);
            let (tx, rx) = oneshot::channel();
            st.slice(..).map_async(wgpu::MapMode::Write, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::Poll);
            maps[slot] = Some(rx);
        }
        Ok(())
    }
    /// Read `bytes` from `src` (offset 0) into a Vec, via a staging buffer.
    pub async fn read(&self, src: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>, String> {
        self.read_at(src, 0, bytes).await
    }
    pub async fn read_f32(&self, src: &wgpu::Buffer, n: usize) -> Result<Vec<f32>, String> {
        let b = self.read(src, n as u64 * 4).await?;
        Ok(bytemuck::cast_slice(&b).to_vec())
    }
    /// Read `n` floats starting at element `from`.
    pub async fn read_range_f32(&self, src: &wgpu::Buffer, from: usize, n: usize) -> Result<Vec<f32>, String> {
        let b = self.read_at(src, from as u64 * 4, n as u64 * 4).await?;
        Ok(bytemuck::cast_slice(&b).to_vec())
    }
    /// Read `bytes` from `src` at byte offset `from`. In slices, with the event
    /// loop yielded between them, for the same reason `upload` writes in slices:
    /// a mapped range's bytes are copied out by Chrome's GPU process, and a full
    /// frame's worth in one go stalls the compositor.
    async fn read_at(&self, src: &wgpu::Buffer, from: u64, bytes: u64) -> Result<Vec<u8>, String> {
        const SLICE: u64 = XFER_SLICE as u64;
        let st = self.staging(bytes.min(SLICE));
        let mut out = Vec::with_capacity(bytes as usize);
        let mut off = 0;
        while off < bytes {
            let len = (bytes - off).min(SLICE);
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(src, from + off, &st, 0, len.div_ceil(4) * 4);
            self.queue.submit([enc.finish()]);
            let (tx, rx) = oneshot::channel();
            st.slice(..len.div_ceil(4) * 4).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::Poll);
            rx.await.map_err(|_| "map callback dropped".to_string())?.map_err(|e| format!("map: {e:?}"))?;
            out.extend_from_slice(&st.slice(..len.div_ceil(4) * 4).get_mapped_range().map_err(|e| format!("{e:?}"))?[..len as usize]);
            st.unmap();
            off += len;
            if off < bytes {
                yield_now().await;
            }
        }
        Ok(out)
    }
}

/// Records dispatches into one command buffer; flushes automatically when the
/// uniform ring is full. Call `submit()` at the end.
pub struct Rec<'a> {
    gpu: &'a Gpu,
    enc: Option<wgpu::CommandEncoder>,
    n: usize,
    keep: Vec<wgpu::BindGroup>,
}

impl<'a> Rec<'a> {
    /// `bufs` = [a, b, o, e, wt, u] (see shaders.wgsl bindings 1..6).
    pub fn dispatch(&mut self, kernel: &'static str, bufs: [Option<&wgpu::Buffer>; 6], p: P, grid: (u32, u32)) {
        if self.n == SLOTS {
            self.flush();
        }
        let g = self.gpu;
        let off = self.n as u64 * SLOT;
        g.queue.write_buffer(&g.uni, off, bytemuck::bytes_of(&p));
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &g.uni,
                offset: 0,
                size: wgpu::BufferSize::new(48),
            }),
        }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.unwrap_or(&g.dummies[i]).as_entire_binding(),
            });
        }
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &g.bgl, entries: &entries });
        let enc = self.enc.as_mut().unwrap();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&g.pipes[kernel]);
            pass.set_bind_group(0, &bg, &[off as u32]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        self.keep.push(bg);
        self.n += 1;
    }
    pub fn copy(&mut self, src: &wgpu::Buffer, so: u64, dst: &wgpu::Buffer, doff: u64, bytes: u64) {
        self.enc.as_mut().unwrap().copy_buffer_to_buffer(src, so, dst, doff, bytes);
    }
    /// Zero a whole buffer.
    pub fn clear(&mut self, buf: &wgpu::Buffer) {
        self.enc.as_mut().unwrap().clear_buffer(buf, 0, None);
    }
    fn flush(&mut self) {
        let enc = self.enc.take().unwrap();
        self.gpu.queue.submit([enc.finish()]);
        self.enc = Some(self.gpu.device.create_command_encoder(&Default::default()));
        self.n = 0;
        self.keep.clear();
    }
    pub fn submit(mut self) {
        let enc = self.enc.take().unwrap();
        self.gpu.queue.submit([enc.finish()]);
    }
}
