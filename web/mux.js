// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

// mux.js — the two containers of the Save step's video export, written here rather than
// pulled in: the encoder is the browser's (WebCodecs), and what it hands back — H.264 samples
// in AVCC form with their avcC record, or VP9 / VP8 frames — only needs a box structure
// around it.
//   muxMp4:  a plain MP4 (ftyp, mdat, moov), one video track in one chunk, with the sample
//            table: fixed durations, the keyframes, and composition offsets if the encoder
//            reordered frames (none of the browsers' encoders do; the table is there in case).
//   muxWebm: a WebM (EBML): header, Info, Tracks, then clusters of SimpleBlocks, a new
//            cluster every 30 s so the block timecodes fit their 16 bits.
// Both take {w, h, fps, samples: [{data: Uint8Array, key, cts (µs)}]} and return a Blob.

const enc = new TextEncoder();
const u8 = (...b) => Uint8Array.from(b);
const u16 = (v) => u8((v >> 8) & 255, v & 255);
const u32 = (v) => u8((v >>> 24) & 255, (v >>> 16) & 255, (v >>> 8) & 255, v & 255);
const u64 = (v) => { const hi = Math.floor(v / 4294967296), lo = v - hi * 4294967296; return new Uint8Array([...u32(hi), ...u32(lo)]); };
const u32s = (arr) => { const out = new Uint8Array(arr.length * 4), dv = new DataView(out.buffer); arr.forEach((v, i) => dv.setUint32(i * 4, v)); return out; };
const str = (s) => enc.encode(s);
const len = (parts) => parts.reduce((n, p) => n + p.length, 0);

// ---- MP4 ----
// a box as a flat list of byte arrays (the parts may nest lists); `full` adds version + flags
const box = (type, ...parts) => { const flat = parts.flat(Infinity); return [u32(8 + len(flat)), str(type), ...flat]; };
const full = (type, version, flags, ...parts) => box(type, u8(version, (flags >> 16) & 255, (flags >> 8) & 255, flags & 255), ...parts);

export function muxMp4({ w, h, fps, samples, description }) {
  const ts = 90000, delta = Math.round(ts / fps), n = samples.length, dur = n * delta;
  const ftyp = box('ftyp', str('isom'), u32(0x200), str('isom'), str('iso2'), str('avc1'), str('mp41'));
  const dataLen = samples.reduce((a, s) => a + s.data.length, 0);
  const big = dataLen + 8 > 0xFFFFFFFF;
  const mdatHead = big ? [u32(1), str('mdat'), u64(16 + dataLen)] : [u32(8 + dataLen), str('mdat')];
  const dataStart = len(ftyp) + (big ? 16 : 8);
  // decode order is the order given; the presentation times come from the encoder
  const cts = samples.map((s) => Math.round((s.cts * ts) / 1e6));
  let off = cts.map((c, i) => c - i * delta);
  const minOff = off.reduce((a, b) => Math.min(a, b), 0);
  if (minOff < 0) off = off.map((o) => o - minOff);
  const matrix = u32s([0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000]);
  const mvhd = full('mvhd', 0, 0, u32(0), u32(0), u32(ts), u32(dur), u32(0x10000), u16(0x100), u16(0), u32(0), u32(0), matrix, u32s([0, 0, 0, 0, 0, 0]), u32(2));
  const tkhd = full('tkhd', 0, 3, u32(0), u32(0), u32(1), u32(0), u32(dur), u32(0), u32(0), u16(0), u16(0), u16(0), u16(0), matrix, u32(w << 16), u32(h << 16));
  const mdhd = full('mdhd', 0, 0, u32(0), u32(0), u32(ts), u32(dur), u16(0x55c4), u16(0));
  const hdlr = full('hdlr', 0, 0, u32(0), str('vide'), u32(0), u32(0), u32(0), str('VideoHandler\0'));
  const vmhd = full('vmhd', 0, 1, u16(0), u16(0), u16(0), u16(0));
  const dinf = box('dinf', full('dref', 0, 0, u32(1), full('url ', 0, 1)));
  const avc1 = box('avc1', new Uint8Array(6), u16(1), u16(0), u16(0), u32s([0, 0, 0]), u16(w), u16(h), u32(0x480000), u32(0x480000), u32(0), u16(1), new Uint8Array(32), u16(0x18), u16(0xFFFF),
    box('avcC', description), box('pasp', u32(1), u32(1)));
  const keys = samples.map((s, i) => (s.key ? i + 1 : 0)).filter(Boolean);
  const stbl = box('stbl',
    full('stsd', 0, 0, u32(1), avc1),
    full('stts', 0, 0, u32(1), u32(n), u32(delta)),
    full('stss', 0, 0, u32(keys.length), u32s(keys)),
    off.some((o) => o !== 0) ? full('ctts', 0, 0, u32(n), u32s(off.flatMap((o) => [1, o]))) : [],
    full('stsc', 0, 0, u32(1), u32(1), u32(n), u32(1)),
    full('stsz', 0, 0, u32(0), u32(n), u32s(samples.map((s) => s.data.length))),
    dataStart > 0xFFFFFFFF ? full('co64', 0, 0, u32(1), u64(dataStart)) : full('stco', 0, 0, u32(1), u32(dataStart)));
  const moov = box('moov', mvhd, box('trak', tkhd, box('mdia', mdhd, hdlr, box('minf', vmhd, dinf, stbl))));
  return new Blob([...ftyp, ...mdatHead, ...samples.map((s) => s.data), ...moov], { type: 'video/mp4' });
}

// ---- WebM ----
// an EBML element: the id as written, the size as a variable-length integer, the payload
const vint = (n) => {
  let l = 1; while (n >= 2 ** (7 * l) - 1 && l < 8) l++;
  const out = new Uint8Array(l); let v = n;
  for (let i = l - 1; i >= 0; i--) { out[i] = v % 256; v = Math.floor(v / 256); }
  out[0] |= 0x80 >> (l - 1); return out;
};
const idBytes = (x) => { const b = []; let v = x; while (v > 0) { b.unshift(v % 256); v = Math.floor(v / 256); } return Uint8Array.from(b); };
const el = (id, ...parts) => { const flat = parts.flat(Infinity); return [idBytes(id), vint(len(flat)), ...flat]; };
const uint = (id, v) => { const b = []; let x = v; do { b.unshift(x % 256); x = Math.floor(x / 256); } while (x > 0); return el(id, Uint8Array.from(b)); };
const float = (id, v) => { const b = new Uint8Array(8); new DataView(b.buffer).setFloat64(0, v); return el(id, b); };
const string = (id, s) => el(id, str(s));

export function muxWebm({ w, h, fps, samples, codec = 'V_VP9' }) {
  const header = el(0x1A45DFA3, uint(0x4286, 1), uint(0x42F7, 1), uint(0x42F2, 4), uint(0x42F3, 8), string(0x4282, 'webm'), uint(0x4287, 4), uint(0x4285, 2));
  const info = el(0x1549A966, uint(0x2AD7B1, 1000000), string(0x4D80, 'lapstack'), string(0x5741, 'lapstack'), float(0x4489, (samples.length * 1000) / fps));
  const tracks = el(0x1654AE6B, el(0xAE, uint(0xD7, 1), uint(0x73C5, 1), uint(0x83, 1), string(0x86, codec), uint(0x23E383, Math.round(1e9 / fps)), el(0xE0, uint(0xB0, w), uint(0xBA, h))));   // DefaultDuration: the frame time in ns, so players know the rate
  const clusters = []; let base = -1, blocks = [];
  const flush = () => { if (blocks.length) clusters.push(el(0x1F43B675, uint(0xE7, base), ...blocks)); blocks = []; };
  for (const s of samples) {
    const t = Math.round(s.cts / 1000);
    if (base < 0 || t - base > 30000) { flush(); base = t; }
    const rel = t - base;
    blocks.push(el(0xA3, u8(0x81, (rel >> 8) & 255, rel & 255, s.key ? 0x80 : 0), s.data));
  }
  flush();
  return new Blob([...header, ...el(0x18538067, info, tracks, ...clusters)], { type: 'video/webm' });
}
