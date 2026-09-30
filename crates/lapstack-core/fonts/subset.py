#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

"""Subset a TrueType (glyf) font to a set of characters, standard library only.

    python3 subset.py FiraSans-Regular.ttf FiraSans-Regular.subset.ttf

Keeps the glyphs of the characters below (and the components of composite
glyphs), rebuilds cmap (one format 4 subtable), glyf/loca (long offsets),
hmtx/hhea, maxp, head and a format 3 post; copies OS/2 and name (the license
notice lives there); drops hinting (cvt, fpgm, prep), GPOS/GSUB/GDEF and
everything else — lapstack's rasteriser (overlay.rs) reads outlines only.
"""
import struct
import sys

CHARS = (
    list(range(0x20, 0x7F)) + list(range(0xA0, 0x100)) +
    [0x152, 0x153, 0x160, 0x161, 0x178, 0x17D, 0x17E,
     0x2013, 0x2014, 0x2018, 0x2019, 0x201A, 0x201C, 0x201D, 0x201E, 0x2020, 0x2022, 0x2026,
     0x2030, 0x2039, 0x203A, 0x20AC, 0x2122,
     0x2190, 0x2191, 0x2192, 0x2193, 0x2212, 0x2248, 0x2260, 0x2264, 0x2265,
     0x394, 0x3A9, 0x3B1, 0x3B2, 0x3B3, 0x3B4, 0x3BB, 0x3BC, 0x3C0, 0x3C3, 0x3C9,
     0x2113])


def main(src, dst):
    data = open(src, 'rb').read()
    assert data[:4] == b'\x00\x01\x00\x00', 'not a TrueType (glyf) font'
    num_tables = struct.unpack('>H', data[4:6])[0]
    tables = {}
    for i in range(num_tables):
        off = 12 + 16 * i
        tag = data[off:off + 4].decode('latin1')
        _, toff, tlen = struct.unpack('>III', data[off + 4:off + 16])
        tables[tag] = data[toff:toff + tlen]
    head = bytearray(tables['head'])
    index_to_loc = struct.unpack('>h', head[50:52])[0]
    maxp = bytearray(tables['maxp'])
    num_glyphs = struct.unpack('>H', maxp[4:6])[0]
    loca = tables['loca']
    if index_to_loc == 0:
        offs = [2 * v for v in struct.unpack('>%dH' % (num_glyphs + 1), loca[:2 * (num_glyphs + 1)])]
    else:
        offs = list(struct.unpack('>%dI' % (num_glyphs + 1), loca[:4 * (num_glyphs + 1)]))
    glyf = tables['glyf']

    def glyph(g):
        return glyf[offs[g]:offs[g + 1]]

    # ---- cmap: every mapping the font has (format 4 and 12 subtables)
    cmap = tables['cmap']
    mapping = {}
    for i in range(struct.unpack('>H', cmap[2:4])[0]):
        _pid, _eid, off = struct.unpack('>HHI', cmap[4 + 8 * i:12 + 8 * i])
        fmt = struct.unpack('>H', cmap[off:off + 2])[0]
        if fmt == 4:
            seg_x2 = struct.unpack('>H', cmap[off + 6:off + 8])[0]
            seg = seg_x2 // 2
            ends = struct.unpack('>%dH' % seg, cmap[off + 14:off + 14 + seg_x2])
            starts = struct.unpack('>%dH' % seg, cmap[off + 16 + seg_x2:off + 16 + 2 * seg_x2])
            deltas = struct.unpack('>%dh' % seg, cmap[off + 16 + 2 * seg_x2:off + 16 + 3 * seg_x2])
            ro_base = off + 16 + 3 * seg_x2
            ros = struct.unpack('>%dH' % seg, cmap[ro_base:ro_base + seg_x2])
            for s in range(seg):
                for c in range(starts[s], min(ends[s], 0xFFFE) + 1):
                    if ros[s] == 0:
                        g = (c + deltas[s]) & 0xFFFF
                    else:
                        gi = ro_base + 2 * s + ros[s] + 2 * (c - starts[s])
                        g = struct.unpack('>H', cmap[gi:gi + 2])[0]
                        if g:
                            g = (g + deltas[s]) & 0xFFFF
                    if g:
                        mapping.setdefault(c, g)
        elif fmt == 12:
            ngroups = struct.unpack('>I', cmap[off + 12:off + 16])[0]
            for k in range(ngroups):
                s, e, g = struct.unpack('>III', cmap[off + 16 + 12 * k:off + 28 + 12 * k])
                for c in range(s, e + 1):
                    mapping.setdefault(c, g + c - s)

    # ---- the glyphs to keep: the characters', and the components of composites
    def components(g):
        d = glyph(g)
        if len(d) < 10 or struct.unpack('>h', d[:2])[0] >= 0:
            return []
        out, p = [], 10
        while True:
            flags, gi = struct.unpack('>HH', d[p:p + 4])
            out.append((gi, p + 2))   # the component's glyph index and where it sits, to rewrite
            p += 4
            p += 4 if flags & 1 else 2
            if flags & 8:
                p += 2
            elif flags & 0x40:
                p += 4
            elif flags & 0x80:
                p += 8
            if not flags & 0x20:
                break
        return out

    chars = {c: mapping[c] for c in CHARS if c in mapping}
    missing = [hex(c) for c in CHARS if c not in mapping]
    keep = {0} | set(chars.values())
    todo = list(keep)
    while todo:
        g = todo.pop()
        for gi, _ in components(g):
            if gi not in keep:
                keep.add(gi)
                todo.append(gi)
    old_ids = sorted(keep)
    new_id = {g: i for i, g in enumerate(old_ids)}
    n = len(old_ids)

    # ---- glyf + loca (long), component indices rewritten
    new_glyf, new_loca = bytearray(), []
    for g in old_ids:
        d = bytearray(glyph(g))
        for gi, at in components(g):
            d[at:at + 2] = struct.pack('>H', new_id[gi])
        new_loca.append(len(new_glyf))
        new_glyf += d
        while len(new_glyf) % 4:
            new_glyf += b'\0'
    new_loca.append(len(new_glyf))
    loca_bytes = struct.pack('>%dI' % (n + 1), *new_loca)

    # ---- hmtx / hhea: full metrics for every kept glyph
    hhea = bytearray(tables['hhea'])
    num_h = struct.unpack('>H', hhea[34:36])[0]
    hmtx = tables['hmtx']

    def metrics(g):
        if g < num_h:
            return struct.unpack('>Hh', hmtx[4 * g:4 * g + 4])
        adv = struct.unpack('>H', hmtx[4 * (num_h - 1):4 * (num_h - 1) + 2])[0]
        p = 4 * num_h + 2 * (g - num_h)
        return adv, struct.unpack('>h', hmtx[p:p + 2])[0]

    new_hmtx = b''.join(struct.pack('>Hh', *metrics(g)) for g in old_ids)
    hhea[34:36] = struct.pack('>H', n)

    # ---- cmap: one format 4 subtable (3, 1)
    items = sorted((c, new_id[g]) for c, g in chars.items())
    segs = []   # [start, end, delta]
    for c, g in items:
        if segs and c == segs[-1][1] + 1 and (g - c) & 0xFFFF == segs[-1][2]:
            segs[-1][1] = c
        else:
            segs.append([c, c, (g - c) & 0xFFFF])
    segs.append([0xFFFF, 0xFFFF, 1])
    seg = len(segs)
    es = 0
    while (2 << es) <= seg:
        es += 1
    es -= 1
    search_range = 2 << es
    sub = struct.pack('>HHHHHHH', 4, 16 + 8 * seg, 0, 2 * seg, search_range, es, 2 * seg - search_range)
    sub += struct.pack('>%dH' % seg, *[s[1] for s in segs]) + b'\0\0'
    sub += struct.pack('>%dH' % seg, *[s[0] for s in segs])
    sub += struct.pack('>%dH' % seg, *[s[2] for s in segs])
    sub += struct.pack('>%dH' % seg, *([0] * seg))
    new_cmap = struct.pack('>HHHHI', 0, 1, 3, 1, 12) + sub

    # ---- head, maxp, post
    head[8:12] = b'\0\0\0\0'
    head[50:52] = struct.pack('>h', 1)
    maxp[4:6] = struct.pack('>H', n)
    post = bytearray(tables['post'][:32])
    post[0:4] = struct.pack('>I', 0x00030000)

    out_tables = {
        'OS/2': tables['OS/2'], 'cmap': new_cmap, 'glyf': bytes(new_glyf), 'head': bytes(head), 'hhea': bytes(hhea),
        'hmtx': new_hmtx, 'loca': loca_bytes, 'maxp': bytes(maxp), 'name': tables['name'], 'post': bytes(post),
    }

    def checksum(b):
        b = b + b'\0' * (-len(b) % 4)
        return sum(struct.unpack('>%dI' % (len(b) // 4), b)) & 0xFFFFFFFF

    tags = sorted(out_tables)
    nt = len(tags)
    es = 0
    while (2 << es) <= nt:
        es += 1
    es -= 1
    sr = (2 << es) * 16
    font = bytearray(struct.pack('>IHHHH', 0x00010000, nt, sr, es, nt * 16 - sr))
    off = 12 + 16 * nt
    body = bytearray()
    for t in tags:
        b = out_tables[t]
        font += t.encode('latin1') + struct.pack('>III', checksum(b), off + len(body), len(b))
        body += b + b'\0' * (-len(b) % 4)
    font += body
    adj = (0xB1B0AFBA - checksum(bytes(font))) & 0xFFFFFFFF
    head_off = struct.unpack('>I', font[12 + 16 * tags.index('head') + 8:12 + 16 * tags.index('head') + 12])[0]
    font[head_off + 8:head_off + 12] = struct.pack('>I', adj)
    open(dst, 'wb').write(font)
    print(f'{dst}: {n} glyphs for {len(chars)} characters, {len(font)} bytes (from {len(data)})' + (f'; not in the font: {" ".join(missing)}' if missing else ''))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
