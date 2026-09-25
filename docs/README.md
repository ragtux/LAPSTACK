# The papers lapstack is written from

lapstack is a from-scratch implementation: the algorithm comes from these
papers, not from anyone's source. They are cited here rather than redistributed
— they are their authors' and publishers' work, under their own copyright, and
this repository has no licence to hand them on.

## The pyramid

**P. J. Burt and E. H. Adelson**, *The Laplacian Pyramid as a Compact Image
Code*. IEEE Transactions on Communications **31**(4), 532–540, April 1983.
[doi:10.1109/TCOM.1983.1095851](https://doi.org/10.1109/TCOM.1983.1095851)

REDUCE and EXPAND, the generating kernel and its `a` parameter, and the
band-pass decomposition itself — the basis of `crates/lapstack-core/src/pyramid.rs`.

## The multifocus composite

**E. H. Adelson, C. H. Anderson, J. R. Bergen, P. J. Burt and J. M. Ogden**,
*Pyramid methods in image processing*. RCA Engineer **29**(6), 33–41, 1984.
[Semantic Scholar](https://www.semanticscholar.org/paper/e49793511ba203e26b99e7e81fd15a7d505b5cea)

The "multifocus composite" lapstack's fusion is built on: pick, node by node,
the pyramid coefficient with the larger magnitude, then expand-and-add. The
blending between frames happens in the reconstruction itself, which is why
there is no seam to hide.

## The selection rule

**Wencheng Wang and Faliang Chang**, *A Multi-focus Image Fusion Method Based
on Laplacian Pyramid*. Journal of Computers **6**(12), 2559–2566, 2011.
[Semantic Scholar](https://www.semanticscholar.org/paper/907927b96fa87283efbc5f9a9a4202a7f8e879ff)

The 5×5 binomial generating kernel, **maximum region energy** selection for the
band-pass levels, and the local deviation + entropy rule for the residual —
`crates/lapstack-core/src/fuse.rs`.

---

The README's **Transform** and **Fusion** sections state every formula lapstack
actually implements, with the deviations from the papers called out, so the code
can be read and checked without the papers to hand.
