//! View layer for the buffer protocol — the pyre analogue of
//! `pypy/interpreter/buffer.py`'s `BufferView`.  A `BufferView` carries the
//! geometry (offset / shape / strides / format / itemsize) over a byte-level
//! [`Buffer`] and gathers the live logical bytes in C order, honouring a
//! strided or N-D layout, without detaching a copy of the backing.
//!
//! `memoryview`'s `W_MemoryView` holds one of these off the GC heap; the GC
//! reaches the refs inside (the backing exporter, the `.obj` exporter, and
//! any stored format / shape / strides objects) through `W_MemoryView`'s
//! custom trace.  The specialised variants (`Simple` / `Raw` / `Slice`)
//! derive their geometry natively and store only exporter refs plus scalars;
//! callers that need a Python `str` / `tuple` wrap a fresh one at the `descr`
//! boundary.

use crate::buffer::Buffer;
use crate::pyobject::PyObjectRef;

/// `_copy_base` — push one `isz`-wide element at byte offset `base`, dropping
/// it when the address falls outside the backing (a reversed / strided slice
/// past the end), so the gather never panics.
fn copy_base(full: &[u8], base: i64, isz: usize, out: &mut Vec<u8>) {
    if isz > 0 && base >= 0 && base as usize + isz <= full.len() {
        let b = base as usize;
        out.extend_from_slice(&full[b..b + isz]);
    }
}

/// `_copy_rec` — recursive C-order copy of dimension `idim`.  The innermost
/// dimension walks `shape[ndim-1]` elements by `strides[ndim-1]`; an outer
/// dimension recurses `shape[idim]` times, advancing `off` by `strides[idim]`.
fn copy_rec(
    full: &[u8],
    shape: &[i64],
    strides: &[i64],
    ndim: i64,
    idim: i64,
    mut off: i64,
    isz: usize,
    out: &mut Vec<u8>,
) {
    let dimshape = shape.get(idim as usize).copied().unwrap_or(0);
    let dimstride = strides.get(idim as usize).copied().unwrap_or(0);
    if idim == ndim - 1 {
        if dimstride == 0 {
            return;
        }
        for _ in 0..dimshape {
            copy_base(full, off, isz, out);
            off += dimstride;
        }
    } else {
        for _ in 0..dimshape {
            copy_rec(full, shape, strides, ndim, idim + 1, off, isz, out);
            off += dimstride;
        }
    }
}

/// Read a `tuple[int]` (shape or strides) into a native vector.
///
/// # Safety
/// `t` must point to a valid tuple of ints.
unsafe fn read_dims(t: PyObjectRef) -> Vec<i64> {
    unsafe {
        let n = crate::tupleobject::w_tuple_len(t);
        (0..n)
            .map(|i| {
                crate::tupleobject::w_tuple_getitem(t, i as i64)
                    .map(|w| crate::intobject::w_int_get_value(w))
                    .unwrap_or(0)
            })
            .collect()
    }
}

/// A view of a [`Buffer`]'s bytes with offset / shape / stride geometry and a
/// buffer-protocol format.
///
/// One variant per class of PyPy's view hierarchy — `SimpleView` /
/// `RawBufferView` (plain 1-D), `BufferSlice` (strided slice),
/// `BufferView1D` / `BufferViewND` (cast), `ReadonlyWrapper` — each carrying
/// only the state its class stores and deriving the rest.
///
/// Views are immutable once built (`_immutable_ = True`), so `Clone` shares
/// semantics: PyPy hands the same view object to a wrapper / copy, pyre
/// clones it into the new owner's box.
#[derive(Clone)]
pub enum BufferView {
    /// `SimpleView` (`pypy/interpreter/buffer.py:270`) — a plain contiguous
    /// 1-D byte view (`bytes` / `bytearray`).  Format `'B'`, itemsize 1,
    /// ndim 1, offset 0, shape `[length]`, strides `[1]` are all derived;
    /// `readonly` comes from the backing, so nothing but the exporter refs is
    /// stored.
    Simple {
        backing: Buffer,
        w_obj: PyObjectRef,
        length: i64,
    },
    /// `RawBufferView` (`buffer.py:231`) — a typed contiguous 1-D view
    /// (`array.array`).  Format / itemsize are explicit; ndim 1, offset 0,
    /// shape `[length / itemsize]` (`[0]` when empty), strides `[itemsize]`
    /// derive; `readonly` comes from the backing.
    Raw {
        backing: Buffer,
        w_obj: PyObjectRef,
        w_fmt: PyObjectRef,
        itemsize: i64,
        length: i64,
    },
    /// `BufferSlice` (`buffer.py:321`) — a strided dimension-0 window over a
    /// parent view, produced by a step≠1 slice (a step==1 slice of a
    /// `Simple` / `Raw` view re-specialises over a [`Buffer::Sub`] window
    /// instead).  `start` / `step` are in parent dimension-0 element units
    /// and `length` is the element count (`shape[0]`); shape / strides /
    /// format / itemsize derive from the parent with dimension 0 replaced,
    /// so no geometry objects are stored.  A slice of a slice composes into
    /// the parent's coordinates, keeping the wrapper depth at 1.
    Slice {
        parent: Box<BufferView>,
        w_obj: PyObjectRef,
        start: i64,
        step: i64,
        length: i64,
    },
    /// `BufferView1D` (`memoryobject.py:867`) — a cast to a new 1-D element
    /// format over the parent's bytes (`_cast_to_1D`).  Format / itemsize are
    /// explicit; ndim 1, shape `[parent_length / itemsize]`, strides
    /// `[itemsize]` derive; byte access delegates to the parent
    /// (`IndirectView`).
    View1D {
        parent: Box<BufferView>,
        w_obj: PyObjectRef,
        w_fmt: PyObjectRef,
        itemsize: i64,
    },
    /// `BufferViewND` (`memoryobject.py:893`) — a cast to an N-dimensional
    /// shape over a 1-D parent (`_cast_to_ND`).  `shape` / `strides` ride as
    /// their tuple objects; format / itemsize come from the parent.
    ViewND {
        parent: Box<BufferView>,
        w_obj: PyObjectRef,
        ndim: i64,
        w_shape: PyObjectRef,
        w_strides: PyObjectRef,
    },
    /// `ReadonlyWrapper` (`buffer.py:415`) — `toreadonly`'s wrapper: every
    /// read delegates to the wrapped view, `readonly` is forced true.
    Readonly {
        view: Box<BufferView>,
        w_obj: PyObjectRef,
    },
}

impl BufferView {
    /// The backing byte storage (the root exporter's [`Buffer`]).
    #[inline]
    pub fn backing(&self) -> &Buffer {
        match self {
            BufferView::Simple { backing, .. } | BufferView::Raw { backing, .. } => backing,
            BufferView::Slice { parent, .. }
            | BufferView::View1D { parent, .. }
            | BufferView::ViewND { parent, .. } => parent.backing(),
            BufferView::Readonly { view, .. } => view.backing(),
        }
    }
    /// The exporter reported by `memoryview.obj`.
    #[inline]
    pub fn w_obj(&self) -> PyObjectRef {
        match self {
            BufferView::Simple { w_obj, .. }
            | BufferView::Raw { w_obj, .. }
            | BufferView::Slice { w_obj, .. }
            | BufferView::View1D { w_obj, .. }
            | BufferView::ViewND { w_obj, .. }
            | BufferView::Readonly { w_obj, .. } => *w_obj,
        }
    }
    /// The element format string (`getformat`), read natively — the callers
    /// that need a Python `str` wrap a fresh one at the `descr` boundary.  A
    /// `Simple` view derives `'B'`; a `Raw` view reads its explicit format.
    ///
    /// # Safety
    /// The view's stored format object must be a live `str`.
    #[inline]
    pub unsafe fn format_str(&self) -> &'static str {
        unsafe {
            match self {
                BufferView::Simple { .. } => "B",
                BufferView::Raw { w_fmt, .. } | BufferView::View1D { w_fmt, .. } => {
                    crate::w_str_get_value(*w_fmt)
                }
                BufferView::Slice { parent, .. } | BufferView::ViewND { parent, .. } => {
                    parent.format_str()
                }
                BufferView::Readonly { view, .. } => view.format_str(),
            }
        }
    }
    /// The per-dimension element counts (`getshape`) as native extents.  A
    /// `Simple` view is `[length]`; a `Raw` view is `[length / itemsize]`
    /// (`[0]` when empty, `buffer.py:254`).
    ///
    /// # Safety
    /// The view's stored shape object must be a live tuple of ints.
    #[inline]
    pub unsafe fn native_shape(&self) -> Vec<i64> {
        unsafe {
            match self {
                BufferView::Simple { length, .. } => vec![*length],
                BufferView::Raw {
                    itemsize, length, ..
                } => {
                    if *length == 0 {
                        vec![0]
                    } else {
                        vec![*length / *itemsize]
                    }
                }
                // Dimension 0 takes the slice's element count; later
                // dimensions ride along (`shape[0] = length`, buffer.py:334).
                BufferView::Slice { parent, length, .. } => {
                    let mut shape = parent.native_shape();
                    if let Some(s0) = shape.first_mut() {
                        *s0 = *length;
                    }
                    shape
                }
                // `[getlength() // itemsize]` (memoryobject.py:888).
                BufferView::View1D {
                    parent, itemsize, ..
                } => vec![parent.length() / *itemsize],
                BufferView::ViewND { w_shape, .. } => read_dims(*w_shape),
                BufferView::Readonly { view, .. } => view.native_shape(),
            }
        }
    }
    /// The per-dimension byte steps (`getstrides`) as native extents.  A
    /// `Simple` view is `[1]`; a `Raw` view is `[itemsize]`.
    ///
    /// # Safety
    /// The view's stored strides object must be a live tuple of ints.
    #[inline]
    pub unsafe fn native_strides(&self) -> Vec<i64> {
        unsafe {
            match self {
                BufferView::Simple { .. } => vec![1],
                BufferView::Raw { itemsize, .. } | BufferView::View1D { itemsize, .. } => {
                    vec![*itemsize]
                }
                // Dimension 0 steps by the parent's stride times the slice
                // step (`strides[0] *= step`, buffer.py:332).
                BufferView::Slice { parent, step, .. } => {
                    let mut strides = parent.native_strides();
                    if let Some(s0) = strides.first_mut() {
                        *s0 *= *step;
                    }
                    strides
                }
                BufferView::ViewND { w_strides, .. } => read_dims(*w_strides),
                BufferView::Readonly { view, .. } => view.native_strides(),
            }
        }
    }
    /// `strides[0]` — the signed byte step between consecutive elements of a
    /// 1-D view, falling back to `itemsize` when the strides are empty.
    ///
    /// # Safety
    /// The view's stored strides object must be a live tuple of ints.
    #[inline]
    pub unsafe fn stride0(&self) -> i64 {
        unsafe {
            match self {
                BufferView::Simple { .. } => 1,
                BufferView::Raw { itemsize, .. } | BufferView::View1D { itemsize, .. } => *itemsize,
                BufferView::Slice { parent, step, .. } => parent.stride0() * *step,
                BufferView::ViewND {
                    parent, w_strides, ..
                } => crate::tupleobject::w_tuple_getitem(*w_strides, 0)
                    .map(|s| crate::intobject::w_int_get_value(s))
                    .unwrap_or_else(|| parent.itemsize()),
                BufferView::Readonly { view, .. } => view.stride0(),
            }
        }
    }
    #[inline]
    pub fn itemsize(&self) -> i64 {
        match self {
            BufferView::Raw { itemsize, .. } | BufferView::View1D { itemsize, .. } => *itemsize,
            BufferView::Simple { .. } => 1,
            BufferView::Slice { parent, .. } | BufferView::ViewND { parent, .. } => {
                parent.itemsize()
            }
            BufferView::Readonly { view, .. } => view.itemsize(),
        }
    }
    #[inline]
    pub fn ndim(&self) -> i64 {
        match self {
            BufferView::Simple { .. } | BufferView::Raw { .. } | BufferView::View1D { .. } => 1,
            BufferView::Slice { parent, .. } => parent.ndim(),
            BufferView::ViewND { ndim, .. } => *ndim,
            BufferView::Readonly { view, .. } => view.ndim(),
        }
    }
    /// Byte offset of the view's first element within the backing storage.
    /// A `Slice` starts `start` parent elements past the parent's own offset
    /// (`getbytes` adds `self.start * self.parent.getstrides()[0]`,
    /// buffer.py:340).
    ///
    /// # Safety
    /// A `Slice` parent's stored strides object must be live.
    #[inline]
    pub unsafe fn offset(&self) -> i64 {
        unsafe {
            match self {
                BufferView::Simple { .. } | BufferView::Raw { .. } => 0,
                BufferView::Slice { parent, start, .. } => {
                    parent.offset() + *start * parent.stride0()
                }
                // IndirectView delegates byte access to the parent unshifted.
                BufferView::View1D { parent, .. } | BufferView::ViewND { parent, .. } => {
                    parent.offset()
                }
                BufferView::Readonly { view, .. } => view.offset(),
            }
        }
    }
    /// `getlength` — the view's byte count.  A `Slice` spans `shape[0]`
    /// elements (`buffer.py:336`); a `ViewND` spans `product(shape)` elements
    /// (`memoryobject.py:922`).
    ///
    /// # Safety
    /// A `ViewND`'s stored shape object must be a live tuple of ints.
    #[inline]
    pub unsafe fn length(&self) -> i64 {
        unsafe {
            match self {
                BufferView::Simple { length, .. } | BufferView::Raw { length, .. } => *length,
                BufferView::Slice { parent, length, .. } => *length * parent.itemsize(),
                BufferView::View1D { parent, .. } => parent.length(),
                BufferView::ViewND {
                    parent, w_shape, ..
                } => read_dims(*w_shape).iter().product::<i64>() * parent.itemsize(),
                BufferView::Readonly { view, .. } => view.length(),
            }
        }
    }
    #[inline]
    pub fn readonly(&self) -> bool {
        match self {
            BufferView::Simple { backing, .. } | BufferView::Raw { backing, .. } => {
                backing.readonly()
            }
            BufferView::Slice { parent, .. }
            | BufferView::View1D { parent, .. }
            | BufferView::ViewND { parent, .. } => parent.readonly(),
            BufferView::Readonly { .. } => true,
        }
    }

    /// `new_slice(start, step, slicelength)` — a live dimension-0 sub-view
    /// sharing the parent's storage (`buffer.py:149,261,310,403`).  `start` /
    /// `slicelength` are in dimension-0 element units.
    ///
    /// A step==1 slice of a `Simple` / `Raw` view re-specialises over a
    /// [`Buffer::Sub`] window of the same storage, so the result stays a
    /// plain contiguous view; every other case wraps the parent in a
    /// [`Slice`](BufferView::Slice).  Slicing a `Slice` composes into its
    /// parent's coordinates — `start` maps through `parent_index`
    /// (`self.start + self.step * idx`, buffer.py:386) so a re-slice of a
    /// strided slice lands on the elements the composed stride selects.
    ///
    /// # Safety
    /// The view's stored geometry objects must be live.
    pub unsafe fn new_slice(&self, start: i64, step: i64, slicelength: i64) -> BufferView {
        let wrap = |view: &BufferView| BufferView::Slice {
            parent: Box::new(view.clone()),
            w_obj: view.w_obj(),
            start,
            step,
            length: slicelength,
        };
        match self {
            BufferView::Simple { backing, w_obj, .. } if step == 1 => BufferView::Simple {
                backing: Buffer::sub(backing.clone(), start as usize, slicelength),
                w_obj: *w_obj,
                length: slicelength,
            },
            BufferView::Raw {
                backing,
                w_obj,
                w_fmt,
                itemsize,
                ..
            } if step == 1 => {
                let n = *itemsize;
                BufferView::Raw {
                    backing: Buffer::sub(backing.clone(), (start * n) as usize, slicelength * n),
                    w_obj: *w_obj,
                    w_fmt: *w_fmt,
                    itemsize: n,
                    length: slicelength * n,
                }
            }
            BufferView::Slice {
                parent,
                w_obj,
                start: pstart,
                step: pstep,
                ..
            } => BufferView::Slice {
                parent: parent.clone(),
                w_obj: *w_obj,
                start: pstart + pstep * start,
                step: pstep * step,
                length: slicelength,
            },
            // A slice of a read-only wrapper stays wrapped (buffer.py:462).
            BufferView::Readonly { w_obj, .. } => BufferView::Readonly {
                view: Box::new(wrap(self)),
                w_obj: *w_obj,
            },
            other => wrap(other),
        }
    }

    /// The LIVE logical bytes of the view (`buffer.py as_str`), read from the
    /// backing object's own storage — no detached copy — so the view observes
    /// later mutation of a bytearray / array source.  Honours offset / shape /
    /// strides so a strided slice (`m[::2]`, `m[::-1]`) or an N-D view gathers
    /// the right elements in C order.
    ///
    /// # Safety
    /// The backing [`Buffer`]'s `w_obj` must point to a live object of its
    /// tagged kind.
    pub unsafe fn gather(&self) -> Vec<u8> {
        unsafe {
            let itemsize = self.itemsize();
            let ndim = self.ndim();
            let offset = self.offset();
            let full = self.backing().as_bytes();
            let isz = itemsize.max(0) as usize;
            if ndim == 0 {
                let mut out = Vec::with_capacity(isz);
                copy_base(full, offset, isz, &mut out);
                return out;
            }
            let shape = self.native_shape();
            let strides = self.native_strides();
            let count = if itemsize > 0 {
                self.length() / itemsize
            } else {
                0
            };
            let mut out = Vec::with_capacity(count.max(0) as usize * isz);
            copy_rec(full, &shape, &strides, ndim, 0, offset, isz, &mut out);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Geometry-only tests: `Simple` / `Raw` derive everything from scalars,
    // so a fake exporter address is never dereferenced.
    fn fake(addr: usize) -> PyObjectRef {
        addr as PyObjectRef
    }

    fn simple(len: i64) -> BufferView {
        BufferView::Simple {
            backing: Buffer::String {
                w_obj: fake(0x1000),
            },
            w_obj: fake(0x1000),
            length: len,
        }
    }

    #[test]
    fn simple_step1_slice_respecializes_over_sub_window() {
        // SimpleView.new_slice step==1 (buffer.py:312): the result is another
        // SimpleView over a SubBuffer window, not a BufferSlice wrapper.
        let s = unsafe { simple(10).new_slice(2, 1, 5) };
        match &s {
            BufferView::Simple {
                backing: Buffer::Sub { offset, size, .. },
                length,
                ..
            } => {
                assert_eq!((*offset, *size, *length), (2, 5, 5));
            }
            _ => panic!("expected Simple over Sub"),
        }
        unsafe {
            assert_eq!(s.native_shape(), vec![5]);
            assert_eq!(s.native_strides(), vec![1]);
            assert_eq!(s.offset(), 0);
        }
    }

    #[test]
    fn raw_step1_slice_scales_the_window_by_itemsize() {
        // RawBufferView.new_slice step==1 (buffer.py:262-265): the SubBuffer
        // window is `start * itemsize .. + slicelength * itemsize`.
        let raw = BufferView::Raw {
            backing: Buffer::Array {
                w_obj: fake(0x2000),
            },
            w_obj: fake(0x2000),
            w_fmt: fake(0x2004),
            itemsize: 4,
            length: 40,
        };
        let s = unsafe { raw.new_slice(1, 1, 3) };
        match &s {
            BufferView::Raw {
                backing: Buffer::Sub { offset, size, .. },
                itemsize,
                length,
                ..
            } => {
                assert_eq!((*offset, *size, *itemsize, *length), (4, 12, 4, 12));
            }
            _ => panic!("expected Raw over Sub"),
        }
        unsafe {
            assert_eq!(s.native_shape(), vec![3]);
            assert_eq!(s.native_strides(), vec![4]);
        }
    }

    #[test]
    fn strided_step_wraps_in_a_slice_with_derived_geometry() {
        let s = unsafe { simple(10).new_slice(0, 2, 5) };
        assert!(matches!(s, BufferView::Slice { .. }));
        unsafe {
            assert_eq!(s.native_shape(), vec![5]);
            assert_eq!(s.native_strides(), vec![2]);
            assert_eq!(s.offset(), 0);
        }
        assert_eq!(s.itemsize(), 1);
        assert_eq!(s.ndim(), 1);
        assert_eq!(unsafe { s.length() }, 5);
    }

    #[test]
    fn slice_of_slice_composes_through_the_parent_step() {
        // Re-slicing a strided slice maps `start` through `parent_index`
        // (buffer.py:386) and multiplies the steps, collapsing to one
        // wrapper over the original parent.
        let s2 = unsafe { simple(10).new_slice(0, 2, 5).new_slice(1, 1, 2) };
        match &s2 {
            BufferView::Slice {
                parent,
                start,
                step,
                length,
                ..
            } => {
                assert_eq!((*start, *step, *length), (2, 2, 2));
                assert!(matches!(**parent, BufferView::Simple { .. }));
            }
            _ => panic!("expected collapsed Slice"),
        }
        unsafe {
            assert_eq!(s2.offset(), 2);
            assert_eq!(s2.native_strides(), vec![2]);
        }
    }

    #[test]
    fn reversed_slice_starts_at_the_high_end() {
        let s = unsafe { simple(10).new_slice(9, -1, 10) };
        unsafe {
            assert_eq!(s.offset(), 9);
            assert_eq!(s.native_strides(), vec![-1]);
            assert_eq!(s.native_shape(), vec![10]);
        }
        assert!(s.readonly()); // String backing stays readonly through the wrapper
    }
}
