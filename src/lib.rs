//! A capacity-limited structured data buffer.
//!
//! This library provides [`StructBuf`] - a capacity-limited buffer for encoding
//! and decoding structured data. The primary use case is for safely handling
//! small, variable-length message packets sent over the network or other
//! transports.
//!
//! The encoder ensures that the message size never exceeds a pre-configured
//! limit. The decoder ensures that malformed or malicious input does not cause
//! the program to panic.
//!
//! Little-endian encoding is assumed.
//!
//! ## `no_std` support
//!
//! `structbuf` is `no_std` by default.
//!
//! # Example
//!
//! ```
//! # use structbuf::{Pack, StructBuf, Unpack};
//! let mut b = StructBuf::new(4);
//! b.append().u8(1).u16(2_u16).u8(3);
//! // b.u8(4); Would panic
//!
//! let mut p = b.unpack();
//! assert_eq!(p.u8(), 1);
//! assert_eq!(p.u16(), 2);
//! assert_eq!(p.u8(), 3);
//! assert!(p.is_ok());
//!
//! assert_eq!(p.u32(), 0);
//! assert!(!p.is_ok());
//! ```

#![no_std]
#![warn(missing_debug_implementations)]
#![warn(non_ascii_idents)]
#![warn(single_use_lifetimes)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_lifetimes)]
#![warn(unused_qualifications)]
#![warn(variant_size_differences)]
#![warn(clippy::cargo)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::inline_always)]
// #![warn(clippy::restriction)]
#![warn(clippy::assertions_on_result_states)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::decimal_literal_representation)]
#![warn(clippy::default_union_representation)]
#![warn(clippy::deref_by_slicing)]
#![warn(clippy::empty_drop)]
#![warn(clippy::empty_structs_with_brackets)]
#![warn(clippy::exhaustive_enums)]
#![warn(clippy::exit)]
#![warn(clippy::fn_to_numeric_cast_any)]
#![warn(clippy::format_push_string)]
#![warn(clippy::get_unwrap)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::missing_enforced_import_renames)]
#![warn(clippy::mixed_read_write_in_expression)]
#![warn(clippy::mod_module_files)]
#![warn(clippy::mutex_atomic)]
#![warn(clippy::pattern_type_mismatch)]
#![warn(clippy::print_stdout)]
#![warn(clippy::rc_buffer)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_add)]
#![warn(clippy::string_to_string)]
#![warn(clippy::suspicious_xor_used_as_pow)]
#![warn(clippy::todo)]
#![warn(clippy::try_err)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unnecessary_safety_comment)]
#![warn(clippy::unnecessary_safety_doc)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::unneeded_field_pattern)]
#![warn(clippy::unseparated_literal_suffix)]

use core::ops::{Deref, DerefMut};
use core::{mem, ptr, slice};

use smallvec::SmallVec;

/// Inline capacity selected to keep [`StructBuf`] within one cache line in most
/// use cases. On x64, this adds two additional machine words over the minimum
/// [`StructBuf`] size.
const INLINE_CAP: usize = 32;

/// Trait for getting a packer for a byte buffer.
pub trait Pack {
    /// Returns a packer for appending values to the end of the buffer.
    #[must_use]
    fn append(&mut self) -> Packer;

    /// Returns a packer for writing values starting at index `i`. This always
    /// succeeds, even if `i` is out of bounds.
    #[must_use]
    fn at(&mut self, i: usize) -> Packer;
}

/// Trait for getting an unpacker for a byte slice.
pub trait Unpack {
    /// Returns an unpacker for reading values from the start of the buffer.
    fn unpack(&self) -> Unpacker;
}

/// Blanket implementation for all `AsRef<[u8]>` types.
impl<T: AsRef<[u8]>> Unpack for T {
    #[inline(always)]
    fn unpack(&self) -> Unpacker {
        Unpacker::new(self.as_ref())
    }
}

/// A capacity-limited buffer for encoding and decoding structured data.
///
/// The buffer starts out with a small internal capacity that does not require
/// allocation. Once the internal capacity is exhausted, it performs at most one
/// heap allocation up to the capacity limit. The relationship
/// `len() <= capacity() <= lim()` always holds.
///
/// # Panics
///
/// It panics if any write operation exceeds the capacity limit.
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct StructBuf {
    lim: usize,
    b: SmallVec<[u8; INLINE_CAP]>,
}

impl StructBuf {
    /// Creates a new capacity-limited buffer without allocating from the heap.
    /// This should be used for creating outbound messages that may not require
    /// the full capacity.
    #[inline(always)]
    pub const fn new(lim: usize) -> Self {
        Self {
            lim,
            b: SmallVec::new_const(),
        }
    }

    /// Creates an empty buffer that can never be written to. This is the same
    /// as [`StructBuf::default()`], but usable in `const` context.
    #[inline(always)]
    pub const fn none() -> Self {
        Self::new(0)
    }

    /// Creates a pre-allocated capacity-limited buffer. This buffer will never
    /// reallocate and should be used for receiving messages up to the capacity
    /// limit.
    #[inline(always)]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            lim: cap,
            b: SmallVec::with_capacity(cap),
        }
    }

    /// Returns the number of initialized bytes in the buffer (`<= capacity()`).
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.b.len()
    }

    /// Returns the number of bytes allocated by the buffer (`<= lim()`).
    #[inline(always)]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.b.capacity().min(self.lim)
    }

    /// Returns the buffer capacity limit.
    #[inline(always)]
    #[must_use]
    pub const fn lim(&self) -> usize {
        self.lim
    }

    /// Returns the number of additional bytes that can be written to the
    /// buffer.
    #[inline(always)]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.lim - self.b.len()
    }

    /// Returns whether the buffer is empty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.b.is_empty()
    }

    /// Returns whether the buffer contains the maximum number of bytes.
    #[inline(always)]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.remaining() == 0
    }

    /// Returns whether the buffer has a limit of 0 and can never be written to.
    #[inline(always)]
    #[must_use]
    pub const fn is_none(&self) -> bool {
        self.lim == 0
    }

    /// Removes all but the first `n` bytes from the buffer without affecting
    /// capacity. It has no effect if `self.len() <= n`.
    #[inline]
    pub fn truncate(&mut self, n: usize) -> &mut Self {
        if n < self.b.len() {
            // SAFETY: n is a valid length and there is nothing to drop
            unsafe { self.b.set_len(n) }
        }
        self
    }

    /// Clears the buffer, resetting its length to 0 without affecting capacity.
    #[inline(always)]
    pub fn clear(&mut self) -> &mut Self {
        self.truncate(0)
    }

    /// Returns the buffer, leaving `self` empty and unwritable.
    #[inline(always)]
    pub fn take(&mut self) -> Self {
        mem::replace(self, Self::none())
    }

    /// Sets the buffer length.
    ///
    /// # Safety
    ///
    /// Caller must ensure that the buffer contains `n` initialized bytes, which
    /// must be `<= capacity()`.
    #[inline]
    pub unsafe fn set_len(&mut self, n: usize) -> &mut Self {
        debug_assert!(n <= self.capacity());
        self.b.set_len(n);
        self
    }

    /// Sets the buffer limit. Any existing data past the limit is truncated.
    #[inline]
    pub fn set_lim(&mut self, n: usize) -> &mut Self {
        self.lim = n;
        self.truncate(n)
    }

    /// Returns whether `n` bytes can be written to the buffer at index `i`.
    #[inline]
    #[must_use]
    pub const fn can_put_at(&self, i: usize, n: usize) -> bool {
        let (sum, overflow) = i.overflowing_add(n);
        sum <= self.lim && !overflow
    }

    /// Writes slice `v` at index `i`. Any existing data at `i` is overwritten.
    /// If `len() < i`, the buffer is padded with `i - len()` zeros.
    ///
    /// # Panics
    ///
    /// Panics if `lim() < i + v.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use structbuf::StructBuf;
    /// let mut b = StructBuf::new(4);
    /// b.put_at(1, [1]);
    /// assert_eq!(b.as_ref(), &[0, 1]);
    ///
    /// b.put_at(4, []);
    /// assert_eq!(b.as_ref(), &[0, 1, 0, 0]);
    /// ```
    #[inline(always)]
    pub fn put_at<T: AsRef<[u8]>>(&mut self, i: usize, v: T) {
        assert!(self.try_put_at(i, v), "buffer limit exceeded");
    }

    /// Writes slice `v` at index `i` if `i + v.len() <= lim()`. Any existing
    /// data at `i` is overwritten. If `len() < i`, the buffer is padded with
    /// `i - len()` zeros. Returns whether `v` was written.
    #[inline]
    pub fn try_put_at<T: AsRef<[u8]>>(&mut self, i: usize, v: T) -> bool {
        let v = v.as_ref();
        let (j, overflow) = i.overflowing_add(v.len());
        let ok = !overflow && j <= self.lim;
        if ok {
            // SAFETY: i + v.len() == j <= self.lim
            unsafe { self.put_at_unchecked(i, j, v) };
        }
        ok
    }

    /// Writes slice `v` at index `i`.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `i + v.len() == j <= self.lim`.
    unsafe fn put_at_unchecked(&mut self, i: usize, j: usize, v: &[u8]) {
        if self.b.capacity() < j {
            self.b.grow(self.lim); // TODO: Limit growth for a large lim?
        }
        let pad = i.saturating_sub(self.b.len());
        let dst = self.b.as_mut_ptr();
        if pad > 0 {
            // SAFETY: `len() + pad == i <= j` and dst is valid for at least j
            // bytes.
            unsafe { dst.add(self.b.len()).write_bytes(0, pad) };
        }
        // SAFETY: `&mut self` prevents v from referencing the buffer
        unsafe { dst.add(i).copy_from_nonoverlapping(v.as_ptr(), v.len()) };
        if j > self.b.len() {
            // SAFETY: self.b contains j initialized bytes
            unsafe { self.b.set_len(j) };
        }
    }
}

impl Pack for StructBuf {
    #[inline(always)]
    fn append(&mut self) -> Packer {
        self.at(self.b.len())
    }

    #[inline(always)]
    fn at(&mut self, i: usize) -> Packer {
        Packer { i, b: self }
    }
}

impl AsRef<[u8]> for StructBuf {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        &self.b
    }
}

impl AsMut<[u8]> for StructBuf {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.b
    }
}

impl Deref for StructBuf {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.b
    }
}

impl DerefMut for StructBuf {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.b
    }
}

/// Packer of POD values into a [`StructBuf`].
///
/// The packer maintains an index (`0 <= i <= lim()`) where new values are
/// written, which is incremented after each write. Write operations panic if
/// the buffer capacity limit is exceeded.
///
/// The packer intentionally does not expose the underlying [`StructBuf`] to
/// guarantee append-only operation.
///
/// Little-endian encoding is assumed.
#[derive(Debug)]
pub struct Packer<'a> {
    i: usize,
    b: &'a mut StructBuf, // TODO: Make generic?
}

impl Packer<'_> {
    /// Returns the current index.
    #[inline(always)]
    #[must_use]
    pub const fn position(&self) -> usize {
        self.i
    }

    /// Returns the number of additional bytes that can be written.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.b.lim.saturating_sub(self.i)
    }

    /// Advances the current index by `n` bytes.
    #[inline(always)]
    pub fn skip(&mut self, n: usize) -> &mut Self {
        self.i += n;
        self
    }

    /// Writes a `bool` as a `u8` at the current index.
    #[inline(always)]
    pub fn bool<T: Into<bool>>(&mut self, v: T) -> &mut Self {
        self.u8(v.into())
    }

    /// Writes a `u8` at the current index.
    #[inline(always)]
    pub fn u8<T: Into<u8>>(&mut self, v: T) -> &mut Self {
        self.put([v.into()])
    }

    /// Writes a `u16` at the current index.
    #[inline(always)]
    pub fn u16<T: Into<u16>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Writes a `u32` as a `u24` at the current index.
    ///
    /// # Panics
    ///
    /// Panics if `v` cannot be represented in 24 bits.
    #[inline(always)]
    pub fn u24<T: Into<u32>>(&mut self, v: T) -> &mut Self {
        let v = v.into().to_le_bytes();
        assert_eq!(v[3], 0);
        self.put(&v[..3])
    }

    /// Writes a `u32` at the current index.
    #[inline(always)]
    pub fn u32<T: Into<u32>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Writes a `u64` at the current index.
    #[inline(always)]
    pub fn u64<T: Into<u64>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Writes a `u128` at the current index.
    #[inline(always)]
    pub fn u128<T: Into<u128>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Writes an `i8` at the current index.
    #[inline(always)]
    pub fn i8<T: Into<i8>>(&mut self, v: T) -> &mut Self {
        #[allow(clippy::cast_sign_loss)]
        self.put([v.into() as u8])
    }

    /// Writes an `i16` at the current index.
    #[inline(always)]
    pub fn i16<T: Into<i16>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Writes an `i32` at the current index.
    #[inline(always)]
    pub fn i32<T: Into<i32>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Writes an `i64` at the current index.
    #[inline(always)]
    pub fn i64<T: Into<i64>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Writes an `i128` at the current index.
    #[inline(always)]
    pub fn i128<T: Into<i128>>(&mut self, v: T) -> &mut Self {
        self.put(v.into().to_le_bytes())
    }

    /// Returns whether `n` bytes can be written at the current index.
    #[inline(always)]
    #[must_use]
    pub const fn can_put(&self, n: usize) -> bool {
        self.b.can_put_at(self.i, n)
    }

    /// Writes `v` at the current index.
    #[inline]
    pub fn put<T: AsRef<[u8]>>(&mut self, v: T) -> &mut Self {
        let v = v.as_ref();
        self.b.put_at(self.i, v);
        self.i += v.len();
        self
    }
}

impl AsRef<[u8]> for Packer<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        // SAFETY: Index is guaranteed to be valid
        unsafe { self.b.get_unchecked(self.b.len().min(self.i)..) }
    }
}

impl AsMut<[u8]> for Packer<'_> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        let i = self.b.len().min(self.i);
        // SAFETY: Index is guaranteed to be valid
        unsafe { self.b.get_unchecked_mut(i..) }
    }
}

/// Allows terminating packer method calls with `.into()`.
impl From<&mut Packer<'_>> for () {
    #[inline(always)]
    fn from(_: &mut Packer) -> Self {}
}

/// Unpacker of POD values from a byte slice.
///
/// Any reads past the end of the slice return default values rather than
/// panicking. The caller must check the error status at the end to determine
/// whether all returned values were valid.
///
/// Little-endian encoding is assumed.
#[allow(single_use_lifetimes)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[must_use]
#[repr(transparent)]
pub struct Unpacker<'a>(&'a [u8]);

impl<'a> Unpacker<'a> {
    /// Creates a new unpacker.
    #[inline(always)]
    pub const fn new(b: &'a [u8]) -> Self {
        Self(b)
    }

    /// Creates an unpacker in an error state.
    #[inline(always)]
    pub const fn invalid() -> Self {
        Self(Self::err())
    }

    /// Returns the remaining byte slice, if any. The caller should use
    /// [`Self::is_ok()`] to check whether the returned slice is valid.
    #[inline(always)]
    #[must_use]
    pub const fn into_inner(self) -> &'a [u8] {
        self.0
    }

    /// Returns the remaining number of bytes.
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the byte slice is empty.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns whether all reads were within the bounds of the original byte
    /// slice.
    #[inline(always)]
    #[must_use]
    pub fn is_ok(&self) -> bool {
        !ptr::eq(self.0, Self::err())
    }

    /// Returns the remaining byte slice in a new unpacker, leaving `self`
    /// empty. This is primarily useful in combination with [`map()`].
    ///
    /// [`map()`]: Self::map()
    #[inline]
    pub fn take(&mut self) -> Self {
        // SAFETY: `len()..` range is always valid for get()
        let empty = unsafe { self.0.get_unchecked(self.0.len()..) };
        Self(mem::replace(&mut self.0, empty))
    }

    /// Returns [`Some`] output of `f`, or `None` if `f` fails to consume the
    /// entire slice without reading past the end.
    #[inline]
    #[must_use]
    pub fn map<T>(mut self, f: impl FnOnce(&mut Self) -> T) -> Option<T> {
        let v = f(&mut self);
        (self.is_ok() && self.0.is_empty()).then_some(v)
    }

    /// Returns the output of `f`, or `default` if `f` fails to consume the
    /// entire slice without reading past the end.
    #[inline(always)]
    #[must_use]
    pub fn map_or<T>(self, default: T, f: impl FnOnce(&mut Self) -> T) -> T {
        self.map(f).unwrap_or(default)
    }

    /// Returns the output of `f`, or the output of `default()` if `f` fails to
    /// consume the entire slice without reading past the end.
    #[inline(always)]
    pub fn map_or_else<T>(self, default: impl FnOnce() -> T, f: impl FnOnce(&mut Self) -> T) -> T {
        self.map(f).unwrap_or_else(default)
    }

    /// Splits the remaining byte slice at `i`, and returns two new unpackers,
    /// both of which will be in an error state if `len() < i`.
    #[inline]
    pub const fn split_at(&self, i: usize) -> (Self, Self) {
        let Some(rem) = self.0.len().checked_sub(i) else {
            return (Self(Self::err()), Self(Self::err()));
        };
        let p = self.0.as_ptr();
        // SAFETY: 0 <= i <= len() and i + rem == len()
        unsafe {
            (
                Self(slice::from_raw_parts(p, i)),
                Self(slice::from_raw_parts(p.add(i), rem)),
            )
        }
    }

    /// Advances `self` by `n` bytes, returning a new unpacker for the skipped
    /// bytes or [`None`] if there is an insufficient number of bytes remaining,
    /// in which case any remaining bytes are discarded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use structbuf::{StructBuf, Unpack};
    /// # let mut b = StructBuf::new(4);
    /// # let mut p = b.unpack();
    /// if let Some(mut hdr) = p.skip(2) {
    ///     let _ = hdr.u16();
    ///     assert!(hdr.is_ok() && p.is_ok());
    /// } else {
    ///     assert!(!p.is_ok());
    /// }
    /// ```
    #[inline]
    pub fn skip(&mut self, n: usize) -> Option<Self> {
        let (a, b) = self.split_at(n);
        self.0 = b.0;
        a.is_ok().then_some(a)
    }

    /// Returns the next `u8` as a `bool` where any non-zero value is converted
    /// to `true`.
    #[inline(always)]
    #[must_use]
    pub fn bool(&mut self) -> bool {
        self.u8() != 0
    }

    /// Returns the next `u8`.
    #[inline(always)]
    #[must_use]
    pub fn u8(&mut self) -> u8 {
        // SAFETY: All bit patterns are valid
        unsafe { self.read() }
    }

    /// Returns the next `u16`.
    #[inline(always)]
    #[must_use]
    pub fn u16(&mut self) -> u16 {
        // SAFETY: All bit patterns are valid
        u16::from_le(unsafe { self.read() })
    }

    /// Returns the next `u32`.
    #[inline(always)]
    #[must_use]
    pub fn u32(&mut self) -> u32 {
        // SAFETY: All bit patterns are valid
        u32::from_le(unsafe { self.read() })
    }

    /// Returns the next `u64`.
    #[inline(always)]
    #[must_use]
    pub fn u64(&mut self) -> u64 {
        // SAFETY: All bit patterns are valid
        u64::from_le(unsafe { self.read() })
    }

    /// Returns the next `u128`.
    #[inline(always)]
    #[must_use]
    pub fn u128(&mut self) -> u128 {
        // SAFETY: All bit patterns are valid
        u128::from_le(unsafe { self.read() })
    }

    /// Returns the next `i8`.
    #[inline(always)]
    #[must_use]
    pub fn i8(&mut self) -> i8 {
        // SAFETY: All bit patterns are valid
        unsafe { self.read() }
    }

    /// Returns the next `i16`.
    #[inline(always)]
    #[must_use]
    pub fn i16(&mut self) -> i16 {
        // SAFETY: All bit patterns are valid
        i16::from_le(unsafe { self.read() })
    }

    /// Returns the next `i32`.
    #[inline(always)]
    #[must_use]
    pub fn i32(&mut self) -> i32 {
        // SAFETY: All bit patterns are valid
        i32::from_le(unsafe { self.read() })
    }

    /// Returns the next `i64`.
    #[inline(always)]
    #[must_use]
    pub fn i64(&mut self) -> i64 {
        // SAFETY: All bit patterns are valid
        i64::from_le(unsafe { self.read() })
    }

    /// Returns the next `i128`.
    #[inline(always)]
    #[must_use]
    pub fn i128(&mut self) -> i128 {
        // SAFETY: All bit patterns are valid
        i128::from_le(unsafe { self.read() })
    }

    /// Returns the next `[u8; N]` array.
    #[inline(always)]
    #[must_use]
    pub fn bytes<const N: usize>(&mut self) -> [u8; N] {
        if let Some(rem) = self.0.len().checked_sub(N) {
            // SAFETY: 0 <= N <= len() and the result has an alignment of 1
            unsafe {
                let p = self.0.as_ptr();
                self.0 = slice::from_raw_parts(p.add(N), rem);
                *p.cast()
            }
        } else {
            self.0 = Self::err();
            [0; N]
        }
    }

    /// Returns the next `T`, or `T::default()` if there is an insufficient
    /// number of bytes remaining, in which case any remaining bytes are
    /// discarded.
    ///
    /// # Safety
    ///
    /// `T` must be able to hold the resulting bit pattern.
    #[inline]
    #[must_use]
    pub unsafe fn read<T: Default>(&mut self) -> T {
        if let Some(rem) = self.0.len().checked_sub(mem::size_of::<T>()) {
            // 0 <= size_of::<T>() <= len()
            let p = self.0.as_ptr().cast::<T>();
            self.0 = slice::from_raw_parts(p.add(1).cast(), rem);
            p.read_unaligned()
        } else {
            self.0 = Self::err();
            T::default()
        }
    }

    /// Returns a sentinel byte slice indicating that the original slice was too
    /// short.
    #[inline(always)]
    #[must_use]
    const fn err() -> &'static [u8] {
        // Can't be a const: https://github.com/rust-lang/rust/issues/105536
        // SAFETY: A dangling pointer is valid for a zero-length slice
        unsafe { slice::from_raw_parts(ptr::NonNull::dangling().as_ptr(), 0) }
    }
}

impl<'a> AsRef<[u8]> for Unpacker<'a> {
    #[inline(always)]
    #[must_use]
    fn as_ref(&self) -> &'a [u8] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packer() {
        let mut b = StructBuf::new(4);
        assert_eq!(b.len(), 0);
        assert_eq!(b.capacity(), 4);
        assert_eq!(b.lim(), 4);

        b.append().u8(1);
        assert_eq!(b.len(), 1);
        assert_eq!(b.as_ref(), &[1]);

        b.append().u8(2).u16(0x0403_u16);
        assert_eq!(b.len(), 4);
        assert_eq!(b.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn packer_limit() {
        let mut b = StructBuf::new(4);
        b.append().u8(1).u16(2_u16);
        b.append().u16(3_u16);
    }

    #[test]
    fn packer_overwrite() {
        let mut b = StructBuf::new(4);
        b.append().put([1, 2, 3, 4]);
        assert_eq!(b.as_ref(), &[1, 2, 3, 4]);
        b.at(1).u16(0x0203_u16);
        assert_eq!(b.as_ref(), &[1, 3, 2, 4]);
    }

    #[test]
    fn packer_pad() {
        let mut b = StructBuf::with_capacity(INLINE_CAP + 1);
        // SAFETY: b is valid for b.capacity() bytes
        unsafe { b.as_mut_ptr().write_bytes(0xFF, b.capacity()) };
        b.at(b.capacity() - 1).u8(1);
        assert!(&b[..INLINE_CAP].iter().all(|&v| v == 0));
        assert_eq!(b[INLINE_CAP], 1);

        b.clear();
        b.put_at(4, []);
        assert_eq!(b.as_ref(), &[0, 0, 0, 0]);
    }

    #[test]
    fn unpacker() {
        let mut p = Unpacker::new(&[1, 2, 3]);
        assert_eq!(p.u8(), 1);
        assert!(p.is_ok());
        assert_eq!(p.u16(), 0x0302);
        assert!(p.is_ok());
        assert_eq!(p.u8(), 0);
        assert!(!p.is_ok());

        let mut p = Unpacker::new(&[1]);
        assert_eq!(p.u16(), 0);
        assert!(!p.is_ok());
        assert_eq!(p.u32(), 0);

        let mut p = Unpacker::new(&[1, 2, 3]);
        assert_eq!(p.bytes::<2>(), [1, 2]);
        assert_eq!(p.bytes::<3>(), [0, 0, 0]);
    }

    #[test]
    fn unpacker_take() {
        let mut p = Unpacker::new(&[1, 2, 3]);
        assert_eq!(p.u8(), 1);

        let mut v = p.take();
        assert!(p.is_ok());
        assert!(p.is_empty());
        assert_eq!((v.u8(), v.u8()), (2, 3));
        assert!(v.is_ok());

        assert_eq!(p.u64(), 0);
        assert!(!p.is_ok());
        let v = p.take();
        assert!(!v.is_ok());
    }

    #[test]
    fn unpacker_skip() {
        let mut p = Unpacker::new(&[1, 2, 3]);
        let mut v = p.skip(2).unwrap();

        assert_eq!((v.u8(), v.u8()), (1, 2));
        assert_eq!(p.u8(), 3);

        assert!(p.skip(0).unwrap().is_ok());
        assert!(p.is_ok());
        assert!(p.skip(1).is_none());
        assert!(!p.is_ok());
    }
}
