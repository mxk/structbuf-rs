StructBuf
=========

[![crates.io](https://img.shields.io/crates/v/structbuf?style=for-the-badge)](https://crates.io/crates/structbuf)
[![docs.rs](https://img.shields.io/badge/docs.rs-structbuf-66c2a5?style=for-the-badge&logo=docs.rs)](https://docs.rs/structbuf)
[![License](https://img.shields.io/crates/l/structbuf?style=for-the-badge)](https://choosealicense.com/licenses/mpl-2.0/)

This library provides a capacity-limited buffer for encoding and decoding structured data. The primary use case is for safely handling small, variable-length message packets sent over the network or other transports.

The encoder ensures that the message size never exceeds a pre-configured limit. The decoder ensures that malformed or malicious input does not cause the program to panic.

## `no_std` support

`structbuf` is `no_std` by default.

## Example

```
cargo add structbuf
```

```rust
use structbuf::StructBuf;

let mut b = StructBuf::new(4);
b.append().u8(1).u16(2_u16).u8(3);
// b.u8(4); Would panic

let mut p = b.unpack();
assert_eq!(p.u8(), 1);
assert_eq!(p.u16(), 2);
assert_eq!(p.u8(), 3);
assert!(p.is_ok());

assert_eq!(p.u32(), 0);
assert!(!p.is_ok());
```
