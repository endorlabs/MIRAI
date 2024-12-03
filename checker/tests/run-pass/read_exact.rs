// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// A test case that checks that effects that join old values with updates do not
// overwrite the caller's state when the joined value refines to the value that
// was there in the previous state.
// This matters when the target path is a slice, because the slice might alias
// another slice/index path and hence writing the pre-state value to the post-state
// (instead of just relying on it being there already) might overwrite a previous effect.

use mirai_annotations::*;
use std::io::Read;

fn read_exact(a: &[u8], buf: &mut [u8]) {
    precondition!(buf.len() <= a.len());
    if buf.len() == 1 {
        buf[0] = a[0];
    } else {
        buf.copy_from_slice(a);
    }
}

pub fn t1(c: &[u8]) {
    precondition!(1 <= c.len());
    let mut buf = [0; 1];
    let _ = read_exact(c, &mut buf);
    verify!(buf[0] == 0); //~ possible false verification condition
}

fn read_u8(_self: &mut std::io::Cursor<&[u8]>) -> std::io::Result<u8> {
    let mut buf = [0; 1];
    _self.read_exact(&mut buf)?;
    Ok(buf[0])
}
pub fn t2(val: &[u8]) -> std::io::Result<()> {
    let mut reader = std::io::Cursor::new(val);
    let _num_nibbles = read_u8(&mut reader)? as usize;
    // verify!(num_nibbles == 0); // ~ possible false verification condition
    Ok(())
}

pub fn main() {}
