// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// A test that uses built-in contracts for the Vec struct.

#[macro_use]
extern crate mirai_annotations;

pub fn main() {
    let v: Vec<i32> = Vec::new();
    verify!(v.is_empty());
}
