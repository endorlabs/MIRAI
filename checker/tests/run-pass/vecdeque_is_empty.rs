// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// A test for VecDeque::is_empty

#[macro_use]
extern crate mirai_annotations;

use std::collections::VecDeque;

pub fn main() {
    let v: VecDeque<i32> = VecDeque::new();
    verify!(v.is_empty());
}
