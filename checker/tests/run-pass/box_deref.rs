// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

#[macro_use]
extern crate mirai_annotations;

fn len(b: Box<[i32]>) -> usize {
    b.len()
}

pub fn main() {
    let boxed_array = Box::new([10]);
    verify!(boxed_array[0] == 10);
    verify!(len(boxed_array) == 1);
}
