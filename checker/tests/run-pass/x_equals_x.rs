// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// A test that checks if expressions like x == x can be simplified.

#[macro_use]
extern crate mirai_annotations;

pub fn main() {
    foo(1, 2.0);
}

pub fn bar(y: f32) {
    if y == y {
        verify!(true);
    } else {
        verify!(false); //~ provably false verification condition
    }
}

fn foo(x: i32, y: f32) {
    if x == x {
        verify!(true);
    } else {
        verify!(false);
    }
    if y == y {
        verify!(true);
    } else {
        verify!(false); //~ provably false verification condition
    }
}
