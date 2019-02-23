// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// A test that calls visit_aggregate

pub fn main() {
    let x = [1, 2];
    let _y = x[2]; //~ array index out of bounds
}

pub fn foo(c: bool) {
    let x = [1, 2];
    let i = if c { 1 } else { 0 };
    let _y = x[i];
}

pub fn bar(c: bool) {
    let x = [1, 2];
    let i = if c { 1 } else { 2 };
    let _y = x[i]; //~ possible array index out of bounds
}