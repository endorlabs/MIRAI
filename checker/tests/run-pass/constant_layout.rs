// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use mirai_annotations::*;

#[derive(Clone, Copy)]
struct Mixed {
    small: u8,
    large: u64,
    middle: u16,
}

const VALUE: Mixed = Mixed {
    small: 3,
    large: 5000,
    middle: 11,
};

pub fn check() {
    let value = VALUE;
    verify!(value.small == 3);
    verify!(value.large == 5000);
    verify!(value.middle == 11);
}

fn fields(value: Mixed) -> (u8, u64, u16) {
    (value.small, value.large, value.middle)
}

pub fn pass_constant_by_value() {
    let (small, large, middle) = fields(VALUE);
    verify!(small == 3);
    verify!(large == 5000);
    verify!(middle == 11);
}

pub fn main() {}
