// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use mirai_annotations::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn swap_returns_old_value() {
    let mut value = AtomicUsize::new(1);
    verify!(value.swap(2, Ordering::AcqRel) == 1);
    verify!(*value.get_mut() == 2);
    verify!(value.swap(3, Ordering::Relaxed) == 2);
    verify!(*value.get_mut() == 3);
}

pub fn main() {}
