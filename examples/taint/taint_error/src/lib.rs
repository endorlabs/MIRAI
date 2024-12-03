// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
#![allow(unexpected_cfgs)]

#[macro_use]
extern crate mirai_annotations;

use std::sync::Arc;

pub struct Foo {
    pub arc: Arc<[i32]>,
}

pub fn source(arc: Arc<[i32]>) -> Foo {
    let result = Foo { arc };
    set_model_field!(&result, tainted, true);
    result
}

pub fn use_arc(f: Foo) -> (Arc<[i32]>, i32) {
    precondition!(!get_model_field!(&f, tainted, false));
    let sum: i32 = f.arc.iter().sum();
    unsafe {
        let arc_array = f.arc;
        let ptr = Arc::into_raw(arc_array);
        (Arc::from_raw(ptr), sum)
    }
}
