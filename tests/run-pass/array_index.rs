// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// A test that visits the Place::Projection case of Visitor::visit_place
// and the ProjectionElem::Index case of Visitor::visit_projection_elem.

pub fn foo(arr: &mut [i32], i: usize) {
    arr[i] = 123; //~ possible array index out of bounds
    // If we get here i is known to be within bounds, so no warning below.
    bar(arr, i);
}

fn bar(arr: &mut [i32], i: usize) {
    arr[i] = 123;
    debug_assert!(arr[i] == 123);
}

fn get_elem(arr: &[i32], i: usize) -> i32 {
    arr[i]
}

pub fn main() {
    let arr = [1, 2, 3];
    let elem = get_elem(&arr, 1);
    debug_assert!(elem == 2);
}

