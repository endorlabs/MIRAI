// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// A test that uses a widened summary.

//use mirai_annotations::*;

fn fact(n: u8) -> u128 {
    if n == 0 {
        1
    } else {
        let n1fac = fact(n - 1);
        //assume!(n1fac <= std::u128::MAX / (n as u128));
        (n as u128) * n1fac //~ possible attempt to multiply with overflow
    }
}

pub fn main() {
    let _x = fact(10);
}
