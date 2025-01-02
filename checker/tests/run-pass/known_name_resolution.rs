// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// A test that visits `KnownNamesCache::get_known_name_for` with a `DefId` for a function
// that's defined inside the body a `KnownName` function, but has a different signature.
// See https://github.com/endorlabs/MIRAI/issues/26#issuecomment-2566638406 for details.

fn try_remove<T>(v: &mut Vec<T>, idx: usize) -> Option<T> {
    if idx < v.len() {
        Some(v.remove(idx))
    } else {
        None
    }
}

pub fn main() {
    let mut data = vec![1, 2, 3];
    try_remove(&mut data, 0);
}