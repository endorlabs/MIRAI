// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// A test that visits `BlockVisitor::visit_unevaluated_const` with a non-promoted MIR constant
// that resolves to an instance with generic args.

pub trait MyTrait {
    const MY_ASSOC_CONST: u8;
}

pub struct MyConstGenericImpl<const N: u8>;

impl<const N: u8> MyTrait for MyConstGenericImpl<N> {
    const MY_ASSOC_CONST: u8 = N;
}

pub fn foo<T>(_x: T) {}

pub fn main() {
    foo(MyConstGenericImpl::<1>::MY_ASSOC_CONST);
}