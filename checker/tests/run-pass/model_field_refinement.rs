// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// A test where function preconditions involve model/tag fields of parameters

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use mirai_annotations::*;

use mirai_annotations::TagPropagationSet;

struct Foo {}

fn func1(foo: &Foo) {
    precondition!(get_model_field!(foo, value, 0) == 99991);
}

pub fn test1() {
    let foo = Foo {};
    set_model_field!(&foo, value, 99991);
    func1(&foo);
    verify!(get_model_field!(&foo, value, 0) == 99991);
}

struct TaintKind<const MASK: TagPropagationSet> {}

const TAINT: TagPropagationSet = tag_propagation_set!();

type Taint = TaintKind<TAINT>;

fn func2(foo: &Foo) {
    precondition!(has_tag!(foo, Taint));
}

pub fn test2() {
    let foo = Foo {};
    add_tag!(&foo, Taint);
    func2(&foo);
    verify!(has_tag!(&foo, Taint));
}

pub fn main() {}
