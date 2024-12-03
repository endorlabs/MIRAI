// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

#![allow(internal_features)]
#![feature(allocator_api)]
#![feature(core_intrinsics)]
#![feature(discriminant_kind)]
#![feature(f16)]
#![feature(f128)]
#![feature(hashmap_internals)]
#![feature(pattern)]
#![feature(ptr_internals)]
#![feature(ptr_metadata)]
#![feature(ptr_alignment_type)]
#![allow(unexpected_cfgs)]

#[macro_use]
extern crate mirai_annotations;

#[macro_use]
mod macros;

pub mod foreign_contracts;
