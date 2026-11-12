// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// A test that does unsafe casting and manipulation of pointers to transparent wrappers

// MIRAI_FLAGS --diag=verify
#![allow(dead_code)]

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const RUNNING: usize = 0x1;
const STATE_MASK: usize = 0x3;

#[repr(align(4))] // Ensure the two lower bits are free to use as state bits.
pub struct Waiter {
    // A scalar handle keeps this pointer test independent of the OS thread implementation.
    thread: Cell<Option<NonZeroUsize>>,
    signaled: AtomicBool,
    next: *const Waiter,
}

pub struct WaiterQueue<'a> {
    state_and_queue: &'a AtomicUsize,
    set_state_on_drop_to: usize,
}

impl Drop for WaiterQueue<'_> {
    fn drop(&mut self) {
        let state_and_queue = self
            .state_and_queue
            .swap(self.set_state_on_drop_to, Ordering::AcqRel);

        assert_eq!(state_and_queue & STATE_MASK, RUNNING);

        unsafe {
            let mut queue = (state_and_queue & !STATE_MASK) as *const Waiter;
            while !queue.is_null() {
                let next = (*queue).next;
                //~ related location
                let _thread = (*queue).thread.replace(None).unwrap(); //~ called `Option::unwrap()` on a `None` value
                (*queue).signaled.store(true, Ordering::Release);
                queue = next;
            }
        }
    }
}

pub fn main() {}
