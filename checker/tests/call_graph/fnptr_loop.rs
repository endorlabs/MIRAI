// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// Call graph with function pointer calls, a single loop, single type, no dominance.

fn fn1(x: u32) -> u32 {
    fn2(x, &(fn3 as fn(u32) -> u32))
}
fn fn2(x: u32, fn3: &fn(u32) -> u32) -> u32 {
    fn3(x)
}
fn fn3(x: u32) -> u32 {
    if x > 1 {
        fn1(x - 1)
    } else {
        x
    }
}
pub fn main() {
    let x = 3;
    fn1(x);
}

/* CONFIG
{
    "reductions": [],
    "included_crates": [],
    "datalog_config": {
        "datalog_backend": "DifferentialDatalog"
    }
}
*/

/* EXPECTED:DOT
digraph {
    0 [ label = "\"fnptr_loop::main\"" ]
    1 [ label = "\"fnptr_loop::fn1\"" ]
    2 [ label = "\"fnptr_loop::fn2\"" ]
    3 [ label = "\"fnptr_loop::fn3\"" ]
    0 -> 1 [ ]
    1 -> 2 [ ]
    1 -> 2 [ ]
    2 -> 3 [ ]
    3 -> 1 [ ]
}
*/

/* EXPECTED:DDLOG
start;
insert Edge(0,0,1);
insert Edge(1,1,2);
insert Edge(2,1,2);
insert Edge(3,2,3);
insert Edge(4,3,1);
insert EdgeType(0,0);
insert EdgeType(1,0);
insert EdgeType(2,1);
insert EdgeType(3,0);
insert EdgeType(4,0);
commit;
*/

/* EXPECTED:TYPEMAP
{
  "0": "u32",
  "1": "&fn(u32) -> u32"
}
*/

/* EXPECTED:CALL_SITES{
  "files": [
    "tests/call_graph/fnptr_loop.rs"
  ],
  "callables": [
    {
      "name": "fnptr_loop.fn1",
      "file_index": 0,
      "first_line": 9,
      "local": true
    },
    {
      "name": "fnptr_loop.fn2",
      "file_index": 0,
      "first_line": 12,
      "local": true
    },
    {
      "name": "fnptr_loop.fn3",
      "file_index": 0,
      "first_line": 15,
      "local": true
    },
    {
      "name": "fnptr_loop.main",
      "file_index": 0,
      "first_line": 22,
      "local": true
    }
  ],
  "calls": [
    [
      0,
      10,
      5,
      0,
      1
    ],
    [
      0,
      13,
      5,
      1,
      2
    ],
    [
      0,
      17,
      9,
      2,
      0
    ],
    [
      0,
      24,
      5,
      3,
      0
    ]
  ]
}*/
