// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// Linear call graph with function pointer calls, single type, dominance, and a loop.

fn fn1(x: u32, fn2: &fn(u32) -> u32, fn3: &fn(u32) -> u32) -> u32 {
    let y = fn2(x);
    fn3(y)
}
fn fn2(x: u32) -> u32 {
    x + 2
}
fn fn3(x: u32) -> u32 {
    fn4(x)
}
fn fn4(x: u32) -> u32 {
    if x > 1 {
        fn3(x - 1)
    } else {
        x
    }
}
pub fn main() {
    let x = 1;
    fn1(x, &(fn2 as fn(u32) -> u32), &(fn3 as fn(u32) -> u32));
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
    0 [ label = "\"fnptr_dom_loop::main\"" ]
    1 [ label = "\"fnptr_dom_loop::fn1\"" ]
    2 [ label = "\"fnptr_dom_loop::fn2\"" ]
    3 [ label = "\"fnptr_dom_loop::fn3\"" ]
    4 [ label = "\"fnptr_dom_loop::fn4\"" ]
    0 -> 1 [ ]
    0 -> 1 [ ]
    1 -> 2 [ ]
    1 -> 3 [ ]
    3 -> 4 [ ]
    4 -> 3 [ ]
}
*/

/* EXPECTED:DDLOG
start;
insert Dom(2,3);
insert Edge(0,0,1);
insert Edge(1,0,1);
insert Edge(2,1,2);
insert Edge(3,1,3);
insert Edge(4,3,4);
insert Edge(5,4,3);
insert EdgeType(0,0);
insert EdgeType(1,1);
insert EdgeType(2,0);
insert EdgeType(3,0);
insert EdgeType(4,0);
insert EdgeType(5,0);
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
    "tests/call_graph/fnptr_dom_loop.rs"
  ],
  "callables": [
    {
      "name": "/fnptr_dom_loop/fn1(u32,&fn(u32) -> u32,&fn(u32) -> u32)->u32",
      "file_index": 0,
      "first_line": 9,
      "local": true
    },
    {
      "name": "/fnptr_dom_loop/fn2(u32)->u32",
      "file_index": 0,
      "first_line": 13,
      "local": true
    },
    {
      "name": "/fnptr_dom_loop/fn3(u32)->u32",
      "file_index": 0,
      "first_line": 16,
      "local": true
    },
    {
      "name": "/fnptr_dom_loop/fn4(u32)->u32",
      "file_index": 0,
      "first_line": 19,
      "local": true
    },
    {
      "name": "/fnptr_dom_loop/main()->()",
      "file_index": 0,
      "first_line": 26,
      "local": true
    }
  ],
  "calls": [
    [
      0,
      10,
      13,
      0,
      1
    ],
    [
      0,
      11,
      5,
      0,
      2
    ],
    [
      0,
      17,
      5,
      2,
      3
    ],
    [
      0,
      21,
      9,
      3,
      2
    ],
    [
      0,
      28,
      5,
      4,
      0
    ]
  ]
}*/
