// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

// Linear call graph with single type, no dominance, no loops.
// Includes call to println which is folded out.

fn fn1(x: u32, fn2: &fn(u32) -> u32) -> u32 {
    fn2(x)
}
fn fn2(x: u32) -> u32 {
    fn3(x)
}
fn fn3(x: u32) -> u32 {
    println!();
    x
}
pub fn main() {
    let x = 1;
    fn1(x, &(fn2 as fn(u32) -> u32));
}

/* CONFIG
{
    "reductions": ["Fold"],
    "included_crates": ["fnptr_fold"],
    "datalog_config": {
        "datalog_backend": "DifferentialDatalog"
    }
}
*/

/* EXPECTED:DOT
digraph {
    0 [ label = "\"fnptr_fold::main\"" ]
    1 [ label = "\"fnptr_fold::fn1\"" ]
    2 [ label = "\"fnptr_fold::fn2\"" ]
    3 [ label = "\"fnptr_fold::fn3\"" ]
    0 -> 1 [ ]
    0 -> 1 [ ]
    1 -> 2 [ ]
    2 -> 3 [ ]
}
*/

/* EXPECTED:DDLOG
start;
insert Edge(0,0,1);
insert Edge(1,0,1);
insert Edge(2,1,2);
insert Edge(3,2,3);
insert EdgeType(0,0);
insert EdgeType(1,1);
insert EdgeType(2,0);
insert EdgeType(3,0);
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
    "tests/call_graph/fnptr_fold.rs",
    "/rustc/2a3e63551fe21458637480a97b65a2d15dec8062/library/std/src/io/stdio.rs",
    "/rustc/2a3e63551fe21458637480a97b65a2d15dec8062/library/core/src/fmt/mod.rs",
    "/rustc/2a3e63551fe21458637480a97b65a2d15dec8062/library/core/src/slice/mod.rs",
    "/rustc/2a3e63551fe21458637480a97b65a2d15dec8062/library/core/src/ptr/metadata.rs"
  ],
  "callables": [
    {
      "name": "/fnptr_fold/fn1(u32,&ReBound(DebruijnIndex(0), BoundRegion { var: 0, kind: BrNamed(DefId(0:7 ~ fnptr_fold[33be]::fn1::'_), '_) }) Binder(fn(u32) -> u32, []))->u32",
      "file_index": 0,
      "first_line": 10,
      "local": true
    },
    {
      "name": "/fnptr_fold/fn2(u32)->u32",
      "file_index": 0,
      "first_line": 13,
      "local": true
    },
    {
      "name": "/fnptr_fold/fn3(u32)->u32",
      "file_index": 0,
      "first_line": 16,
      "local": true
    },
    {
      "name": "/fnptr_fold/main()->()",
      "file_index": 0,
      "first_line": 20,
      "local": true
    },
    {
      "name": "/std/std::io::_print(std::fmt::Arguments<ReBound(DebruijnIndex(0), BoundRegion { var: 0, kind: BrNamed(DefId(1:12928 ~ std[ddb8]::io::stdio::_print::'_), '_) })>)->()",
      "file_index": 1,
      "first_line": 1096,
      "local": false
    },
    {
      "name": "/core/std::fmt::Arguments::<'a>::new_const(&ReEarlyParam(DefId(2:9935 ~ core[c4b9]::fmt::{impl#2}::'a), 0, 'a) [&ReStatic str])->std::fmt::Arguments<ReEarlyParam(DefId(2:9935 ~ core[c4b9]::fmt::{impl#2}::'a), 0, 'a)>",
      "file_index": 2,
      "first_line": 321,
      "local": true
    },
    {
      "name": "/core/core::slice::<impl [T]>::len(&ReBound(DebruijnIndex(0), BoundRegion { var: 0, kind: BrNamed(DefId(2:59819 ~ core[c4b9]::slice::{impl#0}::len::'_), '_) }) [T/#0])->usize",
      "file_index": 3,
      "first_line": 137,
      "local": true
    },
    {
      "name": "/core/std::ptr::metadata(*const T/#0)->Alias(Projection, AliasTy { args: [T/#0], def_id: DefId(2:1880 ~ core[c4b9]::ptr::metadata::Pointee::Metadata) })",
      "file_index": 4,
      "first_line": 94,
      "local": true
    }
  ],
  "calls": [
    [
      0,
      11,
      5,
      0,
      1
    ],
    [
      0,
      14,
      5,
      1,
      2
    ],
    [
      0,
      22,
      5,
      3,
      0
    ],
    [
      0,
      17,
      5,
      2,
      4
    ],
    [
      0,
      17,
      5,
      2,
      5
    ],
    [
      2,
      322,
      12,
      5,
      6
    ],
    [
      2,
      323,
      13,
      5,
      5
    ],
    [
      3,
      138,
      9,
      6,
      7
    ]
  ]
}*/
