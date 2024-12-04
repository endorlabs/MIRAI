// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use rustc_hir::def_id::DefId;
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_middle::ty::TyCtxt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Well known definitions (language provided items) that are treated in special ways.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialOrd, PartialEq, Hash, Ord)]
pub enum KnownNames {
    /// This is not a known name
    None,
    AllocRawVecMinNonZeroCap,
    MiraiAbstractValue,
    MiraiAddTag,
    MiraiAssume,
    MiraiAssumePreconditions,
    MiraiDoesNotHaveTag,
    MiraiGetModelField,
    MiraiHasTag,
    MiraiPostcondition,
    MiraiPrecondition,
    MiraiPreconditionStart,
    MiraiResult,
    MiraiSetModelField,
    MiraiVerify,
    RustAlloc,
    RustAllocZeroed,
    RustDealloc,
    RustRealloc,
    StdCloneClone,
    StdFutureFromGenerator,
    StdIntrinsicsArithOffset,
    StdIntrinsicsBitreverse,
    StdIntrinsicsBswap,
    StdIntrinsicsCeilf16,
    StdIntrinsicsCeilf32,
    StdIntrinsicsCeilf64,
    StdIntrinsicsCeilf128,
    StdIntrinsicsConstEvalSelect,
    StdIntrinsicsCopy,
    StdIntrinsicsCopyNonOverlapping,
    StdIntrinsicsCopysignf16,
    StdIntrinsicsCopysignf32,
    StdIntrinsicsCopysignf64,
    StdIntrinsicsCopysignf128,
    StdIntrinsicsCosf16,
    StdIntrinsicsCosf32,
    StdIntrinsicsCosf64,
    StdIntrinsicsCosf128,
    StdIntrinsicsCtlz,
    StdIntrinsicsCtlzNonzero,
    StdIntrinsicsCtpop,
    StdIntrinsicsCttz,
    StdIntrinsicsCttzNonzero,
    StdIntrinsicsDiscriminantValue,
    StdIntrinsicsExp2f16,
    StdIntrinsicsExp2f32,
    StdIntrinsicsExp2f64,
    StdIntrinsicsExp2f128,
    StdIntrinsicsExpf16,
    StdIntrinsicsExpf32,
    StdIntrinsicsExpf64,
    StdIntrinsicsExpf128,
    StdIntrinsicsFabsf16,
    StdIntrinsicsFabsf32,
    StdIntrinsicsFabsf64,
    StdIntrinsicsFabsf128,
    StdIntrinsicsFaddAlgebraic,
    StdIntrinsicsFaddFast,
    StdIntrinsicsFdivAlgebraic,
    StdIntrinsicsFdivFast,
    StdIntrinsicsFloatToIntUnchecked,
    StdIntrinsicsFloorf16,
    StdIntrinsicsFloorf32,
    StdIntrinsicsFloorf64,
    StdIntrinsicsFloorf128,
    StdIntrinsicsFmulAlgebraic,
    StdIntrinsicsFmulFast,
    StdIntrinsicsFremAlgebraic,
    StdIntrinsicsFremFast,
    StdIntrinsicsFsubAlgebraic,
    StdIntrinsicsFsubFast,
    StdIntrinsicsIsValStaticallyKnown,
    StdIntrinsicsLog10f16,
    StdIntrinsicsLog10f32,
    StdIntrinsicsLog10f64,
    StdIntrinsicsLog10f128,
    StdIntrinsicsLog2f16,
    StdIntrinsicsLog2f32,
    StdIntrinsicsLog2f64,
    StdIntrinsicsLog2f128,
    StdIntrinsicsLogf16,
    StdIntrinsicsLogf32,
    StdIntrinsicsLogf64,
    StdIntrinsicsLogf128,
    StdIntrinsicsMaxnumf16,
    StdIntrinsicsMaxnumf32,
    StdIntrinsicsMaxnumf64,
    StdIntrinsicsMaxnumf128,
    StdIntrinsicsMinAlignOfVal,
    StdIntrinsicsMinnumf16,
    StdIntrinsicsMinnumf32,
    StdIntrinsicsMinnumf64,
    StdIntrinsicsMinnumf128,
    StdIntrinsicsMulWithOverflow,
    StdIntrinsicsNearbyintf16,
    StdIntrinsicsNearbyintf32,
    StdIntrinsicsNearbyintf64,
    StdIntrinsicsNearbyintf128,
    StdIntrinsicsNeedsDrop,
    StdIntrinsicsOffset,
    StdIntrinsicsPowf16,
    StdIntrinsicsPowf32,
    StdIntrinsicsPowf64,
    StdIntrinsicsPowf128,
    StdIntrinsicsPowif16,
    StdIntrinsicsPowif32,
    StdIntrinsicsPowif64,
    StdIntrinsicsPowif128,
    StdIntrinsicsPrefAlignOfVal,
    StdIntrinsicsRawEq,
    StdIntrinsicsRintf16,
    StdIntrinsicsRintf32,
    StdIntrinsicsRintf64,
    StdIntrinsicsRintf128,
    StdIntrinsicsRoundf16,
    StdIntrinsicsRoundf32,
    StdIntrinsicsRoundf64,
    StdIntrinsicsRoundf128,
    StdIntrinsicsRevenf16,
    StdIntrinsicsRevenf32,
    StdIntrinsicsRevenf64,
    StdIntrinsicsRevenf128,
    StdIntrinsicsSinf16,
    StdIntrinsicsSinf32,
    StdIntrinsicsSinf64,
    StdIntrinsicsSinf128,
    StdIntrinsicsSizeOf,
    StdIntrinsicsSizeOfVal,
    StdIntrinsicsSqrtf16,
    StdIntrinsicsSqrtf32,
    StdIntrinsicsSqrtf64,
    StdIntrinsicsSqrtf128,
    StdIntrinsicsThreeWayCompare,
    StdIntrinsicsTransmute,
    StdIntrinsicsTruncf16,
    StdIntrinsicsTruncf32,
    StdIntrinsicsTruncf64,
    StdIntrinsicsTruncf128,
    StdIntrinsicsVariantCount,
    StdIntrinsicsWriteBytes,
    StdMarkerPhantomData,
    StdMemReplace,
    StdOpsFunctionFnCall,
    StdOpsFunctionFnMutCallMut,
    StdOpsFunctionFnOnceCallOnce,
    StdPanickingAssertFailed,
    StdPanickingBeginPanic,
    StdPanickingBeginPanicFmt,
    StdPtrSwapNonOverlapping,
    StdSliceCmpMemcmp,
}

/// An analysis lifetime cache that contains a map from def ids to known names.
pub struct KnownNamesCache {
    name_cache: HashMap<DefId, KnownNames>,
}

type Iter<'a> = std::slice::Iter<'a, rustc_hir::definitions::DisambiguatedDefPathData>;

impl KnownNamesCache {
    /// Create an empty known names cache.
    /// This cache is re-used by every successive MIR visitor instance.
    pub fn create_cache_from_language_items() -> KnownNamesCache {
        let name_cache = HashMap::new();
        KnownNamesCache { name_cache }
    }

    /// Get the well known name for the given def id and cache the association.
    /// I.e. the first call for an unknown def id will be somewhat costly but
    /// subsequent calls will be cheap. If the def_id does not have an actual well
    /// known name, this returns KnownNames::None.
    pub fn get(&mut self, tcx: TyCtxt<'_>, def_id: DefId) -> KnownNames {
        *self
            .name_cache
            .entry(def_id)
            .or_insert_with(|| Self::get_known_name_for(tcx, def_id))
    }

    /// Uses information obtained from tcx to figure out which well known name (if any)
    /// this def id corresponds to.
    pub fn get_known_name_for(tcx: TyCtxt<'_>, def_id: DefId) -> KnownNames {
        use DefPathData::*;

        let def_path = &tcx.def_path(def_id);
        let def_path_data_iter = def_path.data.iter();

        // helper to get next elem from def path and return its name, if it has one
        let get_path_data_elem_name =
            |def_path_data_elem: Option<&rustc_hir::definitions::DisambiguatedDefPathData>| {
                def_path_data_elem.and_then(|ref elem| {
                    let DisambiguatedDefPathData { data, .. } = elem;
                    match &data {
                        TypeNs(name) | ValueNs(name) => Some(*name),
                        _ => None,
                    }
                })
            };

        let is_foreign_module =
            |def_path_data_elem: Option<&rustc_hir::definitions::DisambiguatedDefPathData>| {
                if let Some(elem) = def_path_data_elem {
                    let DisambiguatedDefPathData { data, .. } = elem;
                    matches!(&data, ForeignMod)
                } else {
                    false
                }
            };

        let path_data_elem_as_disambiguator = |def_path_data_elem: Option<
            &rustc_hir::definitions::DisambiguatedDefPathData,
        >| {
            def_path_data_elem.map(|DisambiguatedDefPathData { disambiguator, .. }| *disambiguator)
        };

        let get_known_name_for_alloc_namespace = |mut def_path_data_iter: Iter<'_>| {
            if is_foreign_module(def_path_data_iter.next()) {
                get_path_data_elem_name(def_path_data_iter.next())
                    .map(|n| match n.as_str() {
                        "__rust_alloc" => KnownNames::RustAlloc,
                        "__rust_alloc_zeroed" => KnownNames::RustAllocZeroed,
                        "__rust_dealloc" => KnownNames::RustDealloc,
                        "__rust_realloc" => KnownNames::RustRealloc,
                        _ => KnownNames::None,
                    })
                    .unwrap_or(KnownNames::None)
            } else {
                KnownNames::None
            }
        };

        let get_known_name_for_clone_trait = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "clone" => KnownNames::StdCloneClone,
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_clone_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "Clone" => get_known_name_for_clone_trait(def_path_data_iter),
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_future_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "from_generator" => KnownNames::StdFutureFromGenerator,
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_instrinsics_foreign_namespace =
            |mut def_path_data_iter: Iter<'_>| {
                get_path_data_elem_name(def_path_data_iter.next())
                    .map(|n| match n.as_str() {
                        "float_to_int_unchecked" => KnownNames::StdIntrinsicsFloatToIntUnchecked,
                        _ => KnownNames::None,
                    })
                    .unwrap_or(KnownNames::None)
            };

        let get_known_name_for_intrinsics_namespace = |mut def_path_data_iter: Iter<'_>| {
            let current_elem = def_path_data_iter.next();
            match path_data_elem_as_disambiguator(current_elem) {
                Some(0) => {
                    if is_foreign_module(current_elem) {
                        get_known_name_for_instrinsics_foreign_namespace(def_path_data_iter)
                    } else {
                        get_path_data_elem_name(current_elem)
                            .map(|n| match n.as_str() {
                                "arith_offset" => KnownNames::StdIntrinsicsArithOffset,
                                "bitreverse" => KnownNames::StdIntrinsicsBitreverse,
                                "bswap" => KnownNames::StdIntrinsicsBswap,
                                "ceilf16" => KnownNames::StdIntrinsicsCeilf16,
                                "ceilf32" => KnownNames::StdIntrinsicsCeilf32,
                                "ceilf64" => KnownNames::StdIntrinsicsCeilf64,
                                "ceilf128" => KnownNames::StdIntrinsicsCeilf128,
                                "compare_bytes" => KnownNames::StdSliceCmpMemcmp,
                                "const_eval_select" => KnownNames::StdIntrinsicsConstEvalSelect,
                                "copy" => KnownNames::StdIntrinsicsCopy,
                                "copy_nonoverlapping" => {
                                    if def_path_data_iter.next().is_some() {
                                        KnownNames::None
                                    } else {
                                        KnownNames::StdIntrinsicsCopyNonOverlapping
                                    }
                                }
                                "copysignf16" => KnownNames::StdIntrinsicsCopysignf16,
                                "copysignf32" => KnownNames::StdIntrinsicsCopysignf32,
                                "copysignf64" => KnownNames::StdIntrinsicsCopysignf64,
                                "copysignf128" => KnownNames::StdIntrinsicsCopysignf128,
                                "cosf16" => KnownNames::StdIntrinsicsCosf16,
                                "cosf32" => KnownNames::StdIntrinsicsCosf32,
                                "cosf64" => KnownNames::StdIntrinsicsCosf64,
                                "cosf128" => KnownNames::StdIntrinsicsCosf128,
                                "ctlz" => KnownNames::StdIntrinsicsCtlz,
                                "ctlz_nonzero" => KnownNames::StdIntrinsicsCtlzNonzero,
                                "ctpop" => KnownNames::StdIntrinsicsCtpop,
                                "cttz" => KnownNames::StdIntrinsicsCttz,
                                "cttz_nonzero" => KnownNames::StdIntrinsicsCttzNonzero,
                                "discriminant_value" => KnownNames::StdIntrinsicsDiscriminantValue,
                                "exp2f16" => KnownNames::StdIntrinsicsExp2f16,
                                "exp2f32" => KnownNames::StdIntrinsicsExp2f32,
                                "exp2f64" => KnownNames::StdIntrinsicsExp2f64,
                                "exp2f128" => KnownNames::StdIntrinsicsExp2f128,
                                "expf16" => KnownNames::StdIntrinsicsExpf16,
                                "expf32" => KnownNames::StdIntrinsicsExpf32,
                                "expf64" => KnownNames::StdIntrinsicsExpf64,
                                "expf128" => KnownNames::StdIntrinsicsExpf128,
                                "fabsf16" => KnownNames::StdIntrinsicsFabsf16,
                                "fabsf32" => KnownNames::StdIntrinsicsFabsf32,
                                "fabsf64" => KnownNames::StdIntrinsicsFabsf64,
                                "fabsf128" => KnownNames::StdIntrinsicsFabsf128,
                                "fadd_algebraic" => KnownNames::StdIntrinsicsFaddAlgebraic,
                                "fadd_fast" => KnownNames::StdIntrinsicsFaddFast,
                                "fdiv_algebraic" => KnownNames::StdIntrinsicsFdivAlgebraic,
                                "fdiv_fast" => KnownNames::StdIntrinsicsFdivFast,
                                "floorf16" => KnownNames::StdIntrinsicsFloorf16,
                                "floorf32" => KnownNames::StdIntrinsicsFloorf32,
                                "floorf64" => KnownNames::StdIntrinsicsFloorf64,
                                "floorf128" => KnownNames::StdIntrinsicsFloorf128,
                                "fmul_algebraic" => KnownNames::StdIntrinsicsFmulAlgebraic,
                                "fmul_fast" => KnownNames::StdIntrinsicsFmulFast,
                                "frem_algebraic" => KnownNames::StdIntrinsicsFremAlgebraic,
                                "frem_fast" => KnownNames::StdIntrinsicsFremFast,
                                "fsub_algebraic" => KnownNames::StdIntrinsicsFsubAlgebraic,
                                "fsub_fast" => KnownNames::StdIntrinsicsFsubFast,
                                "is_val_statically_known" => {
                                    KnownNames::StdIntrinsicsIsValStaticallyKnown
                                }
                                "log10f16" => KnownNames::StdIntrinsicsLog10f16,
                                "log10f32" => KnownNames::StdIntrinsicsLog10f32,
                                "log10f64" => KnownNames::StdIntrinsicsLog10f64,
                                "log10f128" => KnownNames::StdIntrinsicsLog10f128,
                                "log2f16" => KnownNames::StdIntrinsicsLog2f16,
                                "log2f32" => KnownNames::StdIntrinsicsLog2f32,
                                "log2f64" => KnownNames::StdIntrinsicsLog2f64,
                                "log2f128" => KnownNames::StdIntrinsicsLog2f128,
                                "logf16" => KnownNames::StdIntrinsicsLogf16,
                                "logf32" => KnownNames::StdIntrinsicsLogf32,
                                "logf64" => KnownNames::StdIntrinsicsLogf64,
                                "logf128" => KnownNames::StdIntrinsicsLogf128,
                                "maxnumf16" => KnownNames::StdIntrinsicsMaxnumf16,
                                "maxnumf32" => KnownNames::StdIntrinsicsMaxnumf32,
                                "maxnumf64" => KnownNames::StdIntrinsicsMaxnumf64,
                                "maxnumf128" => KnownNames::StdIntrinsicsMaxnumf128,
                                "min_align_of_val" => KnownNames::StdIntrinsicsMinAlignOfVal,
                                "minnumf16" => KnownNames::StdIntrinsicsMinnumf16,
                                "minnumf32" => KnownNames::StdIntrinsicsMinnumf32,
                                "minnumf64" => KnownNames::StdIntrinsicsMinnumf64,
                                "minnumf128" => KnownNames::StdIntrinsicsMinnumf128,
                                "mul_with_overflow" => KnownNames::StdIntrinsicsMulWithOverflow,
                                "nearbyintf16" => KnownNames::StdIntrinsicsNearbyintf16,
                                "nearbyintf32" => KnownNames::StdIntrinsicsNearbyintf32,
                                "nearbyintf64" => KnownNames::StdIntrinsicsNearbyintf64,
                                "nearbyintf128" => KnownNames::StdIntrinsicsNearbyintf128,
                                "needs_drop" => KnownNames::StdIntrinsicsNeedsDrop,
                                "offset" => KnownNames::StdIntrinsicsOffset,
                                "powf16" => KnownNames::StdIntrinsicsPowf16,
                                "powf32" => KnownNames::StdIntrinsicsPowf32,
                                "powf64" => KnownNames::StdIntrinsicsPowf64,
                                "powf128" => KnownNames::StdIntrinsicsPowf128,
                                "powif16" => KnownNames::StdIntrinsicsPowif16,
                                "powif32" => KnownNames::StdIntrinsicsPowif32,
                                "powif64" => KnownNames::StdIntrinsicsPowif64,
                                "powif128" => KnownNames::StdIntrinsicsPowif128,
                                "pref_align_of_val" => KnownNames::StdIntrinsicsPrefAlignOfVal,
                                "raw_eq" => KnownNames::StdIntrinsicsRawEq,
                                "rintf16" => KnownNames::StdIntrinsicsRintf16,
                                "rintf32" => KnownNames::StdIntrinsicsRintf32,
                                "rintf64" => KnownNames::StdIntrinsicsRintf64,
                                "rintf128" => KnownNames::StdIntrinsicsRintf128,
                                "roundf16" => KnownNames::StdIntrinsicsRoundf16,
                                "roundf32" => KnownNames::StdIntrinsicsRoundf32,
                                "roundf64" => KnownNames::StdIntrinsicsRoundf64,
                                "roundf128" => KnownNames::StdIntrinsicsRoundf128,
                                "roundevenf16" => KnownNames::StdIntrinsicsRevenf16,
                                "roundevenf32" => KnownNames::StdIntrinsicsRevenf32,
                                "roundevenf64" => KnownNames::StdIntrinsicsRevenf64,
                                "roundevenf128" => KnownNames::StdIntrinsicsRevenf128,
                                "sinf16" => KnownNames::StdIntrinsicsSinf16,
                                "sinf32" => KnownNames::StdIntrinsicsSinf32,
                                "sinf64" => KnownNames::StdIntrinsicsSinf64,
                                "sinf128" => KnownNames::StdIntrinsicsSinf64,
                                "size_of" => KnownNames::StdIntrinsicsSizeOf,
                                "size_of_val" => KnownNames::StdIntrinsicsSizeOfVal,
                                "sqrtf16" => KnownNames::StdIntrinsicsSqrtf16,
                                "sqrtf32" => KnownNames::StdIntrinsicsSqrtf32,
                                "sqrtf64" => KnownNames::StdIntrinsicsSqrtf64,
                                "sqrtf128" => KnownNames::StdIntrinsicsSqrtf128,
                                "three_way_compare" => KnownNames::StdIntrinsicsThreeWayCompare,
                                "transmute" => KnownNames::StdIntrinsicsTransmute,
                                "transmute_unchecked" => KnownNames::StdIntrinsicsTransmute,
                                "truncf16" => KnownNames::StdIntrinsicsTruncf16,
                                "truncf32" => KnownNames::StdIntrinsicsTruncf32,
                                "truncf64" => KnownNames::StdIntrinsicsTruncf64,
                                "truncf128" => KnownNames::StdIntrinsicsTruncf128,
                                "variant_count" => KnownNames::StdIntrinsicsVariantCount,
                                "write_bytes" => {
                                    if def_path_data_iter.next().is_some() {
                                        KnownNames::None
                                    } else {
                                        KnownNames::StdIntrinsicsWriteBytes
                                    }
                                }
                                _ => KnownNames::None,
                            })
                            .unwrap_or(KnownNames::None)
                    }
                }
                _ => {
                    if is_foreign_module(current_elem) {
                        get_known_name_for_instrinsics_foreign_namespace(def_path_data_iter)
                    } else {
                        KnownNames::None
                    }
                }
            }
        };

        let get_known_name_for_marker_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "PhantomData" => KnownNames::StdMarkerPhantomData,
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_mem_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "replace" => KnownNames::StdMemReplace,
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_ops_function_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "Fn" | "FnMut" | "FnOnce" => get_path_data_elem_name(def_path_data_iter.next())
                        .map(|n| match n.as_str() {
                            "call" => KnownNames::StdOpsFunctionFnCall,
                            "call_mut" => KnownNames::StdOpsFunctionFnMutCallMut,
                            "call_once" | "call_once_force" => {
                                KnownNames::StdOpsFunctionFnOnceCallOnce
                            }
                            _ => KnownNames::None,
                        })
                        .unwrap_or(KnownNames::None),
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_ops_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "function" => get_known_name_for_ops_function_namespace(def_path_data_iter),
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_panicking_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "assert_failed" => KnownNames::StdPanickingAssertFailed,
                    "begin_panic" | "panic" => KnownNames::StdPanickingBeginPanic,
                    "begin_panic_fmt" | "panic_fmt" => KnownNames::StdPanickingBeginPanicFmt,
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_ptr_mut_ptr_namespace =
            |mut def_path_data_iter: Iter<'_>| match path_data_elem_as_disambiguator(
                def_path_data_iter.next(),
            ) {
                Some(0) => get_path_data_elem_name(def_path_data_iter.next())
                    .map(|n| match n.as_str() {
                        "write_bytes" => {
                            if def_path_data_iter.next().is_some() {
                                KnownNames::None
                            } else {
                                KnownNames::StdIntrinsicsWriteBytes
                            }
                        }
                        _ => KnownNames::None,
                    })
                    .unwrap_or(KnownNames::None),
                _ => KnownNames::None,
            };

        let get_known_name_for_ptr_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "swap_nonoverlapping" => KnownNames::StdPtrSwapNonOverlapping,
                    "mut_ptr" => get_known_name_for_ptr_mut_ptr_namespace(def_path_data_iter),
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_slice_cmp_namespace =
            |mut def_path_data_iter: Iter<'_>| match path_data_elem_as_disambiguator(
                def_path_data_iter.next(),
            ) {
                Some(0) => get_path_data_elem_name(def_path_data_iter.next())
                    .map(|n| match n.as_str() {
                        "memcmp" => KnownNames::StdSliceCmpMemcmp,
                        _ => KnownNames::None,
                    })
                    .unwrap_or(KnownNames::None),
                _ => KnownNames::None,
            };

        let get_known_name_for_sync_once_namespace =
            |mut def_path_data_iter: Iter<'_>| match path_data_elem_as_disambiguator(
                def_path_data_iter.next(),
            ) {
                Some(2) => get_path_data_elem_name(def_path_data_iter.next())
                    .map(|n| match n.as_str() {
                        "call_once" | "call_once_force" => KnownNames::StdOpsFunctionFnOnceCallOnce,
                        _ => KnownNames::None,
                    })
                    .unwrap_or(KnownNames::None),
                _ => KnownNames::None,
            };

        let get_known_name_for_raw_vec_namespace =
            |mut def_path_data_iter: Iter<'_>| match path_data_elem_as_disambiguator(
                def_path_data_iter.next(),
            ) {
                Some(1) => get_path_data_elem_name(def_path_data_iter.next())
                    .map(|n| match n.as_str() {
                        "MIN_NON_ZERO_CAP" => KnownNames::AllocRawVecMinNonZeroCap,
                        _ => KnownNames::None,
                    })
                    .unwrap_or(KnownNames::None),
                _ => KnownNames::None,
            };

        let get_known_name_for_slice_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "cmp" => get_known_name_for_slice_cmp_namespace(def_path_data_iter),
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        //get_known_name_for_sync_namespace
        let get_known_name_for_sync_namespace = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "once" => get_known_name_for_sync_once_namespace(def_path_data_iter),
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let get_known_name_for_known_crate = |mut def_path_data_iter: Iter<'_>| {
            get_path_data_elem_name(def_path_data_iter.next())
                .map(|n| match n.as_str() {
                    "alloc" => get_known_name_for_alloc_namespace(def_path_data_iter),
                    "clone" => get_known_name_for_clone_namespace(def_path_data_iter),
                    "future" => get_known_name_for_future_namespace(def_path_data_iter),
                    "intrinsics" => get_known_name_for_intrinsics_namespace(def_path_data_iter),
                    "marker" => get_known_name_for_marker_namespace(def_path_data_iter),
                    "mem" => get_known_name_for_mem_namespace(def_path_data_iter),
                    "ops" => get_known_name_for_ops_namespace(def_path_data_iter),
                    "panicking" => get_known_name_for_panicking_namespace(def_path_data_iter),
                    "ptr" => get_known_name_for_ptr_namespace(def_path_data_iter),
                    "mirai_abstract_value" => KnownNames::MiraiAbstractValue,
                    "mirai_add_tag" => KnownNames::MiraiAddTag,
                    "mirai_assume" => KnownNames::MiraiAssume,
                    "mirai_assume_preconditions" => KnownNames::MiraiAssumePreconditions,
                    "mirai_does_not_have_tag" => KnownNames::MiraiDoesNotHaveTag,
                    "mirai_get_model_field" => KnownNames::MiraiGetModelField,
                    "mirai_has_tag" => KnownNames::MiraiHasTag,
                    "mirai_postcondition" => KnownNames::MiraiPostcondition,
                    "mirai_precondition_start" => KnownNames::MiraiPreconditionStart,
                    "mirai_precondition" => KnownNames::MiraiPrecondition,
                    "mirai_result" => KnownNames::MiraiResult,
                    "mirai_set_model_field" => KnownNames::MiraiSetModelField,
                    "mirai_verify" => KnownNames::MiraiVerify,
                    "raw_vec" => get_known_name_for_raw_vec_namespace(def_path_data_iter),
                    "rt" => get_known_name_for_panicking_namespace(def_path_data_iter),
                    "slice" => get_known_name_for_slice_namespace(def_path_data_iter),
                    "sync" => get_known_name_for_sync_namespace(def_path_data_iter),
                    _ => KnownNames::None,
                })
                .unwrap_or(KnownNames::None)
        };

        let crate_name = tcx.crate_name(def_id.krate);
        match crate_name.as_str() {
            "alloc" | "core" | "mirai_annotations" | "std" => {
                get_known_name_for_known_crate(def_path_data_iter)
            }
            _ => KnownNames::None,
        }
    }
}
