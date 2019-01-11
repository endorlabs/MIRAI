// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
use abstract_domains::{self, AbstractDomains};
use syntax_pos::Span;

/// Mirai is an abstract interpreter and thus produces abstract values.
/// In general, an abstract value is a value that is not fully known.
/// For example, we may know that it is a number between 1 and 10, but not
/// which particular number.
///
/// When we do know everything about a value, it is concrete rather than
/// abstract, but is convenient to just use this structure for concrete values
/// as well, since all operations can be uniform.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Hash)]
pub struct AbstractValue {
    /// An abstract value is the result of some expression.
    /// The source location of that expression is stored in provenance.
    /// When an expression is stored somewhere and then retrieved via an accessor expression, a new
    /// abstract value is created (via refinement using the current path condition) with a provenance
    /// that is the source location of accessor expression. If refinement results in an existing
    /// expression (i.e. one with a provenance of its own) then a copy expression is created with
    /// the existing expression as argument, so that both locations are tracked.
    #[serde(skip)]
    pub provenance: Span, //todo: perhaps this should be a list of spans
    /// Various approximations of the actual value.
    /// See https://github.com/facebookexperimental/MIRAI/blob/master/documentation/AbstractValues.md.
    pub value: AbstractDomains,
}

/// An abstract value that can be used as the value for an operation that has no normal result.
pub const BOTTOM: AbstractValue = AbstractValue {
    provenance: syntax_pos::DUMMY_SP,
    value: abstract_domains::BOTTOM,
};

/// An abstract value to use when nothing is known about the value. All possible concrete values
/// are members of the concrete set of values corresponding to this abstract value.
pub const TOP: AbstractValue = AbstractValue {
    provenance: syntax_pos::DUMMY_SP,
    value: abstract_domains::TOP,
};

impl AbstractValue {
    /// True if the set of concrete values that correspond to this abstract value is empty.
    pub fn is_bottom(&self) -> bool {
        self.value.is_bottom()
    }

    /// True if all possible concrete values are elements of the set corresponding to this abstract value.
    pub fn is_top(&self) -> bool {
        self.value.is_top()
    }

    /// Returns an abstract value whose corresponding set of concrete values include all of the values
    /// corresponding to self and other.
    /// In a context where the join condition is known to be true, the result can be refined to be
    /// just self, correspondingly if it is known to be false, the result can be refined to be just other.
    pub fn join(&self, other: &AbstractValue, join_condition: &AbstractValue) -> AbstractValue {
        AbstractValue {
            provenance: syntax_pos::DUMMY_SP,
            value: self.value.join(&other.value, &join_condition.value),
        }
    }

    /// True if all of the concrete values that correspond to self also correspond to other.
    pub fn subset(&self, other: &AbstractValue) -> bool {
        self.value.subset(&other.value)
    }

    /// Returns an abstract value whose corresponding set of concrete values include all of the values
    /// corresponding to self and other. The set of values may be less precise (more inclusive) than
    /// the set returned by join. The chief requirement is that a small number of widen calls
    /// deterministically lead to Top.
    pub fn widen(&self, other: &AbstractValue, join_condition: &AbstractValue) -> AbstractValue {
        AbstractValue {
            provenance: syntax_pos::DUMMY_SP,
            value: self.value.widen(&other.value, &join_condition.value),
        }
    }
}

/// The name of a function or method, sufficiently qualified so that it uniquely identifies it
/// among all functions and methods defined in the project corresponding to a summary store.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Name {
    /// A root name in the current name space. Typically the name of a crate, used module, or
    /// used function or struct/trait/enum/type.
    Root { identifier: String },

    /// A name that selects a named component (specified by selector) of the structure named by the
    /// qualifier.
    QualifiedName {
        qualifier: Box<Name>,
        selector: String,
    },
}

/// A path represents a left hand side expression.
/// When the actual expression is evaluated at runtime it will resolve to a particular memory
/// location. During analysis it is used to keep track of state changes.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Path {
    /// 0 is the return value temporary
    /// [1 ... arg_count] are the parameters
    /// [arg_count ... ] are the local variables and compiler temporaries
    LocalVariable { ordinal: usize },

    /// The name is a summary cache key string.
    StaticVariable { name: String },

    /// The ordinal is an index into a crate level constant table.
    PromotedConstant { ordinal: usize },

    /// The qualifier denotes some reference, struct, or collection.
    /// The selector denotes a de-referenced item, field, or element, or slice.
    QualifiedPath {
        qualifier: Box<Path>,
        selector: Box<PathSelector>,
    },
}

/// The selector denotes a de-referenced item, field, or element, or slice.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PathSelector {
    /// Given a path that denotes a reference, select the thing the reference points to.
    Deref,

    /// Select the struct field with the given index.
    Field(usize),

    /// Select the collection element with the index stored in the local with the given ordinal.
    Index(usize),

    /// These indices are generated by slice patterns. Easiest to explain
    /// by example:
    ///
    /// ```
    /// [X, _, .._, _, _] => { offset: 0, min_length: 4, from_end: false },
    /// [_, X, .._, _, _] => { offset: 1, min_length: 4, from_end: false },
    /// [_, _, .._, X, _] => { offset: 2, min_length: 4, from_end: true },
    /// [_, _, .._, _, X] => { offset: 1, min_length: 4, from_end: true },
    /// ```
    ConstantIndex {
        /// index or -index (in Python terms), depending on from_end
        offset: u32,
        /// thing being indexed must be at least this long
        min_length: u32,
        /// counting backwards from end?
        from_end: bool,
    },

    /// These indices are generated by slice patterns.
    ///
    /// slice[from:-to] in Python terms.
    Subslice { from: u32, to: u32 },

    /// "Downcast" to a variant of an ADT. Currently, MIR only introduces
    /// this for ADTs with more than one variant. The value is the ordinal of the variant.
    Downcast(usize),
}
