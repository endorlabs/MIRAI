// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crate::abstract_value::{AbstractValue, AbstractValueTrait};
use crate::constant_domain::ConstantDomain;
use crate::known_names::KnownNames;
use crate::path::Path;
use crate::tag_domain::Tag;

use log_derive::*;
use mirai_annotations::*;
use rustc_middle::ty::{FloatTy, IntTy, Ty, TyCtxt, TyKind, TypeAndMut, UintTy};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter, Result};
use std::rc::Rc;

/// Closely based on the expressions found in MIR.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Expression {
    /// An expression that represents any possible value
    Top,

    /// An expression that represents an impossible value, such as the value returned by function
    /// that always panics.
    Bottom,

    /// An expression that is the sum of left and right. +
    Add {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is false if left + right overflows.
    AddOverflows {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
        // The type of the result of left + right.
        result_type: ExpressionType,
    },

    /// An expression that is true if both left and right are true. &&
    And {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is the bitwise and of left and right. &
    BitAnd {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that results in the bitwise complement of operand. ! integer
    BitNot {
        // The value of the operand.
        operand: Rc<AbstractValue>,
        // The kind of integer. Needed to specify the number of bits to flip.
        result_type: ExpressionType,
    },

    /// An expression that is the bitwise or of left and right. |
    BitOr {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is the bitwise xor of left and right. ^
    BitXor {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is the operand cast to the target_type. as
    Cast {
        // The value of the operand.
        operand: Rc<AbstractValue>,
        // The type the operand is being cast to.
        target_type: ExpressionType,
    },

    /// An expression that is a compile time constant value, such as a numeric literal or a function.
    CompileTimeConstant(ConstantDomain),

    /// An expression that is either consequent or alternate, depending on the value of condition.
    ConditionalExpression {
        // A condition that results in a Boolean value
        condition: Rc<AbstractValue>,
        // The value of this expression if join_condition is true.
        consequent: Rc<AbstractValue>,
        // The value of this expression if join_condition is false.
        alternate: Rc<AbstractValue>,
    },

    /// An expression that is the left value divided by the right value. /
    Div {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is true if left and right are equal. ==
    Equals {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is true if left is greater than or equal to right. >=
    GreaterOrEqual {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is true if left is greater than right. >
    GreaterThan {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that represents a block of memory allocated from the heap.
    /// The value of expression is an ordinal used to distinguish this allocation from
    /// other allocations. Because this is static analysis, a given allocation site will
    /// always result in the same ordinal. The implication of this is that there will be
    /// some loss of precision when heap blocks are allocated inside loops.
    HeapBlock {
        // A unique ordinal that distinguishes this allocation from other allocations.
        // Not an actual memory address.
        abstract_address: usize,
        // True if the allocator zeroed out this heap memory block.
        is_zeroed: bool,
    },

    /// An expression that represents the byte size, alignment and liveness of a heap allocated memory block.
    /// It is a separate expression because it can be constructed and propagated independently
    /// from the heap block resulting from an allocation construct/call.
    HeapBlockLayout {
        // The number of bytes allocated to the memory block.
        length: Rc<AbstractValue>,
        // The byte alignment of the memory block.
        alignment: Rc<AbstractValue>,
        // The intrinsic call that created this layout.
        source: LayoutSource,
    },

    /// The value of a parameter, or a subcomponent of structured parameter, as it was at function
    /// entry and before any assignments to the parameter. When transferred into a calling context
    /// via a summary, this value must be refined with the environment as it was at the start of the call.
    InitialParameterValue {
        // Must be rooted in a parameter.
        path: Rc<Path>,
        var_type: ExpressionType,
    },

    /// An expression that calls the specified intrinsic binary function with given arguments. left.name(right)
    IntrinsicBinary {
        left: Rc<AbstractValue>,
        right: Rc<AbstractValue>,
        name: KnownNames,
    },

    /// An expression that calls the specified intrinsic bit vector function with given argument. (operand as u(8|16|64|128)).name()
    IntrinsicBitVectorUnary {
        operand: Rc<AbstractValue>,
        bit_length: u8,
        name: KnownNames,
    },

    /// An expression that calls the specified intrinsic floating point unary function with given argument. operand.name()
    IntrinsicFloatingPointUnary {
        operand: Rc<AbstractValue>,
        name: KnownNames,
    },

    /// Either left or right without a condition to tell them apart.
    /// This can happen when there is loss of precision because of a loop fixed point computation.
    /// For instance inside a loop body after the second fixed point iteration, a counter may have
    /// either its initial value or the value computed in the first loop body iteration and we
    /// don't have a way to tell which value it is.
    Join {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is true if left is less than or equal to right. <=
    LessOrEqual {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is true if left is less than right. <
    LessThan {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is true if the operand is false. ! bool
    LogicalNot { operand: Rc<AbstractValue> },

    /// An expression which represents the result of comparing the left operand with
    /// the right operand, according to the rules of of memcmp in unix.
    Memcmp {
        left: Rc<AbstractValue>,
        right: Rc<AbstractValue>,
        length: Rc<AbstractValue>,
    },

    /// An expression that is left multiplied by right. *
    Mul {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is false if left multiplied by right overflows.
    MulOverflows {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
        // The type of the result of left * right.
        result_type: ExpressionType,
    },

    /// An expression that is true if left and right are not equal. !=
    Ne {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is the arithmetic negation of its parameter. -x
    Neg { operand: Rc<AbstractValue> },

    /// An expression that is true if either one of left or right are true. ||
    Or {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is left offset with right. ptr.offset
    Offset {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// The corresponding concrete value is the runtime address of location identified by the path.
    Reference(Rc<Path>),

    /// An expression that is the remainder of left divided by right. %
    Rem {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is the result of left shifted left by right bits. <<
    Shl {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is false if left shifted left by right bits would shift way all bits. <<
    ShlOverflows {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
        // The type of the result of left << right.
        result_type: ExpressionType,
    },

    /// An expression that is the result of left shifted right by right bits. >>
    Shr {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is false if left shifted right by right bits would shift way all bits. >>
    ShrOverflows {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
        // The type of the result of left >> right.
        result_type: ExpressionType,
    },

    /// An expression that is the right subtracted from left. -
    Sub {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
    },

    /// An expression that is false if right subtracted from left overflows. -
    SubOverflows {
        // The value of the left operand.
        left: Rc<AbstractValue>,
        // The value of the right operand.
        right: Rc<AbstractValue>,
        // The type of the result of left - right.
        result_type: ExpressionType,
    },

    /// An expression that selects a value based on the value of a discriminator expression. match
    Switch {
        // The value that is switched on.
        discriminator: Rc<AbstractValue>,
        // (switch_case_val, result_in_this_case)
        cases: Vec<(Rc<AbstractValue>, Rc<AbstractValue>)>,
        // The result value when the discriminator value does not match any switch case value.
        default: Rc<AbstractValue>,
    },

    /// An expression that is the operand attached with the tag.
    TaggedExpression {
        operand: Rc<AbstractValue>,
        tag: Tag,
    },

    /// An expression that interprets the bits of operand as being a value of the target type
    Transmute {
        operand: Rc<AbstractValue>,
        target_type: ExpressionType,
    },

    /// An expression that represents the result of calling an unknown function.
    /// Typically this will be a trait method, function parameter, or an intrinsic/foreign function.
    /// In the case of a trait method or a function parameter, the caller might be able to refine
    /// this expression into a normal call.
    UninterpretedCall {
        // The unknown function to call. This can be just a reference to a location rooted in a parameter
        // (in which case it can be specialized) or it can be a constant, in which case it may be
        // possible to map it to a known function during call site refinement.
        callee: Rc<AbstractValue>,
        // The argument values used for the call. These will be refined if the callee can be resolved
        // at the call site.
        arguments: Vec<Rc<AbstractValue>>,
        /// The type of the call result.
        result_type: ExpressionType,
        /// The path where the result is stored
        path: Rc<Path>,
    },

    /// Like a variable, but during refinement the default value is used
    /// when the qualifier becomes a known value without a defined value for the model field.
    UnknownModelField {
        /// Must be a qualified path with a model field selector.
        path: Rc<Path>,
        default: Rc<AbstractValue>,
    },

    /// Like unknown model fields, this variant records an unknown result of `has_tag!` or
    /// `does_not_have_tag!` on `operand`.
    UnknownTagCheck {
        operand: Rc<AbstractValue>,
        tag: Tag,
        checking_presence: bool,
    },

    /// Like unknown model fields, this variant records an unknown tag field of a non-scalar value.
    UnknownTagField {
        /// Must be a qualified path with a tag field selector.
        path: Rc<Path>,
    },

    /// The unknown value of a place in memory.
    /// This is distinct from Top in that we known something: the place and the type.
    /// This is a useful distinction because it allows us to simplify some expressions
    /// like x == x. The type is needed to prevent this particular optimization if
    /// the variable is a floating point number that could be NaN.
    Variable {
        path: Rc<Path>,
        var_type: ExpressionType,
    },

    /// The partly known value of a place in memory.
    /// The value in operand will be the join of several expressions that all reference
    /// the path of this value. This models a variable that is assigned to from inside a loop
    /// body.
    WidenedJoin {
        /// The path of a location where an indeterminate number of flows join together.
        /// It is attached to a widened join at a join a point and then preserved in derived
        /// values. If such a derived value returns to the join point as its new value, the
        /// original value can be used so that the fix point loop converges.
        /// The path does not provide identity and two widened joins with the same path
        /// are not necessary the same value.
        path: Rc<Path>,
        /// The join of some of the flows to come together at this path.
        /// The first few iterations do joins. Once widening happens, further iterations
        /// all result in the same widened value.
        operand: Rc<AbstractValue>,
    },
}

/// Used by Expression::AbstractHeapBlockLayout
#[derive(Serialize, Deserialize, Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum LayoutSource {
    Alloc,
    DeAlloc,
    ReAlloc,
}

impl Debug for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Expression::Top => f.write_str("TOP"),
            Expression::Bottom => f.write_str("BOTTOM"),
            Expression::Add { left, right } => {
                f.write_fmt(format_args!("({:?}) + ({:?})", left, right))
            }
            Expression::AddOverflows { left, right, .. } => {
                f.write_fmt(format_args!("overflows(({:?}) + ({:?}))", left, right))
            }
            Expression::And { left, right } => {
                f.write_fmt(format_args!("({:?}) && ({:?})", left, right))
            }
            Expression::BitAnd { left, right } => {
                f.write_fmt(format_args!("({:?}) & ({:?})", left, right))
            }
            Expression::BitNot { operand, .. } => f.write_fmt(format_args!("~({:?})", operand)),
            Expression::BitOr { left, right } => {
                f.write_fmt(format_args!("({:?}) | ({:?})", left, right))
            }
            Expression::BitXor { left, right } => {
                f.write_fmt(format_args!("({:?}) ^ ({:?})", left, right))
            }
            Expression::Cast {
                operand,
                target_type,
            } => f.write_fmt(format_args!("({:?}) as {:?}", operand, target_type)),
            Expression::CompileTimeConstant(c) => c.fmt(f),
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => f.write_fmt(format_args!(
                "if {:?} {{ {:?} }} else {{ {:?} }}",
                condition, consequent, alternate
            )),
            Expression::Div { left, right } => {
                f.write_fmt(format_args!("({:?}) / ({:?})", left, right))
            }
            Expression::Equals { left, right } => {
                f.write_fmt(format_args!("({:?}) == ({:?})", left, right))
            }
            Expression::GreaterOrEqual { left, right } => {
                f.write_fmt(format_args!("({:?}) >= ({:?})", left, right))
            }
            Expression::GreaterThan { left, right } => {
                f.write_fmt(format_args!("({:?}) > ({:?})", left, right))
            }
            Expression::HeapBlock {
                abstract_address: address,
                is_zeroed,
            } => f.write_fmt(format_args!(
                "{}heap_{}",
                if *is_zeroed { "zeroed_" } else { "" },
                *address,
            )),
            Expression::HeapBlockLayout {
                length,
                alignment,
                source,
            } => f.write_fmt(format_args!(
                "layout(length: {:?}; alignment: {:?}; source: {:?})",
                length, alignment, source
            )),
            Expression::IntrinsicBinary { left, right, name } => {
                f.write_fmt(format_args!("({:?}).{:?}({:?})", left, name, right))
            }
            Expression::IntrinsicBitVectorUnary {
                operand,
                bit_length,
                name,
            } => f.write_fmt(format_args!(
                "({:?} as (i|u){}).{:?}()",
                operand, bit_length, name
            )),
            Expression::IntrinsicFloatingPointUnary { operand, name } => {
                f.write_fmt(format_args!("({:?}).{:?}()", operand, name))
            }
            Expression::Join { left, right } => {
                f.write_fmt(format_args!("({:?}) join ({:?})", left, right))
            }
            Expression::LessOrEqual { left, right } => {
                f.write_fmt(format_args!("({:?}) <= ({:?})", left, right))
            }
            Expression::LessThan { left, right } => {
                f.write_fmt(format_args!("({:?}) < ({:?})", left, right))
            }
            Expression::LogicalNot { operand } => f.write_fmt(format_args!("!({:?})", operand)),
            Expression::Memcmp {
                left,
                right,
                length,
            } => f.write_fmt(format_args!(
                "memcmp({:?}, {:?}, {:?})",
                left, right, length
            )),
            Expression::Mul { left, right } => {
                f.write_fmt(format_args!("({:?}) * ({:?})", left, right))
            }
            Expression::MulOverflows { left, right, .. } => {
                f.write_fmt(format_args!("overflows(({:?}) * ({:?}))", left, right))
            }
            Expression::Ne { left, right } => {
                f.write_fmt(format_args!("({:?}) != ({:?})", left, right))
            }
            Expression::Neg { operand } => f.write_fmt(format_args!("-({:?})", operand)),
            Expression::Or { left, right } => {
                f.write_fmt(format_args!("({:?}) || ({:?})", left, right))
            }
            Expression::Offset { left, right } => {
                f.write_fmt(format_args!("&({:?})[{:?}]", left, right))
            }
            Expression::Reference(path) => f.write_fmt(format_args!("&({:?})", path)),
            Expression::InitialParameterValue { path, var_type } => {
                f.write_fmt(format_args!("old({:?}): {:?}", path, var_type))
            }
            Expression::Rem { left, right } => {
                f.write_fmt(format_args!("({:?}) % ({:?})", left, right))
            }
            Expression::Shl { left, right } => {
                f.write_fmt(format_args!("({:?}) << ({:?})", left, right))
            }
            Expression::ShlOverflows { left, right, .. } => {
                f.write_fmt(format_args!("overflows(({:?}) << ({:?}))", left, right))
            }
            Expression::Shr { left, right, .. } => {
                f.write_fmt(format_args!("({:?}) >> ({:?})", left, right))
            }
            Expression::ShrOverflows { left, right, .. } => {
                f.write_fmt(format_args!("overflows(({:?}) >> ({:?}))", left, right))
            }
            Expression::Sub { left, right } => {
                f.write_fmt(format_args!("({:?}) - ({:?})", left, right))
            }
            Expression::SubOverflows { left, right, .. } => {
                f.write_fmt(format_args!("overflows(({:?}) - ({:?}))", left, right))
            }
            Expression::Switch {
                discriminator,
                cases,
                default,
            } => {
                f.write_fmt(format_args!("switch {:?} {{", discriminator))?;
                for (switch_case, value) in cases {
                    f.write_fmt(format_args!(" {:?} => {:?},", switch_case, value))?;
                }
                f.write_fmt(format_args!(" _ => {:?}", default))?;
                f.write_str(" }")
            }
            Expression::TaggedExpression { operand, tag } => {
                f.write_fmt(format_args!("{:?} tagged with {:?}", operand, tag))
            }
            Expression::Transmute {
                operand,
                target_type,
            } => f.write_fmt(format_args!("transmute({:?}, {:?})", operand, target_type)),
            Expression::UninterpretedCall {
                callee,
                arguments,
                result_type,
                path,
            } => {
                let _callee_txt = format!("{:?}", callee);
                f.write_fmt(format_args!(
                    "uninterpreted_call({:?}({:?}) -> {:?}) at {:?}",
                    callee, arguments, result_type, path
                ))
            }
            Expression::UnknownModelField { path, default } => {
                f.write_fmt(format_args!("({:?}).default(({:?})", path, default))
            }
            Expression::UnknownTagCheck {
                operand,
                tag,
                checking_presence,
            } => f.write_fmt(format_args!(
                "({:?}).check_tag({:?}, {})",
                operand, tag, checking_presence
            )),
            Expression::UnknownTagField { path } => path.fmt(f),
            Expression::Variable { path, var_type } => {
                f.write_fmt(format_args!("{:?}: {:?}", path, var_type))
            }
            Expression::WidenedJoin { path, operand } => {
                if operand.expression_size > 100 {
                    f.write_fmt(format_args!("widened(..) at {:?}", path))
                } else {
                    f.write_fmt(format_args!("widened({:?}) at {:?}", operand, path))
                }
            }
        }
    }
}

impl Expression {
    /// Returns true if any part of the expression is a local variable.
    /// Use this to weed out inferred preconditions that cannot be satisfied by the caller.
    #[logfn_inputs(TRACE)]
    pub fn contains_local_variable(&self, is_post_condition: bool) -> bool {
        match &self {
            Expression::Add { left, right }
            | Expression::AddOverflows { left, right, .. }
            | Expression::And { left, right }
            | Expression::BitAnd { left, right }
            | Expression::BitOr { left, right }
            | Expression::BitXor { left, right }
            | Expression::Div { left, right }
            | Expression::Equals { left, right }
            | Expression::GreaterOrEqual { left, right }
            | Expression::GreaterThan { left, right }
            | Expression::IntrinsicBinary { left, right, .. }
            | Expression::LessOrEqual { left, right }
            | Expression::LessThan { left, right }
            | Expression::Mul { left, right }
            | Expression::MulOverflows { left, right, .. }
            | Expression::Ne { left, right }
            | Expression::Offset { left, right }
            | Expression::Or { left, right }
            | Expression::Rem { left, right }
            | Expression::Shl { left, right }
            | Expression::ShlOverflows { left, right, .. }
            | Expression::Shr { left, right, .. }
            | Expression::ShrOverflows { left, right, .. }
            | Expression::Sub { left, right }
            | Expression::SubOverflows { left, right, .. } => {
                left.expression.contains_local_variable(is_post_condition)
                    || right.expression.contains_local_variable(is_post_condition)
            }
            Expression::BitNot { operand, .. }
            | Expression::Cast { operand, .. }
            | Expression::IntrinsicBitVectorUnary { operand, .. }
            | Expression::IntrinsicFloatingPointUnary { operand, .. }
            | Expression::TaggedExpression { operand, .. }
            | Expression::Transmute { operand, .. } => operand
                .expression
                .contains_local_variable(is_post_condition),
            Expression::Bottom => true,

            Expression::CompileTimeConstant(..) => false,
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                condition
                    .expression
                    .contains_local_variable(is_post_condition)
                    || consequent
                        .expression
                        .contains_local_variable(is_post_condition)
                    || alternate
                        .expression
                        .contains_local_variable(is_post_condition)
            }
            Expression::HeapBlock { .. } => true,
            Expression::HeapBlockLayout {
                length, alignment, ..
            } => {
                length.expression.contains_local_variable(is_post_condition)
                    || alignment
                        .expression
                        .contains_local_variable(is_post_condition)
            }
            Expression::InitialParameterValue { path, .. } => {
                path.contains_local_variable(is_post_condition)
            }
            Expression::Join { left, right, .. } => {
                left.expression.contains_local_variable(is_post_condition)
                    || right.expression.contains_local_variable(is_post_condition)
            }
            Expression::Memcmp {
                left,
                right,
                length,
            } => {
                left.expression.contains_local_variable(is_post_condition)
                    || right.expression.contains_local_variable(is_post_condition)
                    || length.expression.contains_local_variable(is_post_condition)
            }
            Expression::Neg { operand }
            | Expression::LogicalNot { operand }
            | Expression::UnknownTagCheck { operand, .. } => operand
                .expression
                .contains_local_variable(is_post_condition),
            Expression::Reference(path) => path.contains_local_variable(is_post_condition),
            Expression::Switch {
                discriminator,
                cases,
                default,
            } => {
                discriminator
                    .expression
                    .contains_local_variable(is_post_condition)
                    || default
                        .expression
                        .contains_local_variable(is_post_condition)
                    || cases
                        .iter()
                        .any(|(_, v)| v.expression.contains_local_variable(is_post_condition))
            }
            Expression::Top => true,
            Expression::UninterpretedCall { .. } => true,
            Expression::UnknownModelField { path, default } => {
                path.contains_local_variable(is_post_condition)
                    || default
                        .expression
                        .contains_local_variable(is_post_condition)
            }
            Expression::UnknownTagField { path } | Expression::Variable { path, .. } => {
                path.contains_local_variable(is_post_condition)
            }
            Expression::WidenedJoin { operand, .. } => operand
                .expression
                .contains_local_variable(is_post_condition),
        }
    }

    /// Returns true if any part of the expression is TOP.
    #[logfn_inputs(TRACE)]
    pub fn contains_top(&self) -> bool {
        match &self {
            Expression::Add { left, right }
            | Expression::AddOverflows { left, right, .. }
            | Expression::And { left, right }
            | Expression::BitAnd { left, right }
            | Expression::BitOr { left, right }
            | Expression::BitXor { left, right }
            | Expression::Div { left, right }
            | Expression::Equals { left, right }
            | Expression::GreaterOrEqual { left, right }
            | Expression::GreaterThan { left, right }
            | Expression::IntrinsicBinary { left, right, .. }
            | Expression::LessOrEqual { left, right }
            | Expression::LessThan { left, right }
            | Expression::Mul { left, right }
            | Expression::MulOverflows { left, right, .. }
            | Expression::Ne { left, right }
            | Expression::Offset { left, right }
            | Expression::Or { left, right }
            | Expression::Rem { left, right }
            | Expression::Shl { left, right }
            | Expression::ShlOverflows { left, right, .. }
            | Expression::Shr { left, right, .. }
            | Expression::ShrOverflows { left, right, .. }
            | Expression::Sub { left, right }
            | Expression::SubOverflows { left, right, .. } => {
                left.expression.contains_top() || right.expression.contains_top()
            }
            Expression::BitNot { operand, .. }
            | Expression::Cast { operand, .. }
            | Expression::IntrinsicBitVectorUnary { operand, .. }
            | Expression::IntrinsicFloatingPointUnary { operand, .. }
            | Expression::TaggedExpression { operand, .. }
            | Expression::Transmute { operand, .. } => operand.expression.contains_top(),
            Expression::Bottom => false,

            Expression::CompileTimeConstant(..) => false,
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                condition.expression.contains_top()
                    || consequent.expression.contains_top()
                    || alternate.expression.contains_top()
            }
            Expression::HeapBlock { .. } => false,
            Expression::HeapBlockLayout {
                length, alignment, ..
            } => length.expression.contains_top() || alignment.expression.contains_top(),
            Expression::InitialParameterValue { path, .. } => path.contains_top(),
            Expression::Join { left, right, .. } => {
                left.expression.contains_top() || right.expression.contains_top()
            }
            Expression::Memcmp {
                left,
                right,
                length,
            } => {
                left.expression.contains_top()
                    || right.expression.contains_top()
                    || length.expression.contains_top()
            }
            Expression::Neg { operand }
            | Expression::LogicalNot { operand }
            | Expression::UnknownTagCheck { operand, .. } => operand.expression.contains_top(),
            Expression::Reference(path) => path.contains_top(),
            Expression::Switch {
                discriminator,
                cases,
                default,
            } => {
                discriminator.expression.contains_top()
                    || default.expression.contains_top()
                    || cases.iter().any(|(_, v)| v.expression.contains_top())
            }
            Expression::Top => true,
            Expression::UninterpretedCall { .. } => true,
            Expression::UnknownModelField { path, default } => {
                path.contains_top() || default.expression.contains_top()
            }
            Expression::UnknownTagField { path } | Expression::Variable { path, .. } => {
                path.contains_top()
            }
            Expression::WidenedJoin { operand, .. } => operand.expression.contains_top(),
        }
    }

    /// Returns a value from the enum `TagPropagation` which reflects the expression kind.
    /// If the tag propagation behavior for the expression is not controllable, e.g., for
    /// control-flow expressions such as Conditional or Switch, returns None.
    #[logfn_inputs(TRACE)]
    pub fn get_tag_propagation(&self) -> Option<TagPropagation> {
        match self {
            Expression::Top => None,
            Expression::Bottom => None,
            Expression::Add { .. } => Some(TagPropagation::Add),
            Expression::AddOverflows { .. } => Some(TagPropagation::AddOverflows),
            Expression::And { .. } => Some(TagPropagation::And),
            Expression::BitAnd { .. } => Some(TagPropagation::BitAnd),
            Expression::BitNot { .. } => Some(TagPropagation::BitNot),
            Expression::BitOr { .. } => Some(TagPropagation::BitOr),
            Expression::BitXor { .. } => Some(TagPropagation::BitXor),
            Expression::Cast { .. } => Some(TagPropagation::Cast),
            Expression::CompileTimeConstant { .. } => None,
            Expression::ConditionalExpression { .. } => None,
            Expression::Div { .. } => Some(TagPropagation::Div),
            Expression::Equals { .. } => Some(TagPropagation::Equals),
            Expression::GreaterOrEqual { .. } => Some(TagPropagation::GreaterOrEqual),
            Expression::GreaterThan { .. } => Some(TagPropagation::GreaterThan),
            Expression::HeapBlock { .. } => None,
            Expression::HeapBlockLayout { .. } => None,
            Expression::IntrinsicBinary { .. } => Some(TagPropagation::IntrinsicBinary),
            Expression::IntrinsicBitVectorUnary { .. } => {
                Some(TagPropagation::IntrinsicBitVectorUnary)
            }
            Expression::IntrinsicFloatingPointUnary { .. } => {
                Some(TagPropagation::IntrinsicFloatingPointUnary)
            }
            Expression::Join { .. } => None,
            Expression::LessOrEqual { .. } => Some(TagPropagation::LessOrEqual),
            Expression::LessThan { .. } => Some(TagPropagation::LessThan),
            Expression::LogicalNot { .. } => Some(TagPropagation::LogicalNot),
            Expression::Memcmp { .. } => Some(TagPropagation::Memcmp),
            Expression::Mul { .. } => Some(TagPropagation::Mul),
            Expression::MulOverflows { .. } => Some(TagPropagation::MulOverflows),
            Expression::Ne { .. } => Some(TagPropagation::Ne),
            Expression::Neg { .. } => Some(TagPropagation::Neg),
            Expression::Or { .. } => Some(TagPropagation::Or),
            Expression::Offset { .. } => Some(TagPropagation::Offset),
            Expression::Reference { .. } => None,
            Expression::InitialParameterValue { .. } => None,
            Expression::Rem { .. } => Some(TagPropagation::Rem),
            Expression::Shl { .. } => Some(TagPropagation::Shl),
            Expression::ShlOverflows { .. } => Some(TagPropagation::ShlOverflows),
            Expression::Shr { .. } => Some(TagPropagation::Shr),
            Expression::ShrOverflows { .. } => Some(TagPropagation::ShrOverflows),
            Expression::Sub { .. } => Some(TagPropagation::Sub),
            Expression::SubOverflows { .. } => Some(TagPropagation::SubOverflows),
            Expression::Switch { .. } => None,
            Expression::TaggedExpression { .. } => None,
            Expression::Transmute { .. } => Some(TagPropagation::Transmute),
            Expression::UninterpretedCall { .. } => Some(TagPropagation::UninterpretedCall),
            Expression::UnknownModelField { .. } => None,
            Expression::UnknownTagCheck { .. } => None,
            Expression::UnknownTagField { .. } => None,
            Expression::Variable { .. } => None,
            Expression::WidenedJoin { .. } => None,
        }
    }

    /// Returns the type of value the expression should result in, if well formed.
    /// (both operands are of the same type for binary operators, conditional branches match).
    #[logfn_inputs(TRACE)]
    pub fn infer_type(&self) -> ExpressionType {
        use self::ExpressionType::*;
        match self {
            Expression::Top => NonPrimitive,
            Expression::Bottom => NonPrimitive,
            Expression::Add { left, right } => {
                let lt = left.expression.infer_type();
                if lt == ExpressionType::ThinPointer {
                    return lt;
                }
                right.expression.infer_type()
            }
            Expression::AddOverflows { .. } => Bool,
            Expression::And { .. } => Bool,
            Expression::BitAnd { left, .. } => left.expression.infer_type(),
            Expression::BitNot { result_type, .. } => *result_type,
            Expression::BitOr { left, .. } => left.expression.infer_type(),
            Expression::HeapBlock { .. } => NonPrimitive,
            Expression::HeapBlockLayout { .. } => NonPrimitive,
            Expression::IntrinsicBinary { left, name, .. } => match name {
                KnownNames::StdIntrinsicsCopysignf32
                | KnownNames::StdIntrinsicsMaxnumf32
                | KnownNames::StdIntrinsicsMinnumf32
                | KnownNames::StdIntrinsicsPowf32
                | KnownNames::StdIntrinsicsPowif32 => ExpressionType::F32,
                KnownNames::StdIntrinsicsCopysignf64
                | KnownNames::StdIntrinsicsMaxnumf64
                | KnownNames::StdIntrinsicsMinnumf64
                | KnownNames::StdIntrinsicsPowf64
                | KnownNames::StdIntrinsicsPowif64 => ExpressionType::F64,
                KnownNames::StdIntrinsicsFaddFast
                | KnownNames::StdIntrinsicsFdivFast
                | KnownNames::StdIntrinsicsFmulFast
                | KnownNames::StdIntrinsicsFremFast
                | KnownNames::StdIntrinsicsFsubFast => left.expression.infer_type(),
                _ => assume_unreachable!("invalid name {:?} for intrinsic binary", name),
            },
            Expression::IntrinsicBitVectorUnary {
                operand,
                name,
                bit_length,
            } => match name {
                KnownNames::StdIntrinsicsBitreverse | KnownNames::StdIntrinsicsBswap => {
                    operand.expression.infer_type()
                }
                KnownNames::StdIntrinsicsCtlz
                | KnownNames::StdIntrinsicsCtlzNonzero
                | KnownNames::StdIntrinsicsCtpop
                | KnownNames::StdIntrinsicsCttz
                | KnownNames::StdIntrinsicsCttzNonzero => {
                    // u(8|16|32|64|128)
                    match bit_length {
                        8 => ExpressionType::U8,
                        16 => ExpressionType::U16,
                        32 => ExpressionType::U32,
                        64 => ExpressionType::U64,
                        128 => ExpressionType::U128,
                        _ => unreachable!("the type checker should not allow this"),
                    }
                }
                _ => assume_unreachable!("invalid name {:?} for intrinsic bit vector unary", name),
            },
            Expression::IntrinsicFloatingPointUnary { name, .. } => match name {
                KnownNames::StdIntrinsicsCeilf32
                | KnownNames::StdIntrinsicsCosf32
                | KnownNames::StdIntrinsicsFloorf32
                | KnownNames::StdIntrinsicsExp2f32
                | KnownNames::StdIntrinsicsExpf32
                | KnownNames::StdIntrinsicsFabsf32
                | KnownNames::StdIntrinsicsLog10f32
                | KnownNames::StdIntrinsicsLog2f32
                | KnownNames::StdIntrinsicsLogf32
                | KnownNames::StdIntrinsicsNearbyintf32
                | KnownNames::StdIntrinsicsRintf32
                | KnownNames::StdIntrinsicsRoundf32
                | KnownNames::StdIntrinsicsSinf32
                | KnownNames::StdIntrinsicsSqrtf32
                | KnownNames::StdIntrinsicsTruncf32 => ExpressionType::F32,
                KnownNames::StdIntrinsicsCeilf64
                | KnownNames::StdIntrinsicsCosf64
                | KnownNames::StdIntrinsicsFloorf64
                | KnownNames::StdIntrinsicsExp2f64
                | KnownNames::StdIntrinsicsExpf64
                | KnownNames::StdIntrinsicsFabsf64
                | KnownNames::StdIntrinsicsLog10f64
                | KnownNames::StdIntrinsicsLog2f64
                | KnownNames::StdIntrinsicsLogf64
                | KnownNames::StdIntrinsicsNearbyintf64
                | KnownNames::StdIntrinsicsRintf64
                | KnownNames::StdIntrinsicsRoundf64
                | KnownNames::StdIntrinsicsSinf64
                | KnownNames::StdIntrinsicsSqrtf64
                | KnownNames::StdIntrinsicsTruncf64 => ExpressionType::F64,
                _ => assume_unreachable!("invalid name {:?} for intrinsic unary", name),
            },
            Expression::BitXor { left, .. } => left.expression.infer_type(),
            Expression::Cast { target_type, .. } => *target_type,
            Expression::CompileTimeConstant(c) => c.into(),
            Expression::ConditionalExpression {
                consequent,
                alternate,
                ..
            } => {
                if consequent.is_top() {
                    alternate.expression.infer_type()
                } else {
                    consequent.expression.infer_type()
                }
            }
            Expression::Div { left, .. } => left.expression.infer_type(),
            Expression::Equals { .. } => Bool,
            Expression::GreaterOrEqual { .. } => Bool,
            Expression::GreaterThan { .. } => Bool,
            Expression::Join { left, right } => {
                if left.is_compile_time_constant() {
                    right.expression.infer_type()
                } else {
                    left.expression.infer_type()
                }
            }
            Expression::LessOrEqual { .. } => Bool,
            Expression::LessThan { .. } => Bool,
            Expression::LogicalNot { .. } => Bool,
            Expression::Memcmp { .. } => ExpressionType::I32,
            Expression::Mul { left, .. } => left.expression.infer_type(),
            Expression::MulOverflows { .. } => Bool,
            Expression::Ne { .. } => Bool,
            Expression::Neg { operand } => operand.expression.infer_type(),
            Expression::Or { .. } => Bool,
            Expression::Offset { .. } => ThinPointer,
            Expression::Reference(_) => ThinPointer,
            Expression::InitialParameterValue { var_type, .. } => *var_type,
            Expression::Rem { right, .. } => right.expression.infer_type(),
            Expression::Shl { left, .. } => left.expression.infer_type(),
            Expression::ShlOverflows { .. } => Bool,
            Expression::Shr { left, .. } => left.expression.infer_type(),
            Expression::ShrOverflows { .. } => Bool,
            Expression::Sub { left, .. } => left.expression.infer_type(),
            Expression::SubOverflows { .. } => Bool,
            Expression::Switch { default, .. } => default.expression.infer_type(),
            Expression::TaggedExpression { operand, .. } => operand.expression.infer_type(),
            Expression::Transmute { target_type, .. } => *target_type,
            Expression::UninterpretedCall { result_type, .. } => *result_type,
            Expression::UnknownModelField { default, .. } => default.expression.infer_type(),
            Expression::UnknownTagCheck { .. } => Bool,
            Expression::UnknownTagField { .. } => I8,
            Expression::Variable { var_type, .. } => *var_type,
            Expression::WidenedJoin { operand, .. } => operand.expression.infer_type(),
        }
    }

    /// Determines if the given expression is the result of a non constant binary bitwise operation.
    #[logfn_inputs(TRACE)]
    pub fn is_bit_vector(&self) -> bool {
        match self {
            Expression::BitAnd { .. } | Expression::BitOr { .. } | Expression::BitXor { .. } => {
                true
            }
            Expression::ConditionalExpression {
                consequent: left,
                alternate: right,
                ..
            }
            | Expression::Join { left, right } => {
                left.expression.is_bit_vector() || right.expression.is_bit_vector()
            }
            _ => false,
        }
    }

    /// True if the expression involves a comparison operator (= != > >= < <=)
    #[logfn_inputs(TRACE)]
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Expression::Equals { .. }
                | Expression::Ne { .. }
                | Expression::GreaterOrEqual { .. }
                | Expression::GreaterThan { .. }
                | Expression::LessOrEqual { .. }
                | Expression::LessThan { .. }
        )
    }

    /// Determines if the given expression is the compile time constant 1u128.
    #[logfn_inputs(TRACE)]
    pub fn is_one(&self) -> bool {
        matches!(
            *self,
            Expression::CompileTimeConstant(ConstantDomain::U128(1u128))
        )
    }

    /// Determines if the given expression is the compile time constant 0u128.
    #[logfn_inputs(TRACE)]
    pub fn is_zero(&self) -> bool {
        matches!(
            *self,
            Expression::CompileTimeConstant(ConstantDomain::U128(0u128))
        )
    }

    /// Adds any heap blocks found in the associated expression to the given set.
    #[logfn_inputs(TRACE)]
    pub fn record_heap_blocks_and_strings(&self, result: &mut HashSet<Rc<AbstractValue>>) {
        match &self {
            Expression::Bottom | Expression::Top => (),
            Expression::Add { left, right }
            | Expression::AddOverflows { left, right, .. }
            | Expression::And { left, right }
            | Expression::BitAnd { left, right }
            | Expression::BitOr { left, right }
            | Expression::BitXor { left, right }
            | Expression::Div { left, right }
            | Expression::Equals { left, right }
            | Expression::GreaterOrEqual { left, right }
            | Expression::GreaterThan { left, right }
            | Expression::IntrinsicBinary { left, right, .. }
            | Expression::LessOrEqual { left, right }
            | Expression::LessThan { left, right }
            | Expression::Mul { left, right }
            | Expression::MulOverflows { left, right, .. }
            | Expression::Ne { left, right }
            | Expression::Offset { left, right }
            | Expression::Or { left, right }
            | Expression::Rem { left, right }
            | Expression::Shl { left, right }
            | Expression::ShlOverflows { left, right, .. }
            | Expression::Shr { left, right, .. }
            | Expression::ShrOverflows { left, right, .. }
            | Expression::Sub { left, right }
            | Expression::SubOverflows { left, right, .. } => {
                left.expression.record_heap_blocks_and_strings(result);
                right.expression.record_heap_blocks_and_strings(result);
            }
            Expression::BitNot { operand, .. }
            | Expression::Cast { operand, .. }
            | Expression::IntrinsicBitVectorUnary { operand, .. }
            | Expression::IntrinsicFloatingPointUnary { operand, .. }
            | Expression::Neg { operand }
            | Expression::LogicalNot { operand }
            | Expression::TaggedExpression { operand, .. }
            | Expression::Transmute { operand, .. }
            | Expression::UnknownTagCheck { operand, .. } => {
                operand.expression.record_heap_blocks_and_strings(result);
            }
            Expression::CompileTimeConstant(..) => (),
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                condition.expression.record_heap_blocks_and_strings(result);
                consequent.expression.record_heap_blocks_and_strings(result);
                alternate.expression.record_heap_blocks_and_strings(result);
            }
            Expression::HeapBlock { .. } => {
                result.insert(AbstractValue::make_from(self.clone(), 1));
            }
            Expression::HeapBlockLayout {
                length, alignment, ..
            } => {
                length.expression.record_heap_blocks_and_strings(result);
                alignment.expression.record_heap_blocks_and_strings(result);
            }
            Expression::InitialParameterValue { path, .. } | Expression::Variable { path, .. } => {
                path.record_heap_blocks_and_strings(result)
            }
            Expression::Join { left, right, .. } => {
                left.expression.record_heap_blocks_and_strings(result);
                right.expression.record_heap_blocks_and_strings(result);
            }
            Expression::Memcmp {
                left,
                right,
                length,
            } => {
                left.expression.record_heap_blocks_and_strings(result);
                right.expression.record_heap_blocks_and_strings(result);
                length.expression.record_heap_blocks_and_strings(result);
            }
            Expression::Reference(path) => path.record_heap_blocks_and_strings(result),
            Expression::Switch {
                discriminator,
                cases,
                default,
            } => {
                discriminator.record_heap_blocks_and_strings(result);
                for (case_val, case_result) in cases {
                    case_val.record_heap_blocks_and_strings(result);
                    case_result.record_heap_blocks_and_strings(result);
                }
                default.record_heap_blocks_and_strings(result);
            }
            Expression::UninterpretedCall {
                callee, arguments, ..
            } => {
                callee.record_heap_blocks_and_strings(result);
                for arg in arguments {
                    arg.record_heap_blocks_and_strings(result);
                }
            }
            Expression::UnknownModelField {
                path,
                default: operand,
            }
            | Expression::WidenedJoin { path, operand } => {
                path.record_heap_blocks_and_strings(result);
                operand.record_heap_blocks_and_strings(result);
            }
            Expression::UnknownTagField { path } => {
                path.record_heap_blocks_and_strings(result);
            }
        }
    }
}

/// The type of a place in memory, as understood by MIR.
/// For now, we are only really interested to distinguish between
/// floating point values and other values, because NaN != NaN.
/// In the future the other distinctions may be helpful to SMT solvers.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ExpressionType {
    Bool,
    Char,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    NonPrimitive,
    ThinPointer,
    U8,
    U16,
    U32,
    U64,
    U128,
    Unit,
    Usize,
}

impl From<&ConstantDomain> for ExpressionType {
    #[logfn_inputs(TRACE)]
    fn from(cv: &ConstantDomain) -> ExpressionType {
        use self::ExpressionType::*;
        match cv {
            ConstantDomain::Bottom => NonPrimitive,
            ConstantDomain::Char(..) => Char,
            ConstantDomain::False => Bool,
            ConstantDomain::Function(..) => NonPrimitive,
            ConstantDomain::I128(..) => I128,
            ConstantDomain::F64(..) => F64,
            ConstantDomain::F32(..) => F32,
            ConstantDomain::Str(..) => ThinPointer,
            ConstantDomain::True => Bool,
            ConstantDomain::U128(..) => U128,
            ConstantDomain::Unimplemented => NonPrimitive,
            ConstantDomain::Unit => Unit,
        }
    }
}

impl ExpressionType {
    pub fn from(ty_kind: &TyKind) -> ExpressionType {
        match ty_kind {
            TyKind::Adt(..) => ExpressionType::NonPrimitive,
            TyKind::Bool => ExpressionType::Bool,
            TyKind::Char => ExpressionType::Char,
            TyKind::Int(IntTy::Isize) => ExpressionType::Isize,
            TyKind::Int(IntTy::I8) => ExpressionType::I8,
            TyKind::Int(IntTy::I16) => ExpressionType::I16,
            TyKind::Int(IntTy::I32) => ExpressionType::I32,
            TyKind::Int(IntTy::I64) => ExpressionType::I64,
            TyKind::Int(IntTy::I128) => ExpressionType::I128,
            TyKind::Uint(UintTy::Usize) => ExpressionType::Usize,
            TyKind::Uint(UintTy::U8) => ExpressionType::U8,
            TyKind::Uint(UintTy::U16) => ExpressionType::U16,
            TyKind::Uint(UintTy::U32) => ExpressionType::U32,
            TyKind::Uint(UintTy::U64) => ExpressionType::U64,
            TyKind::Uint(UintTy::U128) => ExpressionType::U128,
            TyKind::Float(FloatTy::F32) => ExpressionType::F32,
            TyKind::Float(FloatTy::F64) => ExpressionType::F64,
            TyKind::RawPtr(TypeAndMut { ty: target, .. }) | TyKind::Ref(_, target, _) => {
                if matches!(target.kind(), TyKind::Slice(..) | TyKind::Str) {
                    // Such pointer types are non primitive because they are (pointer, length) tuples.
                    ExpressionType::NonPrimitive
                } else {
                    ExpressionType::ThinPointer
                }
            }
            TyKind::Tuple(types) => {
                if types.is_empty() {
                    ExpressionType::Unit
                } else {
                    ExpressionType::NonPrimitive
                }
            }
            _ => ExpressionType::NonPrimitive,
        }
    }

    pub fn as_rustc_type<'a>(&self, tcx: TyCtxt<'a>) -> Ty<'a> {
        use self::ExpressionType::*;
        match self {
            Bool => tcx.types.bool,
            Char => tcx.types.char,
            F32 => tcx.types.f32,
            F64 => tcx.types.f64,
            I8 => tcx.types.i8,
            I16 => tcx.types.i16,
            I32 => tcx.types.i32,
            I64 => tcx.types.i64,
            I128 => tcx.types.i128,
            Isize => tcx.types.isize,
            U8 => tcx.types.u8,
            U16 => tcx.types.u16,
            U32 => tcx.types.u32,
            U64 => tcx.types.u64,
            U128 => tcx.types.u128,
            Unit => tcx.types.unit,
            Usize => tcx.types.usize,
            ThinPointer => tcx.mk_imm_ptr(tcx.mk_unit()),
            NonPrimitive => tcx.types.trait_object_dummy_self,
        }
    }

    /// Returns true if this type is one of the floating point number types.
    #[logfn_inputs(TRACE)]
    pub fn is_floating_point_number(&self) -> bool {
        use self::ExpressionType::*;
        matches!(self, F32 | F64)
    }

    /// Returns true if this type is one of the integer types.
    #[logfn_inputs(TRACE)]
    pub fn is_integer(&self) -> bool {
        use self::ExpressionType::*;
        matches!(
            self,
            I8 | I16 | I32 | I64 | I128 | Isize | U8 | U16 | U32 | U64 | U128 | Usize
        )
    }

    /// Returns true if this type is not a primitive type. References are not regarded as
    /// primitives for this purpose.
    #[logfn_inputs(TRACE)]
    pub fn is_primitive(&self) -> bool {
        use self::ExpressionType::*;
        !matches!(self, NonPrimitive | ThinPointer)
    }

    /// Returns true if this type is one of the signed integer types.
    #[logfn_inputs(TRACE)]
    pub fn is_signed_integer(&self) -> bool {
        use self::ExpressionType::*;
        matches!(self, I8 | I16 | I32 | I64 | I128 | Isize)
    }

    /// Returns the unsigned equivalent of a signed integer type, otherwise returns self.
    #[logfn_inputs(TRACE)]
    pub fn as_unsigned(&self) -> ExpressionType {
        use self::ExpressionType::*;
        match self {
            I8 => ExpressionType::U8,
            I16 => ExpressionType::U16,
            I32 => ExpressionType::U32,
            I64 => ExpressionType::U64,
            I128 => ExpressionType::U128,
            Isize => ExpressionType::Usize,
            _ => *self,
        }
    }

    /// Returns true if this type is one of the unsigned integer types.
    #[logfn_inputs(TRACE)]
    pub fn is_unsigned_integer(&self) -> bool {
        use self::ExpressionType::*;
        matches!(self, U8 | U16 | U32 | U64 | U128 | Usize)
    }

    /// Returns the number of bits used to represent the given type, if primitive.
    /// For non primitive types the result is just 0.
    #[logfn_inputs(TRACE)]
    pub fn bit_length(&self) -> u8 {
        use self::ExpressionType::*;
        match self {
            Bool => 8,
            Char => 32,
            F32 => 32,
            F64 => 64,
            I8 => 8,
            I16 => 16,
            I32 => 32,
            I64 => 64,
            I128 => 128,
            Isize => 64,
            U8 => 8,
            U16 => 16,
            U32 => 32,
            U64 => 64,
            U128 => 128,
            Unit => 0,
            Usize => 64,
            ThinPointer => 64,
            NonPrimitive => 128,
        }
    }

    /// Returns the maximum value for this type, as a ConstantDomain element.
    /// If the type is not a primitive value, the result is Bottom.
    #[allow(clippy::cast_lossless)]
    #[logfn_inputs(TRACE)]
    pub fn max_value(&self) -> ConstantDomain {
        use self::ExpressionType::*;
        match self {
            Bool => ConstantDomain::U128(255_u128),
            Char => ConstantDomain::U128(std::char::MAX as u128),
            F32 => ConstantDomain::F32(f32::MAX.to_bits()),
            F64 => ConstantDomain::F64(f64::MAX.to_bits()),
            I8 => ConstantDomain::I128(i8::MAX as i128),
            I16 => ConstantDomain::I128(i16::MAX as i128),
            I32 => ConstantDomain::I128(i32::MAX as i128),
            I64 => ConstantDomain::I128(i64::MAX as i128),
            I128 => ConstantDomain::I128(i128::MAX),
            Isize => ConstantDomain::I128(isize::MAX as i128),
            U8 => ConstantDomain::U128(u8::MAX as u128),
            U16 => ConstantDomain::U128(u16::MAX as u128),
            U32 => ConstantDomain::U128(u32::MAX as u128),
            U64 => ConstantDomain::U128(u64::MAX as u128),
            U128 => ConstantDomain::U128(u128::MAX),
            Usize => ConstantDomain::U128(usize::MAX as u128),
            _ => ConstantDomain::Bottom,
        }
    }

    /// Returns the minimum value for this type, as a ConstantDomain element.
    /// If the type is not a primitive value, the result is Bottom.
    #[allow(clippy::cast_lossless)]
    #[logfn_inputs(TRACE)]
    pub fn min_value(&self) -> ConstantDomain {
        use self::ExpressionType::*;
        match self {
            Bool => ConstantDomain::U128(0_u128),
            Char => ConstantDomain::U128(0_u128),
            F32 => ConstantDomain::F32(f32::MIN.to_bits()),
            F64 => ConstantDomain::F64(f64::MIN.to_bits()),
            I8 => ConstantDomain::I128(i8::MIN as i128),
            I16 => ConstantDomain::I128(i16::MIN as i128),
            I32 => ConstantDomain::I128(i32::MIN as i128),
            I64 => ConstantDomain::I128(i64::MIN as i128),
            I128 => ConstantDomain::I128(i128::MIN),
            Isize => ConstantDomain::I128(isize::MIN as i128),
            U8 => ConstantDomain::U128(u8::MIN as u128),
            U16 => ConstantDomain::U128(u16::MIN as u128),
            U32 => ConstantDomain::U128(u32::MIN as u128),
            U64 => ConstantDomain::U128(u64::MIN as u128),
            U128 => ConstantDomain::U128(u128::MIN),
            Usize => ConstantDomain::U128(usize::MIN as u128),
            _ => ConstantDomain::Bottom,
        }
    }

    /// Returns the maximum value for this type, plus one, as ConstantDomain element.
    /// If the type is not a primitive integer value, the result is Bottom.
    #[allow(clippy::cast_lossless)]
    #[logfn_inputs(TRACE)]
    pub fn modulo_constant(&self) -> Rc<ConstantDomain> {
        use self::ExpressionType::*;
        match self {
            Char => Rc::new(ConstantDomain::U128((char::MAX) as u128 + 1)),
            U8 => Rc::new(ConstantDomain::U128((u8::MAX) as u128 + 1)),
            U16 => Rc::new(ConstantDomain::U128((u16::MAX) as u128 + 1)),
            U32 => Rc::new(ConstantDomain::U128((u32::MAX) as u128 + 1)),
            U64 => Rc::new(ConstantDomain::U128((u64::MAX) as u128 + 1)),
            Usize => Rc::new(ConstantDomain::U128((usize::MAX) as u128 + 1)),
            _ => Rc::new(ConstantDomain::Bottom),
        }
    }

    /// Returns the maximum value for this type, plus one, as an abstract value.
    /// If the type is not a primitive integer value, the result is Bottom.
    #[allow(clippy::cast_lossless)]
    #[logfn_inputs(TRACE)]
    pub fn modulo_value(&self) -> Rc<AbstractValue> {
        use self::ExpressionType::*;
        match self {
            Char => Rc::new(ConstantDomain::U128((char::MAX) as u128 + 1).into()),
            U8 => Rc::new(ConstantDomain::U128((u8::MAX) as u128 + 1).into()),
            U16 => Rc::new(ConstantDomain::U128((u16::MAX) as u128 + 1).into()),
            U32 => Rc::new(ConstantDomain::U128((u32::MAX) as u128 + 1).into()),
            U64 => Rc::new(ConstantDomain::U128((u64::MAX) as u128 + 1).into()),
            Usize => Rc::new(ConstantDomain::U128((usize::MAX) as u128 + 1).into()),
            _ => Rc::new(ConstantDomain::Bottom.into()),
        }
    }
}
