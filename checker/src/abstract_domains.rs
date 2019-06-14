// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#![allow(clippy::declare_interior_mutable_const)]
use crate::abstract_value::PathRefinement;
use crate::abstract_value::{AbstractValue, Path};
use crate::constant_domain::ConstantDomain;
use crate::environment::Environment;
use crate::expression::Expression::{ConditionalExpression, Join, Widen};
use crate::expression::{Expression, ExpressionType};
use crate::interval_domain::{self, IntervalDomain};
use crate::k_limits;

use rustc::ty::TyKind;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt::{Debug, Formatter, Result};
use std::hash::Hash;
use std::hash::Hasher;
use std::rc::Rc;
use syntax::ast;

// See https://github.com/facebookexperimental/MIRAI/blob/master/documentation/AbstractValues.md.

/// Basically, this domain is a structured container for other domains. It is also the only
/// client for the other domains.
#[derive(Serialize, Deserialize, Clone, Eq, Ord, PartialOrd)]
pub struct AbstractDomain {
    // todo: make this private
    // This is not a domain element, but a representation of how this instance has been constructed.
    // It is used to refine the instance with respect to path conditions and actual arguments.
    // It is also used to construct corresponding elements from other domains, when needed.
    pub expression: Expression,
    /// Cached interval computed on demand by get_as_interval.
    #[serde(skip)]
    interval: RefCell<Option<Rc<IntervalDomain>>>,
}

impl Debug for AbstractDomain {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.expression.fmt(f)
    }
}

impl Hash for AbstractDomain {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expression.hash(state);
    }
}

impl PartialEq for AbstractDomain {
    fn eq(&self, other: &Self) -> bool {
        self.expression.eq(&other.expression)
    }
}

/// An abstract domain element that all represent the impossible concrete value.
/// I.e. the corresponding set of possible concrete values is empty.
pub const BOTTOM: AbstractDomain = AbstractDomain {
    expression: Expression::Bottom,
    interval: RefCell::new(None),
};

/// An abstract domain element that all represent the single concrete value, false.
pub const FALSE: AbstractDomain = AbstractDomain {
    expression: Expression::CompileTimeConstant(ConstantDomain::False),
    interval: RefCell::new(None),
};

/// An abstract domain element that all represents all possible concrete values.
pub const TOP: AbstractDomain = AbstractDomain {
    expression: Expression::Top,
    interval: RefCell::new(None),
};

/// An abstract domain element that all represent the single concrete value, true.
pub const TRUE: AbstractDomain = AbstractDomain {
    expression: Expression::CompileTimeConstant(ConstantDomain::True),
    interval: RefCell::new(None),
};

impl<'a> From<&TyKind<'a>> for ExpressionType {
    fn from(ty_kind: &TyKind<'a>) -> ExpressionType {
        match ty_kind {
            TyKind::Bool => ExpressionType::Bool,
            TyKind::Char => ExpressionType::Char,
            TyKind::Int(ast::IntTy::Isize) => ExpressionType::Isize,
            TyKind::Int(ast::IntTy::I8) => ExpressionType::I8,
            TyKind::Int(ast::IntTy::I16) => ExpressionType::I16,
            TyKind::Int(ast::IntTy::I32) => ExpressionType::I32,
            TyKind::Int(ast::IntTy::I64) => ExpressionType::I64,
            TyKind::Int(ast::IntTy::I128) => ExpressionType::I128,
            TyKind::Uint(ast::UintTy::Usize) => ExpressionType::Usize,
            TyKind::Uint(ast::UintTy::U8) => ExpressionType::U8,
            TyKind::Uint(ast::UintTy::U16) => ExpressionType::U16,
            TyKind::Uint(ast::UintTy::U32) => ExpressionType::U32,
            TyKind::Uint(ast::UintTy::U64) => ExpressionType::U64,
            TyKind::Uint(ast::UintTy::U128) => ExpressionType::U128,
            TyKind::Float(ast::FloatTy::F32) => ExpressionType::F32,
            TyKind::Float(ast::FloatTy::F64) => ExpressionType::F64,
            TyKind::Closure(..)
            | TyKind::Dynamic(..)
            | TyKind::Foreign(..)
            | TyKind::FnDef(..)
            | TyKind::FnPtr(..)
            | TyKind::Generator(..)
            | TyKind::GeneratorWitness(..)
            | TyKind::RawPtr(..)
            | TyKind::Ref(..)
            | TyKind::Slice(..)
            | TyKind::Str => ExpressionType::Reference,
            _ => ExpressionType::NonPrimitive,
        }
    }
}

impl From<bool> for AbstractDomain {
    fn from(b: bool) -> AbstractDomain {
        if b {
            AbstractDomain {
                expression: Expression::CompileTimeConstant(ConstantDomain::True),
                interval: RefCell::new(None),
            }
        } else {
            AbstractDomain {
                expression: Expression::CompileTimeConstant(ConstantDomain::False),
                interval: RefCell::new(None),
            }
        }
    }
}

impl From<ConstantDomain> for AbstractDomain {
    fn from(cv: ConstantDomain) -> AbstractDomain {
        AbstractDomain {
            expression: Expression::CompileTimeConstant(cv),
            interval: RefCell::new(None),
        }
    }
}

impl From<Expression> for AbstractDomain {
    fn from(expr: Expression) -> AbstractDomain {
        AbstractDomain {
            expression: expr,
            interval: RefCell::new(None),
        }
    }
}

pub trait AbstractDomainTrait: Sized {
    fn addition(&self, other: Self) -> Self;
    fn add_overflows(&self, other: Self, target_type: ExpressionType) -> Self;
    fn and(&self, other: Self) -> Self;
    fn as_bool_if_known(&self) -> Option<bool>;
    fn bit_and(&self, other: Self) -> Self;
    fn bit_or(&self, other: Self) -> Self;
    fn bit_xor(&self, other: Self) -> Self;
    fn cast(&self, target_type: ExpressionType) -> Self;
    fn conditional_expression(&self, consequent: Self, alternate: Self) -> Self;
    fn divide(&self, other: Self) -> Self;
    fn equals(&self, other: Self) -> Self;
    fn greater_or_equal(&self, other: Self) -> Self;
    fn greater_than(&self, other: Self) -> Self;
    fn implies(&self, other: &Self) -> bool;
    fn implies_not(&self, other: &Self) -> bool;
    fn is_bottom(&self) -> bool;
    fn is_top(&self) -> bool;
    fn join(&self, other: Self, path: &Rc<Path>) -> Self;
    fn less_or_equal(&self, other: Self) -> Self;
    fn less_than(&self, other: Self) -> Self;
    fn multiply(&self, other: Self) -> Self;
    fn mul_overflows(&self, other: Self, target_type: ExpressionType) -> Self;
    fn negate(self) -> Self;
    fn not_equals(&self, other: Self) -> Self;
    fn logical_not(&self) -> Self;
    fn offset(&self, other: Self) -> Self;
    fn or(&self, other: Self) -> Self;
    fn remainder(&self, other: Self) -> Self;
    fn shift_left(&self, other: Self) -> Self;
    fn shl_overflows(&self, other: Self, target_type: ExpressionType) -> Self;
    fn shr(&self, other: Self, expression_type: ExpressionType) -> Self;
    fn shr_overflows(&self, other: Self, target_type: ExpressionType) -> Self;
    fn subtract(&self, other: Self) -> Self;
    fn sub_overflows(&self, other: Self, target_type: ExpressionType) -> Self;
    fn subset(&self, other: &Self) -> bool;
    fn try_to_simplify_binary_op(
        &self,
        other: Self,
        const_op: fn(&ConstantDomain, &ConstantDomain) -> ConstantDomain,
        operation: fn(Self, Self) -> Self,
    ) -> Self;
    fn try_to_distribute_binary_op(&self, other: Self, operation: fn(Self, Self) -> Self) -> Self;
    fn get_cached_interval(&self) -> Rc<IntervalDomain>;
    fn get_as_interval(&self) -> IntervalDomain;
    fn refine_paths(&self, environment: &Environment) -> Self;
    fn refine_parameters(&self, arguments: &[(Rc<Path>, AbstractValue)]) -> Self;
    fn refine_with(&self, path_condition: &Self, depth: usize) -> Self;
    fn widen(&self, path: &Rc<Path>) -> Self;
}

impl AbstractDomainTrait for Rc<AbstractDomain> {
    /// Returns an element that is "self + other".
    fn addition(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let Expression::Add { left, right } = &self.expression {
            if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
                (&right.expression, &other.expression)
            {
                let folded = v1.add(v2);
                if folded != ConstantDomain::Bottom {
                    return left.addition(Rc::new(folded.into()));
                }
            }
        }
        self.try_to_simplify_binary_op(other, ConstantDomain::add, |left, right| {
            Rc::new(
                Expression::Add {
                    left: left.clone(),
                    right: right.clone(),
                }
                .into(),
            )
        })
    }

    /// Returns an element that is true if "self + other" is not in range of target_type.
    fn add_overflows(
        &self,
        other: Rc<AbstractDomain>,
        target_type: ExpressionType,
    ) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.add_overflows(v2, &target_type);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        let interval = self.get_cached_interval().add(&other.get_cached_interval());
        if interval.is_contained_in(&target_type) {
            return Rc::new(false.into());
        }
        Rc::new(
            Expression::AddOverflows {
                left: self.clone(),
                right: other.clone(),
                result_type: target_type,
            }
            .into(),
        )
    }

    /// Returns an element that is "self && other".
    fn and(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        let self_bool = self.as_bool_if_known();
        if let Some(false) = self_bool {
            return Rc::new(false.into());
        };
        let other_bool = other.as_bool_if_known();
        if let Some(false) = other_bool {
            return Rc::new(false.into());
        };
        if self_bool.unwrap_or(false) {
            if other_bool.unwrap_or(false) {
                Rc::new(true.into())
            } else {
                other.clone()
            }
        } else if other_bool.unwrap_or(false)
            || self.is_top()
            || self.is_bottom() && other.is_bottom()
        {
            self.clone()
        } else if other.is_top() {
            other.clone()
        } else {
            // todo: #32 more simplifications
            Rc::new(
                Expression::And {
                    left: self.clone(),
                    right: other.clone(),
                }
                .into(),
            )
        }
    }

    /// The Boolean value of this expression, if known, otherwise None.
    fn as_bool_if_known(&self) -> Option<bool> {
        match self.expression {
            Expression::CompileTimeConstant(ConstantDomain::True) => Some(true),
            Expression::CompileTimeConstant(ConstantDomain::False) => Some(false),
            _ => {
                // todo: ask other domains about this (construct some if need be).
                None
            }
        }
    }

    /// Returns an element that is "self & other".
    fn bit_and(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.bit_and(v2);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        Rc::new(
            Expression::BitAnd {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self | other".
    fn bit_or(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.bit_or(v2);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        Rc::new(
            Expression::BitOr {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self ^ other".
    fn bit_xor(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.bit_xor(v2);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        Rc::new(
            Expression::BitXor {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self as target_type".
    fn cast(&self, target_type: ExpressionType) -> Rc<AbstractDomain> {
        if let Expression::CompileTimeConstant(v1) = &self.expression {
            let result = v1.cast(&target_type);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        match &self.expression {
            Expression::Bottom => self.clone(),
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => condition.conditional_expression(
                consequent.cast(target_type.clone()),
                alternate.cast(target_type),
            ),
            Expression::Join { left, right, path } => left
                .cast(target_type.clone())
                .join(right.cast(target_type), &path),
            _ => Rc::new(
                Expression::Cast {
                    operand: self.clone(),
                    target_type: target_type.clone(),
                }
                .into(),
            ),
        }
    }

    /// Returns an element that is "if self { consequent } else { alternate }".
    fn conditional_expression(
        &self,
        consequent: Rc<AbstractDomain>,
        alternate: Rc<AbstractDomain>,
    ) -> Rc<AbstractDomain> {
        if self.is_bottom() {
            // If the condition is impossible so is the expression.
            return consequent;
        }
        if consequent.expression == alternate.expression {
            // c ? x : x is just x
            return consequent;
        }
        let join_condition_as_bool = self.as_bool_if_known();
        if join_condition_as_bool == Some(true) {
            // true ? x : y is just x
            return consequent;
        } else if join_condition_as_bool == Some(false) {
            // false ? x : y is just y
            return alternate;
        }
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&consequent.expression, &alternate.expression)
        {
            match (v1, v2) {
                (ConstantDomain::True, ConstantDomain::False) => {
                    // c ? true : false is just c
                    return self.clone();
                }
                (ConstantDomain::False, ConstantDomain::True) => {
                    // c ? false : true is just !c
                    return self.logical_not();
                }
                _ => (),
            }
        }
        Rc::new(
            ConditionalExpression {
                condition: self.clone(),
                consequent,
                alternate,
            }
            .into(),
        )
    }

    /// Returns an element that is "self / other".
    fn divide(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        self.try_to_simplify_binary_op(other, ConstantDomain::div, |left, right| {
            Rc::new(
                Expression::Div {
                    left: left.clone(),
                    right: right.clone(),
                }
                .into(),
            )
        })
    }

    /// Returns an element that is "self == other".
    fn equals(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            return Rc::new(v1.equals(v2).into());
        };
        match (&self.expression, &other.expression) {
            // If self and other are the same location in memory, return true unless the value might be NaN.
            (
                Expression::Variable {
                    path: p1,
                    var_type: t1,
                },
                Expression::Variable {
                    path: p2,
                    var_type: t2,
                },
            ) => {
                if p1 == p2 {
                    match (t1, t2) {
                        (ExpressionType::F32, _)
                        | (ExpressionType::F64, _)
                        | (_, ExpressionType::F32)
                        | (_, ExpressionType::F64) => (),
                        _ => {
                            return Rc::new(true.into());
                        }
                    }
                }
            }
            // (c ? 0 : 1) == 0 is the same as c
            // (c ? 1 : 0) == 1 is the same as c
            // (c ? 0 : 1) == 1 is the same as !c
            // (c ? 1 : 0) == 0 is the same as !c
            (
                Expression::ConditionalExpression {
                    condition,
                    consequent,
                    alternate,
                },
                Expression::CompileTimeConstant(ConstantDomain::U128(result_val)),
            ) => {
                if let (
                    Expression::CompileTimeConstant(ConstantDomain::U128(cons_val)),
                    Expression::CompileTimeConstant(ConstantDomain::U128(alt_val)),
                ) = (&consequent.expression, &alternate.expression)
                {
                    match (*cons_val, *alt_val, *result_val) {
                        (0, 1, 0) | (1, 0, 1) => {
                            return condition.clone();
                        }
                        (0, 1, 1) | (1, 0, 0) => {
                            return condition.logical_not();
                        }
                        _ => (),
                    }
                }
            }
            // !x == 0 is the same as x when x is Boolean. Canonicalize it to the latter.
            (
                Expression::Not { operand },
                Expression::CompileTimeConstant(ConstantDomain::U128(val)),
            ) => {
                if *val == 0 && operand.expression.infer_type() == ExpressionType::Bool {
                    return operand.clone();
                }
            }
            // x == 0 is the same as !x when x is a Boolean. Canonicalize it to the latter.
            // x == 1 is the same as x when x is a Boolean. Canonicalize it to the latter.
            (x, Expression::CompileTimeConstant(ConstantDomain::U128(val))) => {
                if x.infer_type() == ExpressionType::Bool {
                    if *val == 0 {
                        return self.logical_not();
                    } else if *val == 1 {
                        return self.clone();
                    }
                }
            }
            (x, y) => {
                // If self and other are the same expression and the expression could not result in NaN
                // and the expression represents exactly one value, we can simplify this to true.
                if x == y && !x.infer_type().is_floating_point_number() {
                    return Rc::new(true.into());
                }
            }
        }
        // Return an equals expression rather than a constant expression.
        Rc::new(
            Expression::Equals {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self >= other".
    fn greater_or_equal(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            return Rc::new(v1.greater_or_equal(v2).into());
        };
        if let Some(result) = self
            .get_cached_interval()
            .greater_or_equal(&other.get_cached_interval())
        {
            return Rc::new(result.into());
        }
        Rc::new(
            Expression::GreaterOrEqual {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self > other".
    fn greater_than(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            return Rc::new(v1.greater_than(v2).into());
        };
        if let Some(result) = self
            .get_cached_interval()
            .greater_than(other.get_cached_interval().as_ref())
        {
            return Rc::new(result.into());
        }
        Rc::new(
            Expression::GreaterThan {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns true if "self => other" is known at compile time to be true.
    /// Returning false does not imply the implication is false, just that we do not know.
    fn implies(&self, other: &Rc<AbstractDomain>) -> bool {
        // x => true, is always true
        // false => x, is always true
        // x => x, is always true
        if other.as_bool_if_known().unwrap_or(false)
            || !self.as_bool_if_known().unwrap_or(true)
            || self.eq(other)
        {
            return true;
        }

        // x && y => x
        // y && x => x
        if let Expression::And { left, right } = &self.expression {
            return left.implies(other) || right.implies(other);
        }
        false
    }

    /// Returns true if "self => !other" is known at compile time to be true.
    /// Returning false does not imply the implication is false, just that we do not know.
    fn implies_not(&self, other: &Rc<AbstractDomain>) -> bool {
        // x => !false, is always true
        // false => !x, is always true
        if !other.as_bool_if_known().unwrap_or(true) || !self.as_bool_if_known().unwrap_or(true) {
            return true;
        };
        // !x => !x
        if let Expression::Not { ref operand } = self.expression {
            return (**operand).eq(other);
        }
        false
    }

    /// True if the set of concrete values that correspond to this domain is empty.
    fn is_bottom(&self) -> bool {
        match self.expression {
            Expression::Bottom => true,
            _ => false,
        }
    }

    /// True if all possible concrete values are elements of the set corresponding to this domain.
    fn is_top(&self) -> bool {
        match self.expression {
            Expression::Top => true,
            _ => false,
        }
    }

    /// Returns a domain whose corresponding set of concrete values include all of the values
    /// corresponding to self and other. In effect this behaves like set union.
    fn join(&self, other: Rc<AbstractDomain>, path: &Rc<Path>) -> Rc<AbstractDomain> {
        // {} union y is just y
        if self.is_bottom() {
            return other;
        }
        // x union {} is just x
        if other.is_bottom() {
            return self.clone();
        }
        // x union x is just x
        if (*self) == other {
            return other;
        }
        Rc::new(
            Expression::Join {
                path: path.clone(),
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self <= other".
    fn less_or_equal(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            return Rc::new(v1.less_or_equal(v2).into());
        };
        if let Some(result) = self
            .get_cached_interval()
            .less_equal(&other.get_cached_interval())
        {
            return Rc::new(result.into());
        }
        Rc::new(
            Expression::LessOrEqual {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is self < other
    fn less_than(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            return Rc::new(v1.less_than(v2).into());
        };
        if let Some(result) = self
            .get_cached_interval()
            .less_than(other.get_cached_interval().as_ref())
        {
            return Rc::new(result.into());
        }
        Rc::new(
            Expression::LessThan {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self * other".
    fn multiply(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        self.try_to_simplify_binary_op(other, ConstantDomain::mul, |left, right| {
            Rc::new(
                Expression::Mul {
                    left: left.clone(),
                    right: right.clone(),
                }
                .into(),
            )
        })
    }

    /// Returns an element that is true if "self * other" is not in range of target_type.
    fn mul_overflows(
        &self,
        other: Rc<AbstractDomain>,
        target_type: ExpressionType,
    ) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.mul_overflows(v2, &target_type);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        let interval = self.get_cached_interval().mul(&other.get_cached_interval());
        if interval.is_contained_in(&target_type) {
            return Rc::new(false.into());
        }
        Rc::new(
            Expression::MulOverflows {
                left: self.clone(),
                right: other.clone(),
                result_type: target_type,
            }
            .into(),
        )
    }

    /// Returns an element that is "-self".
    fn negate(self) -> Rc<AbstractDomain> {
        if let Expression::CompileTimeConstant(v1) = &self.expression {
            let result = v1.neg();
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        Rc::new(
            Expression::Neg {
                operand: self.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self != other".
    fn not_equals(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            return Rc::new(v1.not_equals(v2).into());
        };
        Rc::new(
            Expression::Ne {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "!self".
    fn logical_not(&self) -> Rc<AbstractDomain> {
        if let Expression::CompileTimeConstant(v1) = &self.expression {
            let result = v1.not();
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        match &self.expression {
            Expression::Bottom => self.clone(),
            Expression::Not { operand } => operand.clone(),
            _ => Rc::new(
                Expression::Not {
                    operand: self.clone(),
                }
                .into(),
            ),
        }
    }

    /// Returns an element that is "self.other".
    fn offset(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        Rc::new(
            Expression::Offset {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is "self || other".
    fn or(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if self.as_bool_if_known().unwrap_or(false) || other.as_bool_if_known().unwrap_or(false) {
            Rc::new(true.into())
        } else if self.is_bottom() || !self.as_bool_if_known().unwrap_or(true) {
            other.clone()
        } else if other.is_bottom() || !other.as_bool_if_known().unwrap_or(true) {
            self.clone()
        } else {
            match (&self.expression, &other.expression) {
                (Expression::Not { ref operand }, _) if (**operand).eq(&other) => {
                    Rc::new(true.into())
                }
                (_, Expression::Not { ref operand }) if (**operand).eq(&self) => {
                    Rc::new(true.into())
                }
                _ => Rc::new(
                    Expression::Or {
                        left: self.clone(),
                        right: other.clone(),
                    }
                    .into(),
                ),
            }
        }
    }

    /// Returns an element that is "self % other".
    fn remainder(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        self.try_to_simplify_binary_op(other, ConstantDomain::rem, |left, right| {
            Rc::new(
                Expression::Rem {
                    left: left.clone(),
                    right: right.clone(),
                }
                .into(),
            )
        })
    }

    /// Returns an element that is "self << other".
    fn shift_left(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.shl(v2);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        Rc::new(
            Expression::Shl {
                left: self.clone(),
                right: other.clone(),
            }
            .into(),
        )
    }

    /// Returns an element that is true if "self << other" shifts away all bits.
    fn shl_overflows(
        &self,
        other: Rc<AbstractDomain>,
        target_type: ExpressionType,
    ) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.shl_overflows(v2, &target_type);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        let interval = other.get_cached_interval();
        if interval.is_contained_in_width_of(&target_type) {
            return Rc::new(false.into());
        }
        Rc::new(
            Expression::ShlOverflows {
                left: self.clone(),
                right: other.clone(),
                result_type: target_type,
            }
            .into(),
        )
    }

    /// Returns an element that is "self >> other".
    fn shr(
        &self,
        other: Rc<AbstractDomain>,
        expression_type: ExpressionType,
    ) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.shr(v2);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        Rc::new(
            Expression::Shr {
                left: self.clone(),
                right: other.clone(),
                result_type: expression_type,
            }
            .into(),
        )
    }

    /// Returns an element that is true if "self >> other" shifts away all bits.
    fn shr_overflows(
        &self,
        other: Rc<AbstractDomain>,
        target_type: ExpressionType,
    ) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.shr_overflows(v2, &target_type);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        let interval = &other.get_cached_interval();
        if interval.is_contained_in_width_of(&target_type) {
            return Rc::new(false.into());
        }
        Rc::new(
            Expression::ShrOverflows {
                left: self.clone(),
                right: other.clone(),
                result_type: target_type,
            }
            .into(),
        )
    }

    /// Returns an element that is "self - other".
    fn subtract(&self, other: Rc<AbstractDomain>) -> Rc<AbstractDomain> {
        self.try_to_simplify_binary_op(other, ConstantDomain::sub, |left, right| {
            Rc::new(
                Expression::Sub {
                    left: left.clone(),
                    right: right.clone(),
                }
                .into(),
            )
        })
    }

    /// Returns an element that is true if "self - other" is not in range of target_type.
    fn sub_overflows(
        &self,
        other: Rc<AbstractDomain>,
        target_type: ExpressionType,
    ) -> Rc<AbstractDomain> {
        if let (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) =
            (&self.expression, &other.expression)
        {
            let result = v1.sub_overflows(v2, &target_type);
            if result != ConstantDomain::Bottom {
                return Rc::new(result.into());
            }
        };
        let interval = self.get_cached_interval().sub(&other.get_cached_interval());
        if interval.is_contained_in(&target_type) {
            return Rc::new(false.into());
        }
        Rc::new(
            Expression::SubOverflows {
                left: self.clone(),
                right: other.clone(),
                result_type: target_type,
            }
            .into(),
        )
    }

    /// True if all of the concrete values that correspond to self also correspond to other.
    /// Note: !x.subset(y) does not imply y.subset(x).
    fn subset(&self, other: &Rc<AbstractDomain>) -> bool {
        if self == other {
            return true;
        };
        match (&self.expression, &other.expression) {
            // The empty set is a subset of every other set.
            (Expression::Bottom, _) => true,
            // A non empty set is not a subset of the empty set.
            (_, Expression::Bottom) => false,
            // Every set is a subset of the universal set.
            (_, Expression::Top) => true,
            // The universal set is not a subset of any set other than the universal set.
            (Expression::Top, _) => false,
            // Widened expressions are equal if their paths are equal, regardless of their operand values.
            (Expression::Widen { path: p1, .. }, Expression::Widen { path: p2, .. }) => p1 == p2,
            // (condition ? consequent : alternate) is a subset of x if both consequent and alternate are subsets of x.
            (
                Expression::ConditionalExpression {
                    consequent,
                    alternate,
                    ..
                },
                _,
            ) => {
                // This is a conservative answer. False does not imply other.subset(self).
                consequent.subset(other) && alternate.subset(other)
            }
            // x is a subset of (condition ? consequent : alternate) if x is a subset of consequent or alternate.
            (
                _,
                Expression::ConditionalExpression {
                    consequent,
                    alternate,
                    ..
                },
            ) => {
                // This is a conservative answer. False does not imply other.subset(self).
                self.subset(&consequent) || self.subset(&alternate)
            }
            // x subset widen { z } if x subset z
            (_, Expression::Widen { operand, .. }) => self.subset(&operand),
            // (left join right) is a subset of x if both left and right are subsets of x.
            (Expression::Join { left, right, .. }, _) => {
                // This is a conservative answer. False does not imply other.subset(self).
                left.subset(other) && right.subset(other)
            }
            // x is a subset of (left join right) if x is a subset of either left or right.
            (_, Expression::Join { left, right, .. }) => {
                // This is a conservative answer. False does not imply other.subset(self).
                self.subset(&left) || self.subset(&right)
            }
            // in all other cases we conservatively answer false
            _ => false,
        }
    }

    /// Gets or constructs an interval that is cached.
    fn get_cached_interval(&self) -> Rc<IntervalDomain> {
        {
            let mut cache = self.interval.borrow_mut();
            if cache.is_some() {
                return cache.as_ref().unwrap().clone();
            }
            let interval = self.get_as_interval();
            *cache = Some(Rc::new(interval));
        }
        self.get_cached_interval()
    }

    /// Constructs an element of the Interval domain for simple expressions.
    fn get_as_interval(&self) -> IntervalDomain {
        match &self.expression {
            Expression::Top => interval_domain::BOTTOM,
            Expression::Add { left, right } => left.get_as_interval().add(&right.get_as_interval()),
            Expression::CompileTimeConstant(ConstantDomain::I128(val)) => (*val).into(),
            Expression::CompileTimeConstant(ConstantDomain::U128(val)) => (*val).into(),
            Expression::ConditionalExpression {
                consequent,
                alternate,
                ..
            } => consequent
                .get_as_interval()
                .widen(&alternate.get_as_interval()),
            Expression::Join { left, right, .. } => {
                left.get_as_interval().widen(&right.get_as_interval())
            }
            Expression::Mul { left, right } => left.get_as_interval().mul(&right.get_as_interval()),
            Expression::Neg { operand } => operand.get_as_interval().neg(),
            Expression::Sub { left, right } => left.get_as_interval().sub(&right.get_as_interval()),
            Expression::Variable { .. } => interval_domain::BOTTOM,
            Expression::Widen { operand, .. } => {
                let interval = operand.get_as_interval();
                if interval.is_bottom() {
                    return interval;
                }
                if let Expression::Join { left, .. } = &operand.expression {
                    let left_interval = left.get_as_interval();
                    if left_interval.is_bottom() {
                        return interval_domain::BOTTOM;
                    }
                    match (left_interval.lower_bound(), interval.lower_bound()) {
                        (Some(llb), Some(lb)) if llb == lb => {
                            // The lower bound is finite and does not change as a result of the fixed
                            // point computation, so we can keep it, but we remove the upper bound.
                            return interval.remove_upper_bound();
                        }
                        _ => (),
                    }
                    match (left_interval.upper_bound(), interval.upper_bound()) {
                        (Some(lub), Some(ub)) if lub == ub => {
                            // The upper bound is finite and does not change as a result of the fixed
                            // point computation, so we can keep it, but we remove the lower bound.
                            return interval.remove_lower_bound();
                        }
                        _ => (),
                    }
                }
                interval
            }
            _ => interval_domain::BOTTOM,
        }
    }

    /// Replaces occurrences of Expression::Variable(path) with the value at that path
    /// in the given environment (if there is such a value).
    fn refine_paths(&self, environment: &Environment) -> Rc<AbstractDomain> {
        match &self.expression {
            Expression::Top | Expression::Bottom | Expression::AbstractHeapAddress(..) => {
                self.clone()
            }
            Expression::Add { left, right } => left
                .refine_paths(environment)
                .addition(right.refine_paths(environment)),
            Expression::AddOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_paths(environment)
                .add_overflows(right.refine_paths(environment), result_type.clone()),
            Expression::And { left, right } => left
                .refine_paths(environment)
                .and(right.refine_paths(environment)),
            Expression::BitAnd { left, right } => left
                .refine_paths(environment)
                .bit_and(right.refine_paths(environment)),
            Expression::BitOr { left, right } => left
                .refine_paths(environment)
                .bit_or(right.refine_paths(environment)),
            Expression::BitXor { left, right } => left
                .refine_paths(environment)
                .bit_xor(right.refine_paths(environment)),
            Expression::Cast {
                operand,
                target_type,
            } => operand.refine_paths(environment).cast(target_type.clone()),
            Expression::CompileTimeConstant(..) => self.clone(),
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => condition.refine_paths(environment).conditional_expression(
                consequent.refine_paths(environment),
                alternate.refine_paths(environment),
            ),
            Expression::Div { left, right } => left
                .refine_paths(environment)
                .divide(right.refine_paths(environment)),
            Expression::Equals { left, right } => left
                .refine_paths(environment)
                .equals(right.refine_paths(environment)),
            Expression::GreaterOrEqual { left, right } => left
                .refine_paths(environment)
                .greater_or_equal(right.refine_paths(environment)),
            Expression::GreaterThan { left, right } => left
                .refine_paths(environment)
                .greater_than(right.refine_paths(environment)),
            Expression::Join { left, right, path } => left
                .refine_paths(environment)
                .join(right.refine_paths(environment), &path),
            Expression::LessOrEqual { left, right } => left
                .refine_paths(environment)
                .less_or_equal(right.refine_paths(environment)),
            Expression::LessThan { left, right } => left
                .refine_paths(environment)
                .less_than(right.refine_paths(environment)),
            Expression::Mul { left, right } => left
                .refine_paths(environment)
                .multiply(right.refine_paths(environment)),
            Expression::MulOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_paths(environment)
                .mul_overflows(right.refine_paths(environment), result_type.clone()),
            Expression::Ne { left, right } => left
                .refine_paths(environment)
                .not_equals(right.refine_paths(environment)),
            Expression::Neg { operand } => operand.refine_paths(environment).negate(),
            Expression::Not { operand } => operand.refine_paths(environment).logical_not(),
            Expression::Offset { left, right } => left
                .refine_paths(environment)
                .offset(right.refine_paths(environment)),
            Expression::Or { left, right } => left
                .refine_paths(environment)
                .or(right.refine_paths(environment)),
            Expression::Reference(path) => {
                let refined_path = path.refine_paths(environment);
                Rc::new(Expression::Reference(refined_path).into())
            }
            Expression::Rem { left, right } => left
                .refine_paths(environment)
                .remainder(right.refine_paths(environment)),
            Expression::Shl { left, right } => left
                .refine_paths(environment)
                .shift_left(right.refine_paths(environment)),
            Expression::ShlOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_paths(environment)
                .shl_overflows(right.refine_paths(environment), result_type.clone()),
            Expression::Shr {
                left,
                right,
                result_type,
            } => left
                .refine_paths(environment)
                .shr(right.refine_paths(environment), result_type.clone()),
            Expression::ShrOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_paths(environment)
                .shr_overflows(right.refine_paths(environment), result_type.clone()),
            Expression::Sub { left, right } => left
                .refine_paths(environment)
                .subtract(right.refine_paths(environment)),
            Expression::SubOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_paths(environment)
                .sub_overflows(right.refine_paths(environment), result_type.clone()),
            Expression::UnknownModelField { path, default } => {
                if let Some(val) = environment.value_at(&path) {
                    // This environment has a value for the model field.
                    val.domain.clone()
                } else if let Path::QualifiedPath { qualifier, .. } = path.as_ref() {
                    if environment.value_at(&*qualifier).is_some() {
                        // This environment does have a value for the qualifier, so the buck stops here.
                        default.clone()
                    } else {
                        // Keep passing the buck to the next caller.
                        Rc::new(
                            Expression::UnknownModelField {
                                path: path.clone(),
                                default: default.clone(),
                            }
                            .into(),
                        )
                    }
                } else {
                    unreachable!()
                }
            }
            Expression::Variable { path, var_type } => {
                if let Some(val) = environment.value_at(&path) {
                    val.domain.clone()
                } else {
                    let refined_path = path.refine_paths(environment);
                    if let Path::Constant { value } = refined_path.as_ref() {
                        value.domain.clone()
                    } else if let Some(val) = environment.value_at(&refined_path) {
                        val.domain.clone()
                    } else {
                        Rc::new(
                            Expression::Variable {
                                path: refined_path,
                                var_type: var_type.clone(),
                            }
                            .into(),
                        )
                    }
                }
            }
            Expression::Widen { path, operand } => operand
                .refine_paths(environment)
                .widen(&path.refine_paths(environment)),
        }
    }

    /// Returns a value that is simplified (refined) by replacing parameter values
    /// with their corresponding argument values. If no refinement is possible
    /// the result is simply a clone of this value.
    fn refine_parameters(&self, arguments: &[(Rc<Path>, AbstractValue)]) -> Rc<AbstractDomain> {
        match &self.expression {
            Expression::Top | Expression::Bottom | Expression::AbstractHeapAddress(..) => {
                self.clone()
            }
            Expression::Add { left, right } => left
                .refine_parameters(arguments)
                .addition(right.refine_parameters(arguments)),
            Expression::AddOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_parameters(arguments)
                .add_overflows(right.refine_parameters(arguments), result_type.clone()),
            Expression::And { left, right } => left
                .refine_parameters(arguments)
                .and(right.refine_parameters(arguments)),
            Expression::BitAnd { left, right } => left
                .refine_parameters(arguments)
                .bit_and(right.refine_parameters(arguments)),
            Expression::BitOr { left, right } => left
                .refine_parameters(arguments)
                .bit_or(right.refine_parameters(arguments)),
            Expression::BitXor { left, right } => left
                .refine_parameters(arguments)
                .bit_xor(right.refine_parameters(arguments)),
            Expression::Cast {
                operand,
                target_type,
            } => operand
                .refine_parameters(arguments)
                .cast(target_type.clone()),
            Expression::CompileTimeConstant(..) => self.clone(),
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => condition
                .refine_parameters(arguments)
                .conditional_expression(
                    consequent.refine_parameters(arguments),
                    alternate.refine_parameters(arguments),
                ),
            Expression::Div { left, right } => left
                .refine_parameters(arguments)
                .divide(right.refine_parameters(arguments)),
            Expression::Equals { left, right } => left
                .refine_parameters(arguments)
                .equals(right.refine_parameters(arguments)),
            Expression::GreaterOrEqual { left, right } => left
                .refine_parameters(arguments)
                .greater_or_equal(right.refine_parameters(arguments)),
            Expression::GreaterThan { left, right } => left
                .refine_parameters(arguments)
                .greater_than(right.refine_parameters(arguments)),
            Expression::Join { left, right, path } => left
                .refine_parameters(arguments)
                .join(right.refine_parameters(arguments), &path),
            Expression::LessOrEqual { left, right } => left
                .refine_parameters(arguments)
                .less_or_equal(right.refine_parameters(arguments)),
            Expression::LessThan { left, right } => left
                .refine_parameters(arguments)
                .less_than(right.refine_parameters(arguments)),
            Expression::Mul { left, right } => left
                .refine_parameters(arguments)
                .multiply(right.refine_parameters(arguments)),
            Expression::MulOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_parameters(arguments)
                .mul_overflows(right.refine_parameters(arguments), result_type.clone()),
            Expression::Ne { left, right } => left
                .refine_parameters(arguments)
                .not_equals(right.refine_parameters(arguments)),
            Expression::Neg { operand } => operand.refine_parameters(arguments).negate(),
            Expression::Not { operand } => operand.refine_parameters(arguments).logical_not(),
            Expression::Offset { left, right } => left
                .refine_parameters(arguments)
                .offset(right.refine_parameters(arguments)),
            Expression::Or { left, right } => left
                .refine_parameters(arguments)
                .or(right.refine_parameters(arguments)),
            Expression::Reference(path) => {
                // if the path is a parameter, the reference is an artifact of its type
                // and needs to be removed in the call context
                match path.as_ref() {
                    Path::LocalVariable { ordinal }
                        if 0 < *ordinal && *ordinal <= arguments.len() =>
                    {
                        arguments[*ordinal - 1].1.domain.clone()
                    }
                    _ => {
                        let refined_path = path.refine_parameters(arguments);
                        Rc::new(Expression::Reference(refined_path).into())
                    }
                }
            }
            Expression::Rem { left, right } => left
                .refine_parameters(arguments)
                .remainder(right.refine_parameters(arguments)),
            Expression::Shl { left, right } => left
                .refine_parameters(arguments)
                .shift_left(right.refine_parameters(arguments)),
            Expression::ShlOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_parameters(arguments)
                .shl_overflows(right.refine_parameters(arguments), result_type.clone()),
            Expression::Shr {
                left,
                right,
                result_type,
            } => left
                .refine_parameters(arguments)
                .shr(right.refine_parameters(arguments), result_type.clone()),
            Expression::ShrOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_parameters(arguments)
                .shr_overflows(right.refine_parameters(arguments), result_type.clone()),
            Expression::Sub { left, right } => left
                .refine_parameters(arguments)
                .subtract(right.refine_parameters(arguments)),
            Expression::SubOverflows {
                left,
                right,
                result_type,
            } => left
                .refine_parameters(arguments)
                .sub_overflows(right.refine_parameters(arguments), result_type.clone()),
            Expression::UnknownModelField { path, default } => {
                let refined_path = path.refine_parameters(arguments);
                Rc::new(
                    Expression::UnknownModelField {
                        path: refined_path,
                        default: default.clone(),
                    }
                    .into(),
                )
            }
            Expression::Variable { path, var_type } => {
                let refined_path = path.refine_parameters(arguments);
                if let Path::Constant { value } = refined_path.as_ref() {
                    value.domain.clone()
                } else {
                    Rc::new(
                        Expression::Variable {
                            path: refined_path,
                            var_type: var_type.clone(),
                        }
                        .into(),
                    )
                }
            }
            Expression::Widen { path, operand } => operand
                .refine_parameters(arguments)
                .widen(&path.refine_parameters(arguments)),
        }
    }

    /// Returns a domain that is simplified (refined) by using the current path conditions
    /// (conditions known to be true in the current context). If no refinement is possible
    /// the result is simply a clone of this domain.
    fn refine_with(&self, path_condition: &Self, depth: usize) -> Rc<AbstractDomain> {
        if depth >= k_limits::MAX_REFINE_DEPTH {
            return self.clone();
        }
        match &self.expression {
            Expression::Top | Expression::Bottom | Expression::AbstractHeapAddress(..) => {
                self.clone()
            }
            Expression::Add { left, right } => left
                .refine_with(path_condition, depth + 1)
                .addition(right.refine_with(path_condition, depth + 1)),
            Expression::AddOverflows {
                left,
                right,
                result_type,
            } => left.refine_with(path_condition, depth + 1).add_overflows(
                right.refine_with(path_condition, depth + 1),
                result_type.clone(),
            ),
            Expression::And { left, right } => {
                if path_condition.implies(&left) && path_condition.implies(&right) {
                    Rc::new(true.into())
                } else if path_condition.implies_not(&left) || path_condition.implies_not(&right) {
                    Rc::new(false.into())
                } else {
                    left.refine_with(path_condition, depth + 1)
                        .and(right.refine_with(path_condition, depth + 1))
                }
            }
            Expression::BitAnd { left, right } => left
                .refine_with(path_condition, depth + 1)
                .bit_and(right.refine_with(path_condition, depth + 1)),
            Expression::BitOr { left, right } => left
                .refine_with(path_condition, depth + 1)
                .bit_or(right.refine_with(path_condition, depth + 1)),
            Expression::BitXor { left, right } => left
                .refine_with(path_condition, depth + 1)
                .bit_xor(right.refine_with(path_condition, depth + 1)),
            Expression::Cast {
                operand,
                target_type,
            } => operand
                .refine_with(path_condition, depth + 1)
                .cast(target_type.clone()),
            Expression::CompileTimeConstant(..) => self.clone(),
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                if path_condition.implies(&condition) {
                    consequent.refine_with(path_condition, depth + 1)
                } else if path_condition.implies_not(&condition) {
                    alternate.refine_with(path_condition, depth + 1)
                } else {
                    let refined_condition = condition.refine_with(path_condition, depth + 1);
                    let refined_consequent = consequent.refine_with(path_condition, depth + 1);
                    let refined_alternate = alternate.refine_with(path_condition, depth + 1);
                    let refined_consequent =
                        refined_consequent.refine_with(&refined_condition, depth + 1);
                    let refined_alternate =
                        refined_alternate.refine_with(&refined_condition, depth + 1);
                    refined_condition.conditional_expression(refined_consequent, refined_alternate)
                }
            }
            Expression::Div { left, right } => left
                .refine_with(path_condition, depth + 1)
                .divide(right.refine_with(path_condition, depth + 1)),
            Expression::Equals { left, right } => left
                .refine_with(path_condition, depth + 1)
                .equals(right.refine_with(path_condition, depth + 1)),
            Expression::GreaterOrEqual { left, right } => left
                .refine_with(path_condition, depth + 1)
                .greater_or_equal(right.refine_with(path_condition, depth + 1)),
            Expression::GreaterThan { left, right } => left
                .refine_with(path_condition, depth + 1)
                .greater_than(right.refine_with(path_condition, depth + 1)),
            Expression::Join { left, right, path } => left
                .refine_with(path_condition, depth + 1)
                .join(right.refine_with(path_condition, depth + 1), &path),
            Expression::LessOrEqual { left, right } => left
                .refine_with(path_condition, depth + 1)
                .less_or_equal(right.refine_with(path_condition, depth + 1)),
            Expression::LessThan { left, right } => left
                .refine_with(path_condition, depth + 1)
                .less_than(right.refine_with(path_condition, depth + 1)),
            Expression::Mul { left, right } => left
                .refine_with(path_condition, depth + 1)
                .multiply(right.refine_with(path_condition, depth + 1)),
            Expression::MulOverflows {
                left,
                right,
                result_type,
            } => left.refine_with(path_condition, depth + 1).mul_overflows(
                right.refine_with(path_condition, depth + 1),
                result_type.clone(),
            ),
            Expression::Ne { left, right } => left
                .refine_with(path_condition, depth + 1)
                .not_equals(right.refine_with(path_condition, depth + 1)),
            Expression::Neg { operand } => operand.refine_with(path_condition, depth + 1).negate(),
            Expression::Not { operand } => {
                if path_condition.implies(&operand) {
                    Rc::new(false.into())
                } else if path_condition.implies_not(&operand) {
                    Rc::new(true.into())
                } else {
                    operand.refine_with(path_condition, depth + 1).logical_not()
                }
            }
            Expression::Offset { left, right } => left
                .refine_with(path_condition, depth + 1)
                .offset(right.refine_with(path_condition, depth + 1)),
            Expression::Or { left, right } => {
                if path_condition.implies(&left) || path_condition.implies(&right) {
                    Rc::new(true.into())
                } else if path_condition.implies_not(&left) && path_condition.implies_not(&right) {
                    Rc::new(false.into())
                } else {
                    left.refine_with(path_condition, depth + 1)
                        .or(right.refine_with(path_condition, depth + 1))
                }
            }
            Expression::Reference(..) => self.clone(),
            Expression::Rem { left, right } => left
                .refine_with(path_condition, depth + 1)
                .remainder(right.refine_with(path_condition, depth + 1)),
            Expression::Shl { left, right } => left
                .refine_with(path_condition, depth + 1)
                .shift_left(right.refine_with(path_condition, depth + 1)),
            Expression::ShlOverflows {
                left,
                right,
                result_type,
            } => left.refine_with(path_condition, depth + 1).shl_overflows(
                right.refine_with(path_condition, depth + 1),
                result_type.clone(),
            ),
            Expression::Shr {
                left,
                right,
                result_type,
            } => left.refine_with(path_condition, depth + 1).shr(
                right.refine_with(path_condition, depth + 1),
                result_type.clone(),
            ),
            Expression::ShrOverflows {
                left,
                right,
                result_type,
            } => left.refine_with(path_condition, depth + 1).shr_overflows(
                right.refine_with(path_condition, depth + 1),
                result_type.clone(),
            ),
            Expression::Sub { left, right } => left
                .refine_with(path_condition, depth + 1)
                .subtract(right.refine_with(path_condition, depth + 1)),
            Expression::SubOverflows {
                left,
                right,
                result_type,
            } => left.refine_with(path_condition, depth + 1).sub_overflows(
                right.refine_with(path_condition, depth + 1),
                result_type.clone(),
            ),
            Expression::UnknownModelField { .. } => self.clone(),
            Expression::Variable { .. } => {
                if path_condition.implies(&self) {
                    Rc::new(true.into())
                } else if path_condition.implies_not(&self) {
                    Rc::new(false.into())
                } else {
                    self.clone()
                }
            }
            Expression::Widen { path, operand } => {
                operand.refine_with(path_condition, depth + 1).widen(&path)
            }
        }
    }

    /// Tries to simplify operation(self, other) by constant folding or by distribution
    /// the operation over self and/or other.
    /// Returns operation(self, other) if no simplification is possible.
    fn try_to_simplify_binary_op(
        &self,
        other: Rc<AbstractDomain>,
        const_op: fn(&ConstantDomain, &ConstantDomain) -> ConstantDomain,
        operation: fn(Rc<AbstractDomain>, Rc<AbstractDomain>) -> Rc<AbstractDomain>,
    ) -> Rc<AbstractDomain> {
        match (&self.expression, &other.expression) {
            (Expression::CompileTimeConstant(v1), Expression::CompileTimeConstant(v2)) => {
                let result = const_op(v1, v2);
                if result == ConstantDomain::Bottom {
                    self.try_to_distribute_binary_op(other, operation)
                } else {
                    Rc::new(result.into())
                }
            }
            _ => self.try_to_distribute_binary_op(other, operation),
        }
    }

    /// Tries to distribute the operation over self and/or other.
    /// Return operation(self, other) if no simplification is possible.
    fn try_to_distribute_binary_op(
        &self,
        other: Rc<AbstractDomain>,
        operation: fn(Rc<AbstractDomain>, Rc<AbstractDomain>) -> Rc<AbstractDomain>,
    ) -> Rc<AbstractDomain> {
        if let ConditionalExpression {
            condition,
            consequent,
            alternate,
        } = &self.expression
        {
            return condition.conditional_expression(
                operation(consequent.clone(), other.clone()),
                operation(alternate.clone(), other.clone()),
            );
        };
        if let ConditionalExpression {
            condition,
            consequent,
            alternate,
        } = &other.expression
        {
            return condition.conditional_expression(
                operation(self.clone(), consequent.clone()),
                operation(self.clone(), alternate.clone()),
            );
        };
        if let Join { left, right, path } = &self.expression {
            return operation(left.clone(), other.clone())
                .join(operation(right.clone(), other), &path);
        }
        if let Join { left, right, path } = &other.expression {
            return operation(self.clone(), left.clone())
                .join(operation(self.clone(), right.clone()), &path);
        }
        match (&self.expression, &other.expression) {
            (Widen { .. }, _) => self.clone(),
            (_, Widen { .. }) => other,
            _ => operation(self.clone(), other),
        }
    }

    /// Returns a domain whose corresponding set of concrete values include all of the values
    /// corresponding to self and other. The set of values may be less precise (more inclusive) than
    /// the set returned by join. The chief requirement is that a small number of widen calls
    /// deterministically lead to a set of values that include of the values that could be stored
    /// in memory at the given path.
    fn widen(&self, path: &Rc<Path>) -> Rc<AbstractDomain> {
        match self.expression {
            Expression::Widen { .. }
            | Expression::CompileTimeConstant(..)
            | Expression::Reference(..)
            | Expression::Variable { .. } => self.clone(),
            _ => Rc::new(
                Expression::Widen {
                    path: path.clone(),
                    operand: self.clone(),
                }
                .into(),
            ),
        }
    }
}
