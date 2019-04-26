// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//

use crate::abstract_domains::AbstractDomain;
use crate::constant_domain::ConstantDomain;
use crate::expression::{Expression, ExpressionType};
use crate::smt_solver::SmtResult;
use crate::smt_solver::SmtSolver;

use crate::abstract_value::Path;
use lazy_static::lazy_static;
use log::debug;
use mirai_annotations::{checked_assume, checked_assume_eq};
use std::convert::TryFrom;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Mutex;
use z3_sys;

pub type Z3ExpressionType = z3_sys::Z3_ast;

lazy_static! {
    static ref Z3_MUTEX: Mutex<()> = Mutex::new(());
}

pub struct Z3Solver {
    z3_context: z3_sys::Z3_context,
    z3_solver: z3_sys::Z3_solver,
    any_sort: z3_sys::Z3_sort,
    bool_sort: z3_sys::Z3_sort,
    int_sort: z3_sys::Z3_sort,
    f32_sort: z3_sys::Z3_sort,
    f64_sort: z3_sys::Z3_sort,
    nearest_even: z3_sys::Z3_ast,
    zero: z3_sys::Z3_ast,
    one: z3_sys::Z3_ast,
    two: z3_sys::Z3_ast,
    empty_str: z3_sys::Z3_string,
}

impl Z3Solver {
    pub fn new() -> Z3Solver {
        unsafe {
            let _guard = Z3_MUTEX.lock().unwrap();
            let z3_sys_cfg = z3_sys::Z3_mk_config();
            let time_out = CString::new("timeout").unwrap().into_raw();
            let ms = CString::new("100").unwrap().into_raw();
            z3_sys::Z3_set_param_value(z3_sys_cfg, time_out, ms);

            let z3_context = z3_sys::Z3_mk_context(z3_sys_cfg);
            let z3_solver = z3_sys::Z3_mk_solver(z3_context);
            let empty_str = CString::new("").unwrap().into_raw();
            let symbol = z3_sys::Z3_mk_string_symbol(z3_context, empty_str);

            let any_sort = z3_sys::Z3_mk_uninterpreted_sort(z3_context, symbol);
            let bool_sort = { z3_sys::Z3_mk_bool_sort(z3_context) };
            let int_sort = { z3_sys::Z3_mk_int_sort(z3_context) };
            let f32_sort = { z3_sys::Z3_mk_fpa_sort_32(z3_context) };
            let f64_sort = { z3_sys::Z3_mk_fpa_sort_64(z3_context) };
            let nearest_even = { z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(z3_context) };
            let zero = { z3_sys::Z3_mk_int(z3_context, 0, int_sort) };
            let one = { z3_sys::Z3_mk_int(z3_context, 1, int_sort) };
            let two = { z3_sys::Z3_mk_int(z3_context, 2, int_sort) };

            Z3Solver {
                z3_context,
                z3_solver,
                any_sort,
                bool_sort,
                int_sort,
                f32_sort,
                f64_sort,
                nearest_even,
                zero,
                one,
                two,
                empty_str,
            }
        }
    }
}

impl Default for Z3Solver {
    fn default() -> Self {
        Z3Solver::new()
    }
}

impl SmtSolver<Z3ExpressionType> for Z3Solver {
    fn as_debug_string(&self, expression: &Z3ExpressionType) -> String {
        let _guard = Z3_MUTEX.lock().unwrap();
        unsafe {
            let debug_str_bytes = z3_sys::Z3_ast_to_string(self.z3_context, *expression);
            let debug_str = CStr::from_ptr(debug_str_bytes);
            String::from(debug_str.to_str().unwrap())
        }
    }

    fn assert(&mut self, expression: &Z3ExpressionType) {
        let _guard = Z3_MUTEX.lock().unwrap();
        unsafe {
            z3_sys::Z3_solver_assert(self.z3_context, self.z3_solver, *expression);
        }
    }

    fn backtrack(&mut self) {
        let _guard = Z3_MUTEX.lock().unwrap();
        unsafe {
            z3_sys::Z3_solver_pop(self.z3_context, self.z3_solver, 1);
        }
    }

    fn get_as_smt_predicate(&mut self, mirai_expression: &Expression) -> Z3ExpressionType {
        let _guard = Z3_MUTEX.lock().unwrap();
        self.get_as_bool_z3_ast(mirai_expression)
    }

    fn get_model_as_string(&self) -> String {
        let _guard = Z3_MUTEX.lock().unwrap();
        unsafe {
            let model = z3_sys::Z3_solver_get_model(self.z3_context, self.z3_solver);
            let debug_str_bytes = z3_sys::Z3_model_to_string(self.z3_context, model);
            let debug_str = CStr::from_ptr(debug_str_bytes);
            debug_str.to_str().unwrap().to_string()
        }
    }

    fn get_solver_state_as_string(&self) -> String {
        let _guard = Z3_MUTEX.lock().unwrap();
        unsafe {
            let debug_str_bytes = z3_sys::Z3_solver_to_string(self.z3_context, self.z3_solver);
            let debug_str = CStr::from_ptr(debug_str_bytes);
            debug_str.to_str().unwrap().to_string()
        }
    }

    fn set_backtrack_position(&mut self) {
        let _guard = Z3_MUTEX.lock().unwrap();
        unsafe {
            z3_sys::Z3_solver_push(self.z3_context, self.z3_solver);
        }
    }

    fn solve(&mut self) -> SmtResult {
        let _guard = Z3_MUTEX.lock().unwrap();
        unsafe {
            match z3_sys::Z3_solver_check(self.z3_context, self.z3_solver) {
                z3_sys::Z3_L_TRUE => SmtResult::Satisfiable,
                z3_sys::Z3_L_FALSE => SmtResult::Unsatisfiable,
                z3_sys::Z3_L_UNDEF | _ => SmtResult::Undefined,
            }
        }
    }
}

impl Z3Solver {
    #[allow(clippy::cognitive_complexity)]
    fn get_as_z3_ast(&mut self, expression: &Expression) -> z3_sys::Z3_ast {
        match expression {
            Expression::Add { .. }
            | Expression::AddOverflows { .. }
            | Expression::Div { .. }
            | Expression::Mul { .. }
            | Expression::MulOverflows { .. }
            | Expression::Rem { .. }
            | Expression::Sub { .. }
            | Expression::SubOverflows { .. } => self.get_as_numeric_z3_ast(expression).1,
            Expression::And { left, right } => {
                let left_ast = self.get_as_bool_z3_ast(&(**left).expression);
                let right_ast = self.get_as_bool_z3_ast(&(**right).expression);
                unsafe {
                    let tmp = vec![left_ast, right_ast];
                    z3_sys::Z3_mk_and(self.z3_context, 2, tmp.as_ptr())
                }
            }
            Expression::BitAnd { .. } | Expression::BitOr { .. } | Expression::BitXor { .. } => {
                self.get_as_bv_z3_ast(expression, 128)
            }
            Expression::Cast {
                operand,
                target_type,
            } => {
                if *target_type == ExpressionType::NonPrimitive
                    || *target_type == ExpressionType::Reference
                {
                    self.get_as_z3_ast(&operand.expression)
                } else if *target_type == ExpressionType::Bool {
                    self.get_as_bool_z3_ast(&operand.expression)
                } else {
                    self.get_as_numeric_z3_ast(&operand.expression).1
                }
            }
            Expression::CompileTimeConstant(const_domain) => self.get_constant_as_ast(const_domain),
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                let condition_ast = self.get_as_bool_z3_ast(&(**condition).expression);
                let consequent_ast = self.get_as_z3_ast(&(**consequent).expression);
                let alternate_ast = self.get_as_z3_ast(&(**alternate).expression);
                unsafe {
                    z3_sys::Z3_mk_ite(
                        self.z3_context,
                        condition_ast,
                        consequent_ast,
                        alternate_ast,
                    )
                }
            }
            Expression::Equals { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        z3_sys::Z3_mk_fpa_eq(self.z3_context, left_ast, right_ast)
                    } else {
                        z3_sys::Z3_mk_eq(self.z3_context, left_ast, right_ast)
                    }
                }
            }
            Expression::GreaterOrEqual { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        z3_sys::Z3_mk_fpa_geq(self.z3_context, left_ast, right_ast)
                    } else {
                        z3_sys::Z3_mk_ge(self.z3_context, left_ast, right_ast)
                    }
                }
            }
            Expression::GreaterThan { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        z3_sys::Z3_mk_fpa_gt(self.z3_context, left_ast, right_ast)
                    } else {
                        z3_sys::Z3_mk_gt(self.z3_context, left_ast, right_ast)
                    }
                }
            }
            Expression::LessOrEqual { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        z3_sys::Z3_mk_fpa_leq(self.z3_context, left_ast, right_ast)
                    } else {
                        z3_sys::Z3_mk_le(self.z3_context, left_ast, right_ast)
                    }
                }
            }
            Expression::LessThan { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        z3_sys::Z3_mk_fpa_lt(self.z3_context, left_ast, right_ast)
                    } else {
                        z3_sys::Z3_mk_lt(self.z3_context, left_ast, right_ast)
                    }
                }
            }
            Expression::Ne { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        let l = z3_sys::Z3_mk_fpa_is_nan(self.z3_context, left_ast);
                        let r = z3_sys::Z3_mk_fpa_is_nan(self.z3_context, right_ast);
                        let eq = z3_sys::Z3_mk_fpa_eq(self.z3_context, left_ast, right_ast);
                        let ne = z3_sys::Z3_mk_not(self.z3_context, eq);
                        let tmp = vec![l, r, ne];
                        z3_sys::Z3_mk_or(self.z3_context, 3, tmp.as_ptr())
                    } else {
                        z3_sys::Z3_mk_not(
                            self.z3_context,
                            z3_sys::Z3_mk_eq(self.z3_context, left_ast, right_ast),
                        )
                    }
                }
            }
            Expression::Neg { operand } => {
                let (is_float, operand_ast) = self.get_as_numeric_z3_ast(&(**operand).expression);
                unsafe {
                    if is_float {
                        z3_sys::Z3_mk_fpa_neg(self.z3_context, operand_ast)
                    } else {
                        z3_sys::Z3_mk_unary_minus(self.z3_context, operand_ast)
                    }
                }
            }
            Expression::Not { operand } => {
                let operand_ast = self.get_as_bool_z3_ast(&(**operand).expression);
                unsafe { z3_sys::Z3_mk_not(self.z3_context, operand_ast) }
            }
            Expression::Or { left, right } => {
                let left_ast = self.get_as_bool_z3_ast(&(**left).expression);
                let right_ast = self.get_as_bool_z3_ast(&(**right).expression);
                unsafe {
                    let tmp = vec![left_ast, right_ast];
                    z3_sys::Z3_mk_or(self.z3_context, 2, tmp.as_ptr())
                }
            }
            Expression::Reference(path) => {
                let path_str = CString::new(format!("&{:?}", path)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.any_sort)
                }
            }
            Expression::Shl { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, 128);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, 128);
                unsafe { z3_sys::Z3_mk_bvshl(self.z3_context, left_ast, right_ast) }
            }
            Expression::Shr {
                left,
                right,
                result_type,
            } => {
                let num_bits = u32::from(result_type.bit_length());
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe {
                    if result_type.is_signed_integer() {
                        z3_sys::Z3_mk_bvashr(self.z3_context, left_ast, right_ast)
                    } else {
                        z3_sys::Z3_mk_bvlshr(self.z3_context, left_ast, right_ast)
                    }
                }
            }
            Expression::ShlOverflows {
                right, result_type, ..
            }
            | Expression::ShrOverflows {
                right, result_type, ..
            } => {
                let (f, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume!(!f);
                let num_bits = i32::from(result_type.bit_length());
                unsafe {
                    let num_bits_val = z3_sys::Z3_mk_int(self.z3_context, num_bits, self.int_sort);
                    z3_sys::Z3_mk_ge(self.z3_context, right_ast, num_bits_val)
                }
            }
            Expression::Top | Expression::Bottom => unsafe {
                z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, self.bool_sort)
            },
            Expression::Variable { path, var_type } => {
                let path_str = CString::new(format!("{:?}", path)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    let sort = self.get_sort_for(var_type);
                    let ast = z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort);
                    if var_type.is_integer() {
                        let min_ast = self.get_constant_as_ast(&var_type.min_value());
                        let max_ast = self.get_constant_as_ast(&var_type.max_value());
                        let range_check = self.get_range_check(ast, min_ast, max_ast);

                        // Side-effecting the solver smells bad, but it hard to come up with an expression
                        // of type var_type that only take values that satisfies the range constraint.
                        // Since this is kind of a lazy variable declaration, it is probably OK.
                        z3_sys::Z3_solver_assert(self.z3_context, self.z3_solver, range_check);
                    }
                    ast
                }
            }
            Expression::Widen { path, operand } => {
                self.get_ast_for_widened(path, operand, operand.expression.infer_type())
            }
            Expression::Join { path, .. } => {
                let path_str = CString::new(format!("{:?}", path)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    let sort = self.get_sort_for(&expression.infer_type());
                    z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort)
                }
            }
            _ => unsafe {
                debug!("uninterpreted expression: {:?}", expression);
                let sort = self.get_sort_for(&expression.infer_type());
                z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, sort)
            },
        }
    }

    fn get_symbol_for(&self, path: &Path) -> z3_sys::Z3_symbol {
        let path_str = CString::new(format!("{:?}", path)).unwrap();
        unsafe { z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw()) }
    }

    fn get_sort_for(&self, var_type: &ExpressionType) -> z3_sys::Z3_sort {
        use self::ExpressionType::*;
        match var_type {
            Bool => self.bool_sort,
            Char | I8 | I16 | I32 | I64 | I128 | Isize | U8 | U16 | U32 | U64 | U128 | Usize => {
                self.int_sort
            }
            F32 => self.f32_sort,
            F64 => self.f64_sort,
            NonPrimitive | Reference => self.any_sort,
        }
    }

    fn get_ast_for_widened(
        &mut self,
        path: &Path,
        operand: &AbstractDomain,
        target_type: ExpressionType,
    ) -> z3_sys::Z3_ast {
        let path_str = CString::new(format!("{:?}", path)).unwrap();
        let sort = self.get_sort_for(&target_type);
        unsafe {
            let path_symbol = z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
            let ast = z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort);
            if target_type.is_integer() {
                let domain: AbstractDomain = AbstractDomain::from(Expression::Widen {
                    path: box path.clone(),
                    operand: box operand.clone(),
                });
                let interval = domain.get_as_interval();
                if !interval.is_bottom() {
                    if let Some(lower_bound) = interval.lower_bound() {
                        let lb = self.get_constant_as_ast(&ConstantDomain::I128(lower_bound));
                        let ge_lower_bound = z3_sys::Z3_mk_ge(self.z3_context, ast, lb);
                        // An interval is kind of like a type and the side-effect here is akin to the
                        // one in Expression::Variable. See the comments there for justification.
                        z3_sys::Z3_solver_assert(self.z3_context, self.z3_solver, ge_lower_bound);
                    }
                    if let Some(upper_bound) = interval.upper_bound() {
                        let ub = self.get_constant_as_ast(&ConstantDomain::I128(upper_bound));
                        let le_upper_bound = z3_sys::Z3_mk_le(self.z3_context, ast, ub);
                        // An interval is kind of like a type and the side-effect here is akin to the
                        // one in Expression::Variable. See the comments there for justification.
                        z3_sys::Z3_solver_assert(self.z3_context, self.z3_solver, le_upper_bound);
                    }
                }
            }
            ast
        }
    }

    fn get_constant_as_ast(&mut self, const_domain: &ConstantDomain) -> z3_sys::Z3_ast {
        match const_domain {
            ConstantDomain::Char(v) => unsafe {
                z3_sys::Z3_mk_int(self.z3_context, i32::from(*v as u16), self.int_sort)
            },
            ConstantDomain::False => unsafe { z3_sys::Z3_mk_false(self.z3_context) },
            ConstantDomain::I128(v) => unsafe {
                let v64 = i64::try_from(*v);
                if v64.is_ok() {
                    z3_sys::Z3_mk_int64(self.z3_context, v64.unwrap(), self.int_sort)
                } else {
                    let num_str = format!("{}", *v);
                    let c_string = CString::new(num_str).unwrap();
                    z3_sys::Z3_mk_numeral(self.z3_context, c_string.into_raw(), self.int_sort)
                }
            },
            ConstantDomain::F32(v) => unsafe {
                let fv = f32::from_bits(*v);
                z3_sys::Z3_mk_fpa_numeral_float(self.z3_context, fv, self.f32_sort)
            },
            ConstantDomain::F64(v) => unsafe {
                let fv = f64::from_bits(*v);
                z3_sys::Z3_mk_fpa_numeral_double(self.z3_context, fv, self.f64_sort)
            },
            ConstantDomain::U128(v) => unsafe {
                let v64 = u64::try_from(*v);
                if v64.is_ok() {
                    z3_sys::Z3_mk_unsigned_int64(self.z3_context, v64.unwrap(), self.int_sort)
                } else {
                    let num_str = format!("{}", *v);
                    let c_string = CString::new(num_str).unwrap();
                    z3_sys::Z3_mk_numeral(self.z3_context, c_string.into_raw(), self.int_sort)
                }
            },
            ConstantDomain::True => unsafe { z3_sys::Z3_mk_true(self.z3_context) },
            _ => unsafe {
                //info!("fresh constant for {:?}", const_domain);
                z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, self.any_sort)
            },
        }
    }

    fn get_range_check(
        &mut self,
        operand_ast: z3_sys::Z3_ast,
        min_ast: z3_sys::Z3_ast,
        max_ast: z3_sys::Z3_ast,
    ) -> z3_sys::Z3_ast {
        unsafe {
            let min_is_le = z3_sys::Z3_mk_le(self.z3_context, min_ast, operand_ast);
            let max_is_ge = z3_sys::Z3_mk_ge(self.z3_context, max_ast, operand_ast);
            let tmp = vec![min_is_le, max_is_ge];
            z3_sys::Z3_mk_and(self.z3_context, 2, tmp.as_ptr())
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn get_as_numeric_z3_ast(&mut self, expression: &Expression) -> (bool, z3_sys::Z3_ast) {
        match expression {
            Expression::Add { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        (
                            true,
                            z3_sys::Z3_mk_fpa_add(
                                self.z3_context,
                                self.nearest_even,
                                left_ast,
                                right_ast,
                            ),
                        )
                    } else {
                        let tmp = vec![left_ast, right_ast];
                        (false, z3_sys::Z3_mk_add(self.z3_context, 2, tmp.as_ptr()))
                    }
                }
            }
            Expression::AddOverflows {
                left,
                right,
                result_type,
            } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume!(!(lf || rf));
                unsafe {
                    let tmp = vec![left_ast, right_ast];
                    let sum = z3_sys::Z3_mk_add(self.z3_context, 2, tmp.as_ptr());
                    let min_ast = self.get_constant_as_ast(&result_type.min_value());
                    let min_is_gt = z3_sys::Z3_mk_gt(self.z3_context, min_ast, sum);
                    let max_ast = self.get_constant_as_ast(&result_type.max_value());
                    let max_is_lt = z3_sys::Z3_mk_lt(self.z3_context, max_ast, sum);
                    let tmp = vec![min_is_gt, max_is_lt];
                    let result_overflows = z3_sys::Z3_mk_or(self.z3_context, 2, tmp.as_ptr());
                    let left_in_range = self.get_range_check(left_ast, min_ast, max_ast);
                    let right_in_range = self.get_range_check(right_ast, min_ast, max_ast);
                    let tmp = vec![left_in_range, right_in_range, result_overflows];
                    (false, z3_sys::Z3_mk_and(self.z3_context, 3, tmp.as_ptr()))
                }
            }
            Expression::Div { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        (
                            true,
                            z3_sys::Z3_mk_fpa_div(
                                self.z3_context,
                                self.nearest_even,
                                left_ast,
                                right_ast,
                            ),
                        )
                    } else {
                        (
                            false,
                            z3_sys::Z3_mk_div(self.z3_context, left_ast, right_ast),
                        )
                    }
                }
            }
            Expression::Join { path, .. } => {
                let path_str = CString::new(format!("{:?}", path)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    let mut var_type = &expression.infer_type();
                    if !(var_type.is_integer() || var_type.is_floating_point_number()) {
                        var_type = &ExpressionType::I128
                    };
                    let sort = self.get_sort_for(var_type);
                    (
                        var_type.is_floating_point_number(),
                        z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort),
                    )
                }
            }
            Expression::Mul { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        (
                            true,
                            z3_sys::Z3_mk_fpa_mul(
                                self.z3_context,
                                self.nearest_even,
                                left_ast,
                                right_ast,
                            ),
                        )
                    } else {
                        let tmp = vec![left_ast, right_ast];
                        (false, z3_sys::Z3_mk_mul(self.z3_context, 2, tmp.as_ptr()))
                    }
                }
            }
            Expression::MulOverflows {
                left,
                right,
                result_type,
            } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume!(!(lf || rf));
                unsafe {
                    let tmp = vec![left_ast, right_ast];
                    let sum = z3_sys::Z3_mk_mul(self.z3_context, 2, tmp.as_ptr());
                    let min_ast = self.get_constant_as_ast(&result_type.min_value());
                    let min_is_gt = z3_sys::Z3_mk_gt(self.z3_context, min_ast, sum);
                    let max_ast = self.get_constant_as_ast(&result_type.max_value());
                    let max_is_lt = z3_sys::Z3_mk_lt(self.z3_context, max_ast, sum);
                    let tmp = vec![min_is_gt, max_is_lt];
                    let result_overflows = z3_sys::Z3_mk_or(self.z3_context, 2, tmp.as_ptr());
                    let left_in_range = self.get_range_check(left_ast, min_ast, max_ast);
                    let right_in_range = self.get_range_check(right_ast, min_ast, max_ast);
                    let tmp = vec![left_in_range, right_in_range, result_overflows];
                    (false, z3_sys::Z3_mk_and(self.z3_context, 3, tmp.as_ptr()))
                }
            }
            Expression::Rem { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        (
                            true,
                            z3_sys::Z3_mk_fpa_rem(self.z3_context, left_ast, right_ast),
                        )
                    } else {
                        (
                            false,
                            z3_sys::Z3_mk_rem(self.z3_context, left_ast, right_ast),
                        )
                    }
                }
            }
            Expression::Sub { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume_eq!(lf, rf);
                unsafe {
                    if lf {
                        (
                            true,
                            z3_sys::Z3_mk_fpa_sub(
                                self.z3_context,
                                self.nearest_even,
                                left_ast,
                                right_ast,
                            ),
                        )
                    } else {
                        let tmp = vec![left_ast, right_ast];
                        (false, z3_sys::Z3_mk_sub(self.z3_context, 2, tmp.as_ptr()))
                    }
                }
            }
            Expression::SubOverflows {
                left,
                right,
                result_type,
            } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume!(!(lf || rf));
                unsafe {
                    let tmp = vec![left_ast, right_ast];
                    let sum = z3_sys::Z3_mk_sub(self.z3_context, 2, tmp.as_ptr());
                    let min_ast = self.get_constant_as_ast(&result_type.min_value());
                    let min_is_gt = z3_sys::Z3_mk_gt(self.z3_context, min_ast, sum);
                    let max_ast = self.get_constant_as_ast(&result_type.max_value());
                    let max_is_lt = z3_sys::Z3_mk_lt(self.z3_context, max_ast, sum);
                    let tmp = vec![min_is_gt, max_is_lt];
                    let result_overflows = z3_sys::Z3_mk_or(self.z3_context, 2, tmp.as_ptr());
                    let left_in_range = self.get_range_check(left_ast, min_ast, max_ast);
                    let right_in_range = self.get_range_check(right_ast, min_ast, max_ast);
                    let tmp = vec![left_in_range, right_in_range, result_overflows];
                    (false, z3_sys::Z3_mk_and(self.z3_context, 3, tmp.as_ptr()))
                }
            }
            Expression::And { .. }
            | Expression::Equals { .. }
            | Expression::GreaterOrEqual { .. }
            | Expression::GreaterThan { .. }
            | Expression::LessOrEqual { .. }
            | Expression::LessThan { .. }
            | Expression::Ne { .. }
            | Expression::Not { .. }
            | Expression::Or { .. } => {
                let ast = self.get_as_z3_ast(expression);
                unsafe {
                    (
                        false,
                        z3_sys::Z3_mk_ite(self.z3_context, ast, self.one, self.zero),
                    )
                }
            }
            Expression::BitAnd { .. } | Expression::BitOr { .. } | Expression::BitXor { .. } => {
                let ast = self.get_as_bv_z3_ast(expression, 128);
                unsafe { (false, z3_sys::Z3_mk_bv2int(self.z3_context, ast, false)) }
            }
            Expression::Cast { target_type, .. } => {
                let path_str = CString::new(format!("{:?}", expression)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    match target_type {
                        ExpressionType::F32 => (
                            true,
                            z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.f32_sort),
                        ),
                        ExpressionType::F64 => (
                            true,
                            z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.f64_sort),
                        ),
                        _ => (
                            false,
                            z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.int_sort),
                        ),
                    }
                }
            }
            Expression::CompileTimeConstant(const_domain) => match const_domain {
                ConstantDomain::False => unsafe {
                    (false, z3_sys::Z3_mk_int(self.z3_context, 0, self.int_sort))
                },
                ConstantDomain::True => unsafe {
                    (false, z3_sys::Z3_mk_int(self.z3_context, 1, self.int_sort))
                },
                ConstantDomain::F32(..) | ConstantDomain::F64(..) => {
                    (true, self.get_as_z3_ast(expression))
                }
                _ => (false, self.get_as_z3_ast(expression)),
            },
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                debug!("{:?}", expression);
                let condition_ast = self.get_as_bool_z3_ast(&(**condition).expression);
                let (cf, consequent_ast) = self.get_as_numeric_z3_ast(&(**consequent).expression);
                let (af, alternate_ast) = self.get_as_numeric_z3_ast(&(**alternate).expression);
                checked_assume_eq!(cf, af);
                unsafe {
                    (
                        cf,
                        z3_sys::Z3_mk_ite(
                            self.z3_context,
                            condition_ast,
                            consequent_ast,
                            alternate_ast,
                        ),
                    )
                }
            }
            Expression::Neg { operand } => {
                let (is_float, operand_ast) = self.get_as_numeric_z3_ast(&(**operand).expression);
                unsafe {
                    if is_float {
                        (true, z3_sys::Z3_mk_fpa_neg(self.z3_context, operand_ast))
                    } else {
                        (
                            false,
                            z3_sys::Z3_mk_unary_minus(self.z3_context, operand_ast),
                        )
                    }
                }
            }
            Expression::Reference(path) => unsafe {
                let path_symbol = self.get_symbol_for(path);
                (
                    false,
                    z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.int_sort),
                )
            },
            Expression::Shl { left, right } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume!(!(lf || rf));
                unsafe {
                    let right_power = z3_sys::Z3_mk_power(self.z3_context, self.two, right_ast);
                    let tmp = vec![left_ast, right_power];
                    (false, z3_sys::Z3_mk_mul(self.z3_context, 2, tmp.as_ptr()))
                }
            }
            Expression::Shr { left, right, .. } => {
                let (lf, left_ast) = self.get_as_numeric_z3_ast(&(**left).expression);
                let (rf, right_ast) = self.get_as_numeric_z3_ast(&(**right).expression);
                checked_assume!(!(lf || rf));
                unsafe {
                    let right_power = z3_sys::Z3_mk_power(self.z3_context, self.two, right_ast);
                    (
                        false,
                        z3_sys::Z3_mk_div(self.z3_context, left_ast, right_power),
                    )
                }
            }
            Expression::Top | Expression::Bottom => unsafe {
                (
                    false,
                    z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, self.int_sort),
                )
            },
            Expression::Variable { path, var_type } => {
                use self::ExpressionType::*;
                match var_type {
                    Bool | Reference | NonPrimitive => unsafe {
                        let path_symbol = self.get_symbol_for(&**path);
                        (
                            false,
                            z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.int_sort),
                        )
                    },
                    F32 | F64 => (true, self.get_as_z3_ast(expression)),
                    _ => (false, self.get_as_z3_ast(expression)),
                }
            }
            Expression::Widen { path, operand } => {
                use self::ExpressionType::*;
                let expr_type = operand.expression.infer_type();
                let expr_type = match expr_type {
                    Bool | Reference | NonPrimitive => ExpressionType::I128,
                    _ => expr_type,
                };
                let is_float = expr_type.is_floating_point_number();
                let ast = self.get_ast_for_widened(path, operand, expr_type);
                (is_float, ast)
            }
            _ => (false, self.get_as_z3_ast(expression)),
        }
    }

    fn get_as_bool_z3_ast(&mut self, expression: &Expression) -> z3_sys::Z3_ast {
        match expression {
            Expression::BitAnd { .. } | Expression::BitOr { .. } | Expression::BitXor { .. } => {
                //todo: get operands as booleans and treat this operands as logical
                let bv = self.get_as_bv_z3_ast(expression, 128);
                unsafe {
                    let i = z3_sys::Z3_mk_bv2int(self.z3_context, bv, false);
                    let f = z3_sys::Z3_mk_eq(self.z3_context, i, self.zero);
                    z3_sys::Z3_mk_not(self.z3_context, f)
                }
            }
            Expression::CompileTimeConstant(const_domain) => match const_domain {
                ConstantDomain::False => unsafe { z3_sys::Z3_mk_false(self.z3_context) },
                ConstantDomain::True => unsafe { z3_sys::Z3_mk_true(self.z3_context) },
                ConstantDomain::U128(val) => unsafe {
                    if *val == 0 {
                        z3_sys::Z3_mk_false(self.z3_context)
                    } else {
                        z3_sys::Z3_mk_true(self.z3_context)
                    }
                },
                _ => self.get_as_z3_ast(expression),
            },
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                let condition_ast = self.get_as_bool_z3_ast(&(**condition).expression);
                let consequent_ast = self.get_as_bool_z3_ast(&(**consequent).expression);
                let alternate_ast = self.get_as_bool_z3_ast(&(**alternate).expression);
                unsafe {
                    z3_sys::Z3_mk_ite(
                        self.z3_context,
                        condition_ast,
                        consequent_ast,
                        alternate_ast,
                    )
                }
            }
            Expression::Reference(path) => {
                let path_str = CString::new(format!("&{:?}", path)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.bool_sort)
                }
            }
            Expression::Top | Expression::Bottom => unsafe {
                z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, self.bool_sort)
            },
            Expression::Variable { path, var_type } => {
                if *var_type != ExpressionType::Bool {
                    debug!("path {:?}, type {:?}", path, var_type);
                }
                let path_str = CString::new(format!("{:?}", path)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.bool_sort)
                }
            }
            Expression::Widen { path, operand } => {
                self.get_ast_for_widened(path, operand, ExpressionType::Bool)
            }
            _ => self.get_as_z3_ast(expression),
        }
    }

    fn get_as_bv_z3_ast(&mut self, expression: &Expression, num_bits: u32) -> z3_sys::Z3_ast {
        match expression {
            Expression::Add { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvadd(self.z3_context, left_ast, right_ast) }
            }
            Expression::AddOverflows {
                left,
                right,
                result_type,
            } => {
                let num_bits = u32::from(result_type.bit_length());
                let is_signed = result_type.is_signed_integer();
                let left_bv = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_bv = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe {
                    let does_not_overflow = z3_sys::Z3_mk_bvadd_no_overflow(
                        self.z3_context,
                        left_bv,
                        right_bv,
                        is_signed,
                    );
                    if is_signed {
                        let does_not_underflow =
                            z3_sys::Z3_mk_bvadd_no_underflow(self.z3_context, left_bv, right_bv);
                        let tmp = vec![does_not_overflow, does_not_underflow];
                        let stays_in_range = z3_sys::Z3_mk_and(self.z3_context, 2, tmp.as_ptr());
                        z3_sys::Z3_mk_not(self.z3_context, stays_in_range)
                    } else {
                        z3_sys::Z3_mk_not(self.z3_context, does_not_overflow)
                    }
                }
            }
            Expression::Div { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvsdiv(self.z3_context, left_ast, right_ast) }
            }
            Expression::Mul { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvmul(self.z3_context, left_ast, right_ast) }
            }
            Expression::MulOverflows {
                left,
                right,
                result_type,
            } => {
                let num_bits = u32::from(result_type.bit_length());
                let is_signed = result_type.is_signed_integer();
                let left_bv = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_bv = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe {
                    let does_not_overflow = z3_sys::Z3_mk_bvmul_no_overflow(
                        self.z3_context,
                        left_bv,
                        right_bv,
                        is_signed,
                    );
                    if is_signed {
                        let does_not_underflow =
                            z3_sys::Z3_mk_bvmul_no_underflow(self.z3_context, left_bv, right_bv);
                        let tmp = vec![does_not_overflow, does_not_underflow];
                        let stays_in_range = z3_sys::Z3_mk_and(self.z3_context, 2, tmp.as_ptr());
                        z3_sys::Z3_mk_not(self.z3_context, stays_in_range)
                    } else {
                        z3_sys::Z3_mk_not(self.z3_context, does_not_overflow)
                    }
                }
            }
            Expression::Rem { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvsrem(self.z3_context, left_ast, right_ast) }
            }
            Expression::Sub { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvsub(self.z3_context, left_ast, right_ast) }
            }
            Expression::SubOverflows {
                left,
                right,
                result_type,
            } => {
                let num_bits = u32::from(result_type.bit_length());
                let is_signed = result_type.is_signed_integer();
                let left_bv = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_bv = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe {
                    let does_not_underflow = z3_sys::Z3_mk_bvsub_no_underflow(
                        self.z3_context,
                        left_bv,
                        right_bv,
                        is_signed,
                    );
                    if is_signed {
                        let does_not_overflow =
                            z3_sys::Z3_mk_bvsub_no_overflow(self.z3_context, left_bv, right_bv);
                        let tmp = vec![does_not_overflow, does_not_underflow];
                        let stays_in_range = z3_sys::Z3_mk_and(self.z3_context, 2, tmp.as_ptr());
                        z3_sys::Z3_mk_not(self.z3_context, stays_in_range)
                    } else {
                        z3_sys::Z3_mk_not(self.z3_context, does_not_underflow)
                    }
                }
            }
            Expression::And { .. }
            | Expression::Equals { .. }
            | Expression::GreaterOrEqual { .. }
            | Expression::GreaterThan { .. }
            | Expression::LessOrEqual { .. }
            | Expression::LessThan { .. }
            | Expression::Ne { .. }
            | Expression::Not { .. }
            | Expression::Or { .. } => {
                let ast = self.get_as_z3_ast(expression);
                // ast results in a boolean, but we want a bit vector.
                unsafe {
                    //todo: use a different primitive
                    let bv_one = z3_sys::Z3_mk_int2bv(self.z3_context, num_bits, self.one);
                    let bv_zero = z3_sys::Z3_mk_int2bv(self.z3_context, num_bits, self.zero);
                    z3_sys::Z3_mk_ite(self.z3_context, ast, bv_one, bv_zero)
                }
            }
            Expression::BitAnd { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvand(self.z3_context, left_ast, right_ast) }
            }
            Expression::BitOr { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvor(self.z3_context, left_ast, right_ast) }
            }
            Expression::BitXor { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvxor(self.z3_context, left_ast, right_ast) }
            }
            Expression::Cast { .. } => {
                let path_str = CString::new(format!("{:?}", expression)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    let sort = z3_sys::Z3_mk_bv_sort(self.z3_context, num_bits);
                    z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort)
                }
            }
            Expression::CompileTimeConstant(const_domain) => match const_domain {
                ConstantDomain::Char(v) => unsafe {
                    z3_sys::Z3_mk_bv_numeral(self.z3_context, 16, v as *const char as *const bool)
                },
                ConstantDomain::False => unsafe {
                    let v = false;
                    z3_sys::Z3_mk_bv_numeral(self.z3_context, 1, &v as *const bool)
                },
                ConstantDomain::F32(v) => unsafe {
                    let fv = f32::from_bits(*v);
                    z3_sys::Z3_mk_fpa_numeral_float(self.z3_context, fv, self.f32_sort)
                },
                ConstantDomain::F64(v) => unsafe {
                    let fv = f64::from_bits(*v);
                    z3_sys::Z3_mk_fpa_numeral_double(self.z3_context, fv, self.f64_sort)
                },
                ConstantDomain::I128(v) => unsafe {
                    z3_sys::Z3_mk_bv_numeral(self.z3_context, 128, v as *const i128 as *const bool)
                },
                ConstantDomain::U128(v) => unsafe {
                    z3_sys::Z3_mk_bv_numeral(self.z3_context, 128, v as *const u128 as *const bool)
                },
                ConstantDomain::True => unsafe {
                    let v = true;
                    z3_sys::Z3_mk_bv_numeral(self.z3_context, 1, &v as *const bool)
                },
                _ => unsafe {
                    let sort = z3_sys::Z3_mk_bv_sort(self.z3_context, num_bits);
                    z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, sort)
                },
            },
            Expression::ConditionalExpression {
                condition,
                consequent,
                alternate,
            } => {
                let condition_ast = self.get_as_bool_z3_ast(&(**condition).expression);
                let consequent_ast = self.get_as_bv_z3_ast(&(**consequent).expression, num_bits);
                let alternate_ast = self.get_as_bv_z3_ast(&(**alternate).expression, num_bits);
                unsafe {
                    z3_sys::Z3_mk_ite(
                        self.z3_context,
                        condition_ast,
                        consequent_ast,
                        alternate_ast,
                    )
                }
            }
            Expression::Join { path, .. } => unsafe {
                let sort = z3_sys::Z3_mk_bv_sort(self.z3_context, num_bits);
                let path_symbol = self.get_symbol_for(&**path);
                z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort)
            },
            Expression::Reference(path) => unsafe {
                let sort = z3_sys::Z3_mk_bv_sort(self.z3_context, num_bits);
                let path_symbol = self.get_symbol_for(path);
                z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort)
            },
            Expression::Shl { left, right } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe { z3_sys::Z3_mk_bvshl(self.z3_context, left_ast, right_ast) }
            }
            Expression::Shr {
                left,
                right,
                result_type,
            } => {
                let left_ast = self.get_as_bv_z3_ast(&(**left).expression, num_bits);
                let right_ast = self.get_as_bv_z3_ast(&(**right).expression, num_bits);
                unsafe {
                    if result_type.is_signed_integer() {
                        z3_sys::Z3_mk_bvashr(self.z3_context, left_ast, right_ast)
                    } else {
                        z3_sys::Z3_mk_bvlshr(self.z3_context, left_ast, right_ast)
                    }
                }
            }
            Expression::Top | Expression::Bottom => unsafe {
                let sort = z3_sys::Z3_mk_bv_sort(self.z3_context, num_bits);
                z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, sort)
            },
            Expression::Variable { path, var_type } => {
                use self::ExpressionType::*;
                let path_str = CString::new(format!("{:?}", path)).unwrap();
                unsafe {
                    let path_symbol =
                        z3_sys::Z3_mk_string_symbol(self.z3_context, path_str.into_raw());
                    let sort =
                        z3_sys::Z3_mk_bv_sort(self.z3_context, u32::from(var_type.bit_length()));
                    match var_type {
                        Bool | Char | I8 | I16 | I32 | I64 | I128 | Isize | U8 | U16 | U32
                        | U64 | U128 | Usize | Reference => {
                            z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort)
                        }
                        F32 => z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.f32_sort),
                        F64 => z3_sys::Z3_mk_const(self.z3_context, path_symbol, self.f64_sort),
                        NonPrimitive => {
                            z3_sys::Z3_mk_fresh_const(self.z3_context, self.empty_str, sort)
                        }
                    }
                }
            }
            Expression::Widen { path, operand } => {
                use self::ExpressionType::*;
                let expr_type = operand.expression.infer_type();
                let expr_type = match expr_type {
                    Bool | Reference | NonPrimitive => ExpressionType::I128,
                    _ => expr_type,
                };
                let path_symbol = self.get_symbol_for(&**path);
                unsafe {
                    let sort =
                        z3_sys::Z3_mk_bv_sort(self.z3_context, u32::from(expr_type.bit_length()));
                    z3_sys::Z3_mk_const(self.z3_context, path_symbol, sort)
                }
            }
            _ => self.get_as_z3_ast(expression),
        }
    }
}
