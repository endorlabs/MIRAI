// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use abstract_domains::{AbstractDomains, ExpressionDomain};
use abstract_value::{self, AbstractValue, Path, PathSelector};
use constant_value::{ConstantValue, ConstantValueCache};
use environment::Environment;
use rpds::List;
use rustc::session::Session;
use rustc::ty::{Const, LazyConst, Ty, TyCtxt, TyKind, UserTypeAnnotationIndex};
use rustc::{hir, mir};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::TryFrom;
use summaries;
use summaries::{PersistentSummaryCache, Summary};
use syntax::errors::{Diagnostic, DiagnosticBuilder};
use syntax_pos;

pub struct MirVisitorCrateContext<'a, 'b: 'a, 'tcx: 'b> {
    /// A place where diagnostic messages can be buffered by the test harness.
    pub buffered_diagnostics: &'a mut Vec<Diagnostic>,
    /// A call back that the test harness can use to buffer the diagnostic message.
    /// By default this just calls emit on the diagnostic.
    pub emit_diagnostic: fn(&mut DiagnosticBuilder, buf: &mut Vec<Diagnostic>) -> (),
    pub session: &'tcx Session,
    pub tcx: TyCtxt<'b, 'tcx, 'tcx>,
    pub def_id: hir::def_id::DefId,
    pub mir: &'a mir::Mir<'tcx>,
    pub constant_value_cache: &'a mut ConstantValueCache,
    pub summary_cache: &'a mut PersistentSummaryCache<'b, 'tcx>,
}

/// Holds the state for the MIR test visitor.
pub struct MirVisitor<'a, 'b: 'a, 'tcx: 'b> {
    buffered_diagnostics: &'a mut Vec<Diagnostic>,
    emit_diagnostic: fn(&mut DiagnosticBuilder, buf: &mut Vec<Diagnostic>) -> (),
    session: &'tcx Session,
    tcx: TyCtxt<'b, 'tcx, 'tcx>,
    def_id: hir::def_id::DefId,
    mir: &'a mir::Mir<'tcx>,
    constant_value_cache: &'a mut ConstantValueCache,
    summary_cache: &'a mut PersistentSummaryCache<'b, 'tcx>,

    check_for_errors: bool,
    current_environment: Environment,
    current_location: mir::Location,
    current_span: syntax_pos::Span,
    exit_environment: Environment,
    heap_addresses: HashMap<mir::Location, AbstractValue>,
    inferred_preconditions: List<AbstractValue>,
    post_conditions: List<AbstractValue>,
    preconditions: List<AbstractValue>,
    unwind_condition: List<AbstractValue>,
}

/// A visitor that simply traverses enough of the MIR associated with a particular code body
/// so that we can test a call to every default implementation of the MirVisitor trait.
impl<'a, 'b: 'a, 'tcx: 'b> MirVisitor<'a, 'b, 'tcx> {
    pub fn new(crate_context: MirVisitorCrateContext<'a, 'b, 'tcx>) -> MirVisitor<'a, 'b, 'tcx> {
        MirVisitor {
            buffered_diagnostics: crate_context.buffered_diagnostics,
            emit_diagnostic: crate_context.emit_diagnostic,
            session: crate_context.session,
            tcx: crate_context.tcx,
            def_id: crate_context.def_id,
            mir: crate_context.mir,
            constant_value_cache: crate_context.constant_value_cache,
            summary_cache: crate_context.summary_cache,

            check_for_errors: false,
            current_environment: Environment::default(),
            current_location: mir::Location::START,
            current_span: syntax_pos::DUMMY_SP,
            exit_environment: Environment::default(),
            heap_addresses: HashMap::default(),
            inferred_preconditions: List::new(),
            post_conditions: List::new(),
            preconditions: List::new(),
            unwind_condition: List::new(),
        }
    }

    /// Use the local and global environments to resolve Path to an abstract value.
    /// For now, statics and promoted constants just return Top.
    /// If a local value cannot be found the result is Bottom.
    fn lookup_path_and_refine_result(&mut self, path: Path) -> AbstractValue {
        let refined_val = {
            let val_at_ref = self.try_to_deref(&path);
            match val_at_ref {
                Some(val) => {
                    val.refine_with(&self.current_environment.entry_condition, self.current_span)
                }
                None => {
                    let bottom = abstract_value::BOTTOM;
                    let local_val = self.current_environment.value_at(&path).unwrap_or(&bottom);
                    local_val
                        .refine_with(&self.current_environment.entry_condition, self.current_span)
                }
            }
        };
        if refined_val.is_bottom() {
            // Not found locally, so try statics and promoted constants
            let mut val: Option<AbstractValue> = None;
            if let Path::StaticVariable { ref name } = path {
                let summary = self.summary_cache.get_persistent_summary_for(name);
                val = Some(summary.result.unwrap_or_else(|| abstract_value::TOP));
            }
            if let Path::PromotedConstant { .. } = path {
                // todo: #34 provide a crate level environment for storing promoted constants
                val = Some(abstract_value::TOP);
            }
            // This bit of awkwardness is needed so that we can move path into the environment.
            // Hopefully LLVM will optimize this away.
            if let Some(val) = val {
                self.current_environment.update_value_at(path, val.clone());
                return val;
            }
        }
        refined_val
    }

    /// For PathSelector::Deref, lookup the reference value using the qualifier, and then dereference that.
    /// Otherwise return None.
    fn try_to_deref(&mut self, path: &Path) -> Option<AbstractValue> {
        if let Path::QualifiedPath {
            ref qualifier,
            ref selector,
        } = path
        {
            if let PathSelector::Deref = **selector {
                let ref_val = self.lookup_path_and_refine_result((**qualifier).clone());
                return Some(self.dereference(ref_val));
            }
        }
        None
    }

    /// Given a value that is known to be of the form &path, return the value of path.
    /// Otherwise just return TOP.
    fn dereference(&mut self, reference: AbstractValue) -> AbstractValue {
        match reference.value.expression_domain {
            ExpressionDomain::Reference(path) => self.lookup_path_and_refine_result(path.clone()),
            _ => abstract_value::TOP,
        }
    }

    /// Analyze the body and store a summary of its behavior in self.summary_cache.
    pub fn visit_body(&mut self) {
        debug!("visit_body({:?})", self.def_id);
        // in_state[bb] is the join (or widening) of the out_state values of each predecessor of bb
        let mut in_state: HashMap<mir::BasicBlock, Environment> = HashMap::new();
        // out_state[bb] is the environment that results from analyzing block bb, given in_state[bb]
        let mut out_state: HashMap<mir::BasicBlock, Environment> = HashMap::new();
        for bb in self.mir.basic_blocks().indices() {
            in_state.insert(bb, Environment::default());
            out_state.insert(bb, Environment::default());
        }
        // The entry block has no predecessors and its initial state is the function parameters.
        let first_state = Environment::with_parameters(self.mir.arg_count);

        // Compute a fixed point, which is a value of out_state that will not grow with more iterations.
        let mut changed = true;
        let mut iteration_count = 0;
        while changed {
            changed = false;
            for bb in self.mir.basic_blocks().indices() {
                // Merge output states of predecessors of bb
                let i_state = if bb.index() == 0 {
                    first_state.clone()
                } else {
                    let mut predecessor_states_and_conditions: Vec<(
                        &Environment,
                        Option<&AbstractValue>,
                    )> = self
                        .mir
                        .predecessors_for(bb)
                        .iter()
                        .map(|pred_bb| {
                            let pred_state = &out_state[pred_bb];
                            let pred_exit_condition = pred_state.exit_conditions.get(&bb);
                            (pred_state, pred_exit_condition)
                        })
                        .filter(|(_, pred_exit_condition)| pred_exit_condition.is_some())
                        .collect();
                    if predecessor_states_and_conditions.is_empty() {
                        // unreachable block
                        let mut i_state = in_state[&bb].clone();
                        i_state.entry_condition = abstract_value::FALSE;
                        i_state
                    } else {
                        // We want to do right associative operations and that is easier if we reverse.
                        predecessor_states_and_conditions.reverse();
                        let (p_state, pred_exit_condition) = predecessor_states_and_conditions[0];
                        let mut i_state = p_state.clone();
                        i_state.entry_condition = pred_exit_condition
                            .unwrap()
                            .with_provenance(self.current_span);
                        for (p_state, pred_exit_condition) in
                            predecessor_states_and_conditions.iter().skip(1)
                        {
                            let join_condition = pred_exit_condition.unwrap();
                            // Once all paths have already been analyzed for a second time (iteration_count >= 3)
                            // we to abstract more aggressively in order to ensure reaching a fixed point.
                            let mut j_state = if iteration_count < 3 {
                                p_state.join(&i_state, join_condition)
                            } else {
                                p_state.widen(&i_state, join_condition)
                            };
                            j_state.entry_condition = join_condition
                                .or(&i_state.entry_condition, Some(self.current_span));
                            i_state = j_state;
                        }
                        i_state
                    }
                };
                // Analyze the basic block.
                in_state.insert(bb, i_state.clone());
                self.current_environment = i_state;
                self.visit_basic_block(bb);

                // Check for a fixed point.
                if !self.current_environment.subset(&out_state[&bb]) {
                    // There is some path for which self.current_environment.value_at(path) includes
                    // a value this is not present in out_state[bb].value_at(path), so any block
                    // that used out_state[bb] as part of its input state now needs to get reanalyzed.
                    out_state.insert(bb, self.current_environment.clone());
                    changed = true;
                } else {
                    // If the environment at the end of this block does not have any new values,
                    // we have reached a fixed point for this block.
                }
            }
            iteration_count += 1;
        }

        // Now traverse the blocks again, doing checks and emitting diagnostics.
        // in_state[bb] is now complete for every basic block bb in the body.
        debug!(
            "Fixed point loop took {} iterations, now checking for errors.",
            iteration_count
        );
        self.check_for_errors = true;
        for bb in self.mir.basic_blocks().indices() {
            let i_state = (&in_state[&bb]).clone();
            self.current_environment = i_state;
            self.visit_basic_block(bb);
        }

        // Now create a summary of the body that can be in-lined into call sites.
        let summary = summaries::summarize(
            &self.exit_environment,
            &self.inferred_preconditions,
            &self.preconditions,
            &self.post_conditions,
            &self.unwind_condition,
        );
        self.summary_cache.set_summary_for(self.def_id, summary);
    }

    /// Visits each statement in order and then visits the terminator.
    fn visit_basic_block(&mut self, bb: mir::BasicBlock) {
        debug!("visit_basic_block({:?})", bb);

        let mir::BasicBlockData {
            ref statements,
            ref terminator,
            ..
        } = &self.mir[bb];
        let mut location = bb.start_location();
        let terminator_index = statements.len();

        while location.statement_index < terminator_index {
            self.visit_statement(location, &statements[location.statement_index]);
            location.statement_index += 1;
        }

        if let Some(mir::Terminator {
            ref source_info,
            ref kind,
        }) = *terminator
        {
            self.visit_terminator(*source_info, kind);
        }
    }

    /// Calls a specialized visitor for each kind of statement.
    fn visit_statement(&mut self, location: mir::Location, statement: &mir::Statement) {
        self.current_location = location;
        let mir::Statement { kind, source_info } = statement;
        debug!("{:?}", source_info);
        self.current_span = source_info.span;
        match kind {
            mir::StatementKind::Assign(place, rvalue) => self.visit_assign(place, rvalue.borrow()),
            mir::StatementKind::FakeRead(..) => unreachable!(),
            mir::StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => self.visit_set_discriminant(place, *variant_index),
            mir::StatementKind::StorageLive(local) => self.visit_storage_live(*local),
            mir::StatementKind::StorageDead(local) => self.visit_storage_dead(*local),
            mir::StatementKind::InlineAsm {
                asm,
                outputs,
                inputs,
            } => self.visit_inline_asm(asm, outputs, inputs),
            mir::StatementKind::Retag(retag_kind, place) => self.visit_retag(*retag_kind, place),
            mir::StatementKind::AscribeUserType(..) => unreachable!(),
            mir::StatementKind::Nop => return,
        }
    }

    /// Write the RHS Rvalue to the LHS Place.
    fn visit_assign(&mut self, place: &mir::Place, rvalue: &mir::Rvalue) {
        debug!(
            "default visit_assign(place: {:?}, rvalue: {:?})",
            place, rvalue
        );
        let path = self.visit_place(place);
        self.visit_rvalue(path, rvalue);
    }

    /// Write the discriminant for a variant to the enum Place.
    fn visit_set_discriminant(
        &self,
        place: &mir::Place,
        variant_index: rustc::ty::layout::VariantIdx,
    ) {
        debug!(
            "default visit_set_discriminant(place: {:?}, variant_index: {:?})",
            place, variant_index
        );
    }

    /// Start a live range for the storage of the local.
    fn visit_storage_live(&self, local: mir::Local) {
        debug!("default visit_storage_live(local: {:?})", local);
    }

    /// End the current live range for the storage of the local.
    fn visit_storage_dead(&self, local: mir::Local) {
        debug!("default visit_storage_dead(local: {:?})", local);
    }

    /// Execute a piece of inline Assembly.
    fn visit_inline_asm(
        &self,
        asm: &hir::InlineAsm,
        outputs: &[mir::Place],
        inputs: &[(syntax_pos::Span, mir::Operand)],
    ) {
        debug!(
            "default visit_inline_asm(asm: {:?}, outputs: {:?}, inputs: {:?})",
            asm, outputs, inputs
        );
    }

    /// Retag references in the given place, ensuring they got fresh tags.  This is
    /// part of the Stacked Borrows model. These statements are currently only interpreted
    /// by miri and only generated when "-Z mir-emit-retag" is passed.
    /// See <https://internals.rust-lang.org/t/stacked-borrows-an-aliasing-model-for-rust/8153/>
    /// for more details.
    fn visit_retag(&self, retag_kind: mir::RetagKind, place: &mir::Place) {
        debug!(
            "default visit_retag(retag_kind: {:?}, place: {:?})",
            retag_kind, place
        );
    }

    /// Calls a specialized visitor for each kind of terminator.
    fn visit_terminator(&mut self, source_info: mir::SourceInfo, kind: &mir::TerminatorKind) {
        debug!("{:?}", source_info);
        self.current_span = source_info.span;
        match kind {
            mir::TerminatorKind::Goto { target } => self.visit_goto(*target),
            mir::TerminatorKind::SwitchInt {
                discr,
                switch_ty,
                values,
                targets,
            } => self.visit_switch_int(discr, switch_ty, values, targets),
            mir::TerminatorKind::Resume => self.visit_resume(),
            mir::TerminatorKind::Abort => self.visit_abort(),
            mir::TerminatorKind::Return => self.visit_return(),
            mir::TerminatorKind::Unreachable => self.visit_unreachable(),
            mir::TerminatorKind::Drop {
                location,
                target,
                unwind,
            } => self.visit_drop(location, *target, *unwind),
            mir::TerminatorKind::DropAndReplace { .. } => unreachable!(),
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                cleanup,
                from_hir_call,
            } => self.visit_call(func, args, destination, *cleanup, *from_hir_call),
            mir::TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                cleanup,
            } => self.visit_assert(cond, *expected, msg, *target, *cleanup),
            mir::TerminatorKind::Yield { .. } => unreachable!(),
            mir::TerminatorKind::GeneratorDrop => unreachable!(),
            mir::TerminatorKind::FalseEdges { .. } => unreachable!(),
            mir::TerminatorKind::FalseUnwind { .. } => unreachable!(),
        }
    }

    /// block should have one successor in the graph; we jump there
    fn visit_goto(&mut self, target: mir::BasicBlock) {
        debug!("default visit_goto(local: {:?})", target);
        // Propagate the entry condition to the successor block.
        self.current_environment
            .exit_conditions
            .insert(target, self.current_environment.entry_condition.clone());
    }

    /// `discr` evaluates to an integer; jump depending on its value
    /// to one of the targets, and otherwise fallback to last element of `targets`.
    ///
    /// # Arguments
    /// * `discr` - Discriminant value being tested
    /// * `switch_ty` - type of value being tested
    /// * `values` - Possible values. The locations to branch to in each case
    /// are found in the corresponding indices from the `targets` vector.
    /// * `targets` - Possible branch sites. The last element of this vector is used
    /// for the otherwise branch, so targets.len() == values.len() + 1 should hold.
    fn visit_switch_int(
        &mut self,
        discr: &mir::Operand,
        switch_ty: rustc::ty::Ty,
        values: &[u128],
        targets: &[mir::BasicBlock],
    ) {
        debug!(
            "default visit_switch_int(discr: {:?}, switch_ty: {:?}, values: {:?}, targets: {:?})",
            discr, switch_ty, values, targets
        );
        let mut default_exit_condition = self.current_environment.entry_condition.clone();
        let discr = self.visit_operand(discr);
        for i in 0..values.len() {
            let val: AbstractValue = ConstantValue::U128(values[i]).into();
            let cond = discr.equals(&val, None);
            let not_cond = cond.not(None);
            default_exit_condition = default_exit_condition.and(&not_cond, None);
            let target = targets[i];
            self.current_environment
                .exit_conditions
                .insert(target, cond);
        }
        self.current_environment
            .exit_conditions
            .insert(targets[values.len()], default_exit_condition);
    }

    /// Indicates that the landing pad is finished and unwinding should
    /// continue. Emitted by build::scope::diverge_cleanup.
    fn visit_resume(&self) {
        debug!("default visit_resume()");
    }

    /// Indicates that the landing pad is finished and that the process
    /// should abort. Used to prevent unwinding for foreign items.
    fn visit_abort(&self) {
        debug!("default visit_abort()");
    }

    /// Indicates a normal return. The return place should have
    /// been filled in by now. This should occur at most once.
    fn visit_return(&mut self) {
        debug!("default visit_return()");
        if self.check_for_errors {
            // Done with fixed point, so prepare to summarize.
            let return_guard = self.current_environment.entry_condition.as_bool_if_known();
            if return_guard.unwrap_or(false) {
                self.exit_environment = self.current_environment.clone();
            } else if return_guard.unwrap_or(true) {
                self.exit_environment = self.current_environment.join(
                    &self.exit_environment,
                    &self.current_environment.entry_condition,
                );
            }
        }
    }

    /// Indicates a terminator that can never be reached.
    fn visit_unreachable(&self) {
        debug!("default visit_unreachable()");
    }

    /// Drop the Place
    fn visit_drop(
        &mut self,
        location: &mir::Place,
        target: mir::BasicBlock,
        unwind: Option<mir::BasicBlock>,
    ) {
        debug!(
            "default visit_drop(location: {:?}, target: {:?}, unwind: {:?})",
            location, target, unwind
        );
        // Propagate the entry condition to the successor blocks.
        self.current_environment
            .exit_conditions
            .insert(target, self.current_environment.entry_condition.clone());
        if let Some(unwind_target) = unwind {
            self.current_environment.exit_conditions.insert(
                unwind_target,
                self.current_environment.entry_condition.clone(),
            );
        }
    }

    /// Block ends with a call of a converging function
    ///
    /// #Arguments
    /// * `func` - The function that’s being called
    /// * `args` - Arguments the function is called with.
    /// These are owned by the callee, which is free to modify them.
    /// This allows the memory occupied by "by-value" arguments to be
    /// reused across function calls without duplicating the contents.
    /// * `destination` - Destination for the return value. If some, the call is converging.
    /// * `cleanup` - Cleanups to be done if the call unwinds.
    /// * `from_hir_call` - Whether this is from a call in HIR, rather than from an overloaded
    /// operator. True for overloaded function call.
    fn visit_call(
        &mut self,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &Option<(mir::Place, mir::BasicBlock)>,
        cleanup: Option<mir::BasicBlock>,
        from_hir_call: bool,
    ) {
        debug!("default visit_call(func: {:?}, args: {:?}, destination: {:?}, cleanup: {:?}, from_hir_call: {:?})", func, args, destination, cleanup, from_hir_call);
        let func_to_call = self.visit_operand(func);
        let function_summary = match func_to_call.value.expression_domain {
            ExpressionDomain::CompileTimeConstant(ConstantValue::Function {
                def_id: Some(def_id),
                ..
            }) => self
                .summary_cache
                .get_summary_for(def_id, Some(self.def_id))
                .clone(),
            ExpressionDomain::CompileTimeConstant(ConstantValue::Function {
                ref summary_cache_key,
                ..
            }) => self
                .summary_cache
                .get_persistent_summary_for(summary_cache_key),
            _ => Summary::default(),
        };
        if let Some((place, target)) = destination {
            // Assign function result to place
            let return_value_path = self.visit_place(place);
            let return_value = function_summary.result.unwrap_or(abstract_value::TOP);
            self.current_environment
                .update_value_at(return_value_path, return_value);
            // Propagate the entry condition to the successor blocks.
            self.current_environment
                .exit_conditions
                .insert(*target, self.current_environment.entry_condition.clone());
        }
        if let Some(cleanup_target) = cleanup {
            self.current_environment.exit_conditions.insert(
                cleanup_target,
                self.current_environment.entry_condition.clone(),
            );
        }
        if !self.check_for_errors {
            return;
        }
        if let ExpressionDomain::CompileTimeConstant(fun) = func_to_call.value.expression_domain {
            if self
                .constant_value_cache
                .check_if_std_intrinsics_unreachable_function(&fun)
            {
                let span = self.current_span;
                let mut err = self.session.struct_span_warn(
                    span,
                    "Control might reach a call to std::intrinsics::unreachable",
                );
                (self.emit_diagnostic)(&mut err, &mut self.buffered_diagnostics);
            }
        }
    }

    /// Jump to the target if the condition has the expected value,
    /// otherwise panic with a message and a cleanup target.
    fn visit_assert(
        &mut self,
        cond: &mir::Operand,
        expected: bool,
        msg: &mir::AssertMessage,
        target: mir::BasicBlock,
        cleanup: Option<mir::BasicBlock>,
    ) {
        debug!("default visit_assert(cond: {:?}, expected: {:?}, msg: {:?}, target: {:?}, cleanup: {:?})", cond, expected, msg, target, cleanup);
        let cond = self.visit_operand(cond);
        // Propagate the entry condition to the successor blocks, conjoined with cond (or !cond).
        let exit_condition = self.current_environment.entry_condition.and(&cond, None);
        self.current_environment
            .exit_conditions
            .insert(target, exit_condition);
        if let Some(cleanup_target) = cleanup {
            let cleanup_condition = self
                .current_environment
                .entry_condition
                .and(&cond.not(None), None);
            self.current_environment
                .exit_conditions
                .insert(cleanup_target, cleanup_condition);
        }
    }

    /// Calls a specialized visitor for each kind of Rvalue
    fn visit_rvalue(&mut self, path: Path, rvalue: &mir::Rvalue) {
        match rvalue {
            mir::Rvalue::Use(operand) => {
                self.visit_use(path, operand);
            }
            mir::Rvalue::Repeat(operand, count) => {
                self.visit_repeat(path, operand, *count);
            }
            mir::Rvalue::Ref(region, borrow_kind, place) => {
                self.visit_ref(path, region, *borrow_kind, place);
            }
            mir::Rvalue::Len(place) => {
                self.visit_len(path, place);
            }
            mir::Rvalue::Cast(cast_kind, operand, ty) => {
                self.visit_cast(path, *cast_kind, operand, ty);
            }
            mir::Rvalue::BinaryOp(bin_op, left_operand, right_operand) => {
                self.visit_binary_op(path, *bin_op, left_operand, right_operand);
            }
            mir::Rvalue::CheckedBinaryOp(bin_op, left_operand, right_operand) => {
                self.visit_checked_binary_op(path, *bin_op, left_operand, right_operand);
            }
            mir::Rvalue::NullaryOp(null_op, ty) => {
                self.visit_nullary_op(path, *null_op, ty);
            }
            mir::Rvalue::UnaryOp(unary_op, operand) => {
                self.visit_unary_op(path, *unary_op, operand);
            }
            mir::Rvalue::Discriminant(place) => {
                self.visit_discriminant(path, place);
            }
            mir::Rvalue::Aggregate(aggregate_kinds, operands) => {
                self.visit_aggregate(path, aggregate_kinds, operands);
            }
        }
    }

    /// path = x (either a move or copy, depending on type of x), or path = constant.
    fn visit_use(&mut self, path: Path, operand: &mir::Operand) {
        debug!(
            "default visit_use(path: {:?}, operand: {:?})",
            path, operand
        );
        match operand {
            mir::Operand::Copy(place) => {
                self.visit_used_copy(path, place);
            }
            mir::Operand::Move(place) => {
                self.visit_used_move(path, place);
            }
            mir::Operand::Constant(constant) => {
                let mir::Constant {
                    span,
                    ty,
                    user_ty,
                    literal,
                } = constant.borrow();
                let const_value: AbstractValue =
                    self.visit_constant(ty, *user_ty, literal).clone().into();
                self.current_environment
                    .update_value_at(path, const_value.with_provenance(*span));
            }
        };
    }

    /// For each (path', value) pair in the environment where path' is rooted in place,
    /// add a (path'', value) pair to the environment where path'' is a copy of path re-rooted
    /// with place.
    fn visit_used_copy(&mut self, target_path: Path, place: &mir::Place) {
        debug!(
            "default visit_used_copy(target_path: {:?}, place: {:?})",
            target_path, place
        );
        let rpath = self.visit_place(place);
        if let Some(value) = self.try_to_deref(&rpath) {
            debug!("copying {:?} to {:?}", value, target_path);
            self.current_environment
                .value_map
                .insert(target_path, value.with_provenance(self.current_span));
            return;
        }
        let value_map = &self.current_environment.value_map;
        for (path, value) in value_map
            .iter()
            .filter(|(p, _)| Self::path_is_rooted_by(*p, &rpath))
        {
            let qualified_path = Self::replace_root(&path, target_path.clone());
            debug!("copying {:?} to {:?}", value, qualified_path);
            value_map.insert(qualified_path, value.with_provenance(self.current_span));
        }
    }

    /// For each (path', value) pair in the environment where path' is rooted in place,
    /// add a (path'', value) pair to the environment where path'' is a copy of path re-rooted
    /// with place, and also remove the (path', value) pair from the environment.
    fn visit_used_move(&mut self, target_path: Path, place: &mir::Place) {
        debug!(
            "default visit_used_move(target_path: {:?}, place: {:?})",
            target_path, place
        );
        let rpath = self.visit_place(place);
        let value_map = &self.current_environment.value_map;
        for (path, value) in value_map
            .iter()
            .filter(|(p, _)| Self::path_is_rooted_by(*p, &rpath))
        {
            let qualified_path = Self::replace_root(&path, target_path.clone());
            debug!("moving {:?} to {:?}", value, qualified_path);
            value_map.remove(&path);
            value_map.insert(qualified_path, value.with_provenance(self.current_span));
        }
    }

    /// True if path qualifies root, or another qualified path rooted by root.
    fn path_is_rooted_by(path: &Path, root: &Path) -> bool {
        *path == *root
            || match path {
                Path::QualifiedPath { qualifier, .. } => Self::path_is_rooted_by(qualifier, root),
                _ => false,
            }
    }

    /// Returns a copy path with the root replaced by new_root.
    fn replace_root(path: &Path, new_root: Path) -> Path {
        match path {
            Path::QualifiedPath {
                qualifier,
                selector,
            } => {
                let new_qualifier = Self::replace_root(qualifier, new_root);
                Path::QualifiedPath {
                    qualifier: box new_qualifier,
                    selector: selector.clone(),
                }
            }
            _ => new_root,
        }
    }

    /// path = [x; 32]
    fn visit_repeat(&mut self, path: Path, operand: &mir::Operand, count: u64) {
        debug!(
            "default visit_repeat(path: {:?}, operand: {:?}, count: {:?})",
            path, operand, count
        );
        self.visit_operand(operand);
        //todo:
        // get a heap address and put it in Path::AbstractHeapAddress
        // get an abs value for x
        // create a PathSelector::Index paths where the value is the range 0..count
        // add qualified path to the environment with value x.
        self.current_environment
            .update_value_at(path, abstract_value::TOP);
    }

    /// path = &x or &mut x
    fn visit_ref(
        &mut self,
        path: Path,
        region: rustc::ty::Region,
        borrow_kind: mir::BorrowKind,
        place: &mir::Place,
    ) {
        debug!(
            "default visit_ref(path: {:?}, region: {:?}, borrow_kind: {:?}, place: {:?})",
            path, region, borrow_kind, place
        );
        let value_path = self.visit_place(place);
        let value = ExpressionDomain::Reference(value_path).into();
        self.current_environment.update_value_at(path, value);
    }

    /// path = length of a [X] or [X;n] value.
    fn visit_len(&mut self, path: Path, place: &mir::Place) {
        debug!("default visit_len(path: {:?}, place: {:?})", path, place);
        let value_path = self.visit_place(place);
        let _value = self.lookup_path_and_refine_result(value_path);
        //todo: get a value that is the length of _value.
        self.current_environment
            .update_value_at(path, abstract_value::TOP);
    }

    /// path = operand. The cast is a no-op for the interpreter.
    fn visit_cast(
        &mut self,
        path: Path,
        cast_kind: mir::CastKind,
        operand: &mir::Operand,
        ty: rustc::ty::Ty,
    ) {
        debug!(
            "default visit_cast(path: {:?}, cast_kind: {:?}, operand: {:?}, ty: {:?})",
            path, cast_kind, operand, ty
        );
        let value = self.visit_operand(operand);
        self.current_environment.update_value_at(path, value);
    }

    /// Apply the given binary operator to the two operands and assign result to path.
    fn visit_binary_op(
        &mut self,
        path: Path,
        bin_op: mir::BinOp,
        left_operand: &mir::Operand,
        right_operand: &mir::Operand,
    ) {
        debug!(
            "default visit_binary_op(path: {:?}, bin_op: {:?}, left_operand: {:?}, right_operand: {:?})",
            path, bin_op, left_operand, right_operand
        );
        let _left = self.visit_operand(left_operand);
        let _right = self.visit_operand(right_operand);
        //todo: get a value that is the bin_op of _left and _right.
        self.current_environment
            .update_value_at(path, abstract_value::TOP);
    }

    /// Apply the given binary operator to the two operands, with overflow checking where appropriate
    /// and assign the result to path.
    fn visit_checked_binary_op(
        &mut self,
        path: Path,
        bin_op: mir::BinOp,
        left_operand: &mir::Operand,
        right_operand: &mir::Operand,
    ) {
        debug!("default visit_checked_binary_op(path: {:?}, bin_op: {:?}, left_operand: {:?}, right_operand: {:?})", path, bin_op, left_operand, right_operand);
        let _left = self.visit_operand(left_operand);
        let _right = self.visit_operand(right_operand);
        //todo: get a value that is the checked bin_op of _left and _right.
        //todo: what should happen if the operation overflows?
        self.current_environment
            .update_value_at(path, abstract_value::TOP);
    }

    /// Create a value based on the given type and assign it to path.
    fn visit_nullary_op(&mut self, path: Path, null_op: mir::NullOp, ty: rustc::ty::Ty) {
        debug!(
            "default visit_nullary_op(path: {:?}, null_op: {:?}, ty: {:?})",
            path, null_op, ty
        );
        let value = match null_op {
            mir::NullOp::Box => self.get_new_heap_address(),
            mir::NullOp::SizeOf => {
                //todo: figure out how to get the size from ty.
                abstract_value::TOP
            }
        };
        self.current_environment.update_value_at(path, value);
    }

    /// Allocates a new heap address and caches it, keyed with the current location
    /// so that subsequent visits deterministically use the same address when processing
    /// the instruction at this location. If we don't do this the fixed point loop wont converge.
    fn get_new_heap_address(&mut self) -> AbstractValue {
        let addresses = &mut self.heap_addresses;
        let constants = &mut self.constant_value_cache;
        addresses
            .entry(self.current_location)
            .or_insert_with(|| constants.get_new_heap_address().into())
            .clone()
    }

    /// Apply the given unary operator to the operand and assign to path.
    fn visit_unary_op(&mut self, path: Path, un_op: mir::UnOp, operand: &mir::Operand) {
        debug!(
            "default visit_unary_op(path: {:?}, un_op: {:?}, operand: {:?})",
            path, un_op, operand
        );
        let _operand = self.visit_operand(operand);
        //todo: get a value that is the un_op of _operand.
        self.current_environment
            .update_value_at(path, abstract_value::TOP);
    }

    /// Read the discriminant of an ADT and assign to path.
    ///
    /// Undefined (i.e. no effort is made to make it defined, but there’s no reason why it cannot
    /// be defined to return, say, a 0) if ADT is not an enum.
    fn visit_discriminant(&mut self, path: Path, place: &mir::Place) {
        debug!(
            "default visit_discriminant(path: {:?}, place: {:?})",
            path, place
        );
        let _value_path = self.visit_place(place);
        //todo: modify _value_path to get the discriminant and look it up in the environment
        self.current_environment
            .update_value_at(path, abstract_value::TOP);
    }

    /// Create an aggregate value, like a tuple or struct and assign it to path.  This is
    /// only needed because we want to distinguish `dest = Foo { x:
    /// ..., y: ... }` from `dest.x = ...; dest.y = ...;` in the case
    /// that `Foo` has a destructor. These rvalues can be optimized
    /// away after type-checking and before lowering.
    fn visit_aggregate(
        &mut self,
        path: Path,
        aggregate_kinds: &mir::AggregateKind,
        operands: &[mir::Operand],
    ) {
        debug!(
            "default visit_aggregate(path: {:?}, aggregate_kinds: {:?}, operands: {:?})",
            path, aggregate_kinds, operands
        );
        let aggregate_value = self.get_new_heap_address();
        self.current_environment
            .update_value_at(path, aggregate_value);
        //todo: an assignment for each operand.
    }

    /// These are values that can appear inside an rvalue. They are intentionally
    /// limited to prevent rvalues from being nested in one another.
    fn visit_operand(&mut self, operand: &mir::Operand) -> AbstractValue {
        let span = self.current_span;
        let (expression_domain, span) = match operand {
            mir::Operand::Copy(place) => {
                self.visit_copy(place);
                (ExpressionDomain::Top, span)
            }
            mir::Operand::Move(place) => {
                self.visit_move(place);
                (ExpressionDomain::Top, span)
            }
            mir::Operand::Constant(constant) => {
                let mir::Constant {
                    span,
                    ty,
                    user_ty,
                    literal,
                } = constant.borrow();
                let const_value = self.visit_constant(ty, *user_ty, literal).clone();
                (ExpressionDomain::CompileTimeConstant(const_value), *span)
            }
        };
        AbstractValue {
            provenance: vec![span],
            value: AbstractDomains { expression_domain },
        }
    }

    /// Copy: The value must be available for use afterwards.
    ///
    /// This implies that the type of the place must be `Copy`; this is true
    /// by construction during build, but also checked by the MIR type checker.
    fn visit_copy(&self, place: &mir::Place) {
        debug!("default visit_copy(place: {:?})", place);
    }

    /// Move: The value (including old borrows of it) will not be used again.
    ///
    /// Safe for values of all types (modulo future developments towards `?Move`).
    /// Correct usage patterns are enforced by the borrow checker for safe code.
    /// `Copy` may be converted to `Move` to enable "last-use" optimizations.
    fn visit_move(&mut self, place: &mir::Place) {
        debug!("default visit_move(place: {:?})", place);
        self.visit_place(place);
    }

    /// Synthesizes a constant value.
    fn visit_constant(
        &mut self,
        ty: Ty,
        user_ty: Option<UserTypeAnnotationIndex>,
        literal: &LazyConst,
    ) -> &ConstantValue {
        use rustc::mir::interpret::ConstValue;
        use rustc::mir::interpret::Scalar;
        debug!(
            "default visit_constant(ty: {:?}, user_ty: {:?}, literal: {:?})",
            ty, user_ty, literal
        );
        match literal {
            LazyConst::Evaluated(Const { val, .. }) => {
                debug!("sty: {:?}", ty.sty);
                match ty.sty {
                    TyKind::Bool => match val {
                        ConstValue::Scalar(Scalar::Bits { bits, .. }) => {
                            if *bits == 0 {
                                &ConstantValue::False
                            } else {
                                &ConstantValue::True
                            }
                        }
                        _ => unreachable!(),
                    },
                    TyKind::Char => {
                        if let ConstValue::Scalar(Scalar::Bits { bits, .. }) = val {
                            &mut self
                                .constant_value_cache
                                .get_char_for(char::try_from(*bits as u32).unwrap())
                        } else {
                            unreachable!()
                        }
                    }
                    TyKind::Float(..) => match val {
                        ConstValue::Scalar(Scalar::Bits { bits, size }) => {
                            let mut value: u64 = match *size {
                                4 => u64::from(*bits as u32),
                                _ => *bits as u64,
                            };
                            &mut self.constant_value_cache.get_f64_for(value)
                        }
                        _ => unreachable!(),
                    },
                    TyKind::FnDef(def_id, ..) => self.visit_function_reference(def_id),
                    TyKind::Int(..) => match val {
                        ConstValue::Scalar(Scalar::Bits { bits, size }) => {
                            let mut value: i128 = match *size {
                                1 => i128::from(*bits as i8),
                                2 => i128::from(*bits as i16),
                                4 => i128::from(*bits as i32),
                                8 => i128::from(*bits as i64),
                                _ => *bits as i128,
                            };
                            &mut self.constant_value_cache.get_i128_for(value)
                        }
                        _ => unreachable!(),
                    },
                    TyKind::Ref(
                        _,
                        &rustc::ty::TyS {
                            sty: TyKind::Str, ..
                        },
                        _,
                    ) => {
                        if let ConstValue::ScalarPair(ptr, len) = val {
                            if let Scalar::Ptr(ptr) = ptr {
                                if let Scalar::Bits { bits: len, .. } = len {
                                    let alloc = self.tcx.alloc_map.lock().get(ptr.alloc_id);
                                    if let Some(mir::interpret::AllocKind::Memory(alloc)) = alloc {
                                        let slice = &alloc.bytes[(ptr.offset.bytes() as usize)..]
                                            [..(*len as usize)];
                                        let s = std::str::from_utf8(slice).expect("non utf8 str");
                                        return &mut self.constant_value_cache.get_string_for(s);
                                    } else {
                                        panic!("pointer to erroneous constant {:?}, {:?}", ptr, len)
                                    }
                                }
                            }
                        };
                        unreachable!()
                    }
                    TyKind::Uint(..) => match val {
                        ConstValue::Scalar(Scalar::Bits { bits, .. }) => {
                            &mut self.constant_value_cache.get_u128_for(*bits)
                        }
                        _ => unreachable!(),
                    },
                    _ => &ConstantValue::Unimplemented,
                }
            }
            _ => &ConstantValue::Unimplemented,
        }
    }

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type, which is output (for a function
    /// named `foo` returning an `i32`) as `fn() -> i32 {foo}`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar = foo; // bar: fn() -> i32 {foo}
    /// ```
    fn visit_function_reference(&mut self, def_id: hir::def_id::DefId) -> &ConstantValue {
        &mut self.constant_value_cache.get_function_constant_for(
            def_id,
            &self.tcx,
            &mut self.summary_cache,
        )
    }

    /// Returns a Path instance that is the essentially the same as the Place instance, but which
    /// can be serialized and used as a cache key.
    fn visit_place(&mut self, place: &mir::Place) -> Path {
        debug!("default visit_place(place: {:?})", place);
        match place {
            mir::Place::Local(local) => Path::LocalVariable {
                ordinal: local.as_usize(),
            },
            mir::Place::Static(boxed_static) => {
                let def_id = boxed_static.def_id;
                let name = summaries::summary_key_str(&self.tcx, def_id);
                Path::StaticVariable { name }
            }
            mir::Place::Promoted(boxed_promoted) => {
                let index = boxed_promoted.0;
                Path::PromotedConstant {
                    ordinal: index.as_usize(),
                }
            }
            mir::Place::Projection(boxed_place_projection) => {
                let base = self.visit_place(&boxed_place_projection.base);
                let selector = self.visit_projection_elem(&boxed_place_projection.elem);
                Path::QualifiedPath {
                    qualifier: box base,
                    selector: box selector,
                }
            }
        }
    }

    /// Returns a PathSelector instance that is essentially the same as the ProjectionElem instance
    /// but which can be serialized.
    fn visit_projection_elem(
        &mut self,
        projection_elem: &mir::ProjectionElem<mir::Local, &rustc::ty::TyS>,
    ) -> PathSelector {
        debug!(
            "visit_projection_elem(projection_elem: {:?})",
            projection_elem
        );
        match projection_elem {
            mir::ProjectionElem::Deref => PathSelector::Deref,
            mir::ProjectionElem::Field(field, _) => PathSelector::Field(field.index()),
            mir::ProjectionElem::Index(local) => {
                let local_path = Path::LocalVariable {
                    ordinal: local.as_usize(),
                };
                let index_value = self.lookup_path_and_refine_result(local_path);
                PathSelector::Index(box index_value)
            }
            mir::ProjectionElem::ConstantIndex {
                offset,
                min_length,
                from_end,
            } => PathSelector::ConstantIndex {
                offset: *offset,
                min_length: *min_length,
                from_end: *from_end,
            },
            mir::ProjectionElem::Subslice { from, to } => PathSelector::Subslice {
                from: *from,
                to: *to,
            },
            mir::ProjectionElem::Downcast(_, index) => PathSelector::Downcast(index.as_usize()),
        }
    }
}
