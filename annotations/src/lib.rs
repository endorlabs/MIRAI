// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// A set of macros and functions to use for annotating source files that are being checked with MIRAI

/// Provides a way to specify a value that should be treated abstractly by the verifier.
/// The concrete argument provides type information to the verifier and a meaning for
/// the expression when compiled by the rust compiler.
///
/// The expected use case for this is inside test cases. Principally this would be test cases
/// for the verifier itself, but it can also be used to "fuzz" unit tests in user code.
#[macro_export]
macro_rules! abstract_value {
    ($value:expr) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_abstract_value($value)
        } else {
            $value
        }
    };
}

/// Equivalent to a no op when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition unless it can
/// prove it to be false.
#[macro_export]
macro_rules! assume {
    ($condition:expr) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($condition)
        }
    };
}

/// Equivalent to a no op when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume that the preconditions of the next
/// function call have been met.
/// This is to be used when the precondition has been inferred and involves private state that
/// cannot be constrained by a normal assumption.
/// Note that it is bad style for an API to rely on preconditions that cannot be checked by the
/// caller, so this is only here for supporting legacy APIs.
#[macro_export]
macro_rules! assume_preconditions {
    () => {
        if cfg!(mirai) {
            mirai_annotations::mirai_assume_preconditions()
        }
    };
}

/// Equivalent to the standard assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition unless it can
/// prove it to be false.
#[macro_export]
macro_rules! checked_assume {
    ($condition:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($condition)
        } else {
            assert!($condition);
        }
    );
    ($condition:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($condition);
        } else {
            assert!($condition, $($arg)*);
        }
    );
}

/// Equivalent to the standard assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition unless it can
/// prove it to be false.
#[macro_export]
macro_rules! checked_assume_eq {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left == $right)
        } else {
            assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left == $right);
        } else {
            assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard assert_ne! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition unless it can
/// prove it to be false.
#[macro_export]
macro_rules! checked_assume_ne {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left != $right)
        } else {
            assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left != $right);
        } else {
            assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition unless it can
/// prove it to be false.
#[macro_export]
macro_rules! debug_checked_assume {
    ($condition:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($condition)
        } else {
            debug_assert!($condition);
        }
    );
    ($condition:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($condition);
        } else {
            debug_assert!($condition, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition unless it can
/// prove it to be false.
#[macro_export]
macro_rules! debug_checked_assume_eq {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left == $right)
        } else {
            debug_assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left == $right);
        } else {
            debug_assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_ne! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition unless it can
/// prove it to be false.
#[macro_export]
macro_rules! debug_checked_assume_ne {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left != $right)
        } else {
            debug_assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_assume($left != $right);
        } else {
            debug_assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to a no op when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! postcondition {
    ($condition:expr) => {
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false, "unsatisfied postcondition");
        }
    };
    ($condition:expr, $message:literal) => {
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  concat!("unsatisfied postcondition: ", $message));
        }
    };
    ($condition:expr, $($arg:tt)*) => {
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  concat!("unsatisfied postcondition: ", stringify!($($arg)*)));
        }
    };
}

/// Equivalent to a no op when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! assumed_postcondition {
    ($condition:expr) => {
        #[cfg(mirai)]
        {
            mirai_annotations::mirai_postcondition($condition, true, "")
        }
        #[cfg(not(mirai))]
        {
            debug_assert!($condition);
        }
    };
}

/// Equivalent to the standard assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! checked_postcondition {
    ($condition:expr) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  "unsatisfied postcondition")
        }
        #[cfg(not(mirai))] {
            assert!($condition);
        }
    );
    ($condition:expr, $message:literal) => {
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  concat!("unsatisfied postcondition: ", $message))
        }
        #[cfg(not(mirai))] {
            assert!($condition, $message);
        }
    };
    ($condition:expr, $($arg:tt)*) => {
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  concat!("unsatisfied postcondition: ", stringify!($($arg)*)));
        }
        #[cfg(not(mirai))] {
            assert!($condition, $($arg)*);
        }
    };
}

/// Equivalent to the standard assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! checked_postcondition_eq {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_postcondition($left == $right, false,  concat!("unsatisfied postcondition: ", stringify!($left == $right)))
        } else {
            assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_postcondition($left == $right, false,  concat!("unsatisfied postcondition: ", stringify!($left == $right), ", ", $message))
        } else {
            assert_eq!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_postcondition($left == $right, false,  concat!("unsatisfied postcondition: ", stringify!($left == $right), ", ", stringify!($($arg)*)));
        } else {
            assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard assert_ne! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! checked_postcondition_ne {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_postcondition($left != $right, false,  concat!("unsatisfied postcondition: ", stringify!($left != $right)))
        } else {
            assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_postcondition($left != $right, false,  concat!("unsatisfied postcondition: ", stringify!($left != $right), ", ", $message))
        } else {
            assert_ne!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_postcondition($left != $right, false,  concat!("unsatisfied postcondition: ", stringify!($left != $right), ", ", stringify!($($arg)*)));
        } else {
            assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! debug_checked_postcondition {
    ($condition:expr) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  "unsatisfied postcondition")
        }
        #[cfg(not(mirai))] {
            debug_assert!($condition);
        }
    );
    ($condition:expr, $message:literal) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  concat!("unsatisfied postcondition: ", $message))
        }
        #[cfg(not(mirai))] {
            debug_assert!($condition, $message);
        }
    );
    ($condition:expr, $($arg:tt)*) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($condition, false,  concat!("unsatisfied postcondition: ", stringify!($($arg)*)));
        }
        #[cfg(not(mirai))] {
            debug_assert!($condition, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! debug_checked_postcondition_eq {
    ($left:expr, $right:expr) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($left == $right, false,  concat!("unsatisfied postcondition: ", stringify!($left == $right)))
        }
        #[cfg(not(mirai))] {
            debug_assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($left == $right, false,  concat!("unsatisfied postcondition: ", stringify!($left == $right), ", ", $message))
        }
        #[cfg(not(mirai))] {
            debug_assert_eq!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($left == $right, false,  concat!("unsatisfied postcondition: ", stringify!($left == $right), ", ", stringify!($($arg)*)));
        }
        #[cfg(not(mirai))] {
            debug_assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_ne! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify the condition at the
/// point where it appears in a function, but to also add it a postcondition that can
/// be assumed by the caller of the function.
#[macro_export]
macro_rules! debug_checked_postcondition_ne {
    ($left:expr, $right:expr) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($left != $right, false,  concat!("unsatisfied postcondition: ", stringify!($left != $right)))
        }
        #[cfg(not(mirai))] {
            debug_assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($left != $right, false,  concat!("unsatisfied postcondition: ", stringify!($left != $right), ", ", $message))
        }
        #[cfg(not(mirai))] {
            debug_assert_ne!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        #[cfg(mirai)] {
            mirai_annotations::mirai_postcondition($left != $right, false,  concat!("unsatisfied postcondition: ", stringify!($left != $right), ", ", stringify!($($arg)*)));
        }
        #[cfg(not(mirai))] {
            debug_assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to a no op when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a precondition that must
/// be verified by the caller of the function.
#[macro_export]
macro_rules! precondition {
    ($condition:expr) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, "unsatisfied precondition")
        }
    };
    ($condition:expr, $message:literal) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, concat!("unsatisfied precondition: ", $message))
        }
    };
    ($condition:expr, $($arg:tt)*) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, concat!("unsatisfied precondition: ", stringify!($($arg)*)));
        }
    };
}

/// Equivalent to the standard assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a precondition that must
/// be verified by the caller of the function.
#[macro_export]
macro_rules! checked_precondition {
    ($condition:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, "unsatisfied precondition")
        } else {
            assert!($condition);
        }
    );
    ($condition:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, concat!("unsatisfied precondition: ", $message))
        } else {
            assert!($condition, $message);
        }
    );
    ($condition:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, concat!("unsatisfied precondition: ", stringify!($($arg)*)));
        } else {
            assert!($condition, $($arg)*);
        }
    );
}

/// Equivalent to the standard assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a precondition that must
/// be verified by the caller of the function.
#[macro_export]
macro_rules! checked_precondition_eq {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left == $right, concat!("unsatisfied precondition: ", stringify!($left == $right)))
        } else {
            assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left == $right, concat!("unsatisfied precondition: ", stringify!($left == $right), ", ", $message))
        } else {
            assert_eq!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left == $right, concat!("unsatisfied precondition: ", stringify!($left == $right), ", ", stringify!($($arg)*)));
        } else {
            assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard assert_ne! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a precondition that must
/// be verified by the caller of the function.
#[macro_export]
macro_rules! checked_precondition_ne {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left != $right, concat!("unsatisfied precondition: ", stringify!($left != $right)))
        } else {
            assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left != $right, concat!("unsatisfied precondition: ", stringify!($left != $right), ", ", $message))
        } else {
            assert_ne!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left != $right, concat!("unsatisfied precondition: ", stringify!($left != $right), ", ", stringify!($($arg)*)));
        } else {
            assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a precondition that must
/// be verified by the caller of the function.
#[macro_export]
macro_rules! debug_checked_precondition {
    ($condition:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, "unsatisfied precondition")
        } else {
            debug_assert!($condition);
        }
    );
    ($condition:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, concat!("unsatisfied precondition: ", $message))
        } else {
            debug_assert!($condition, $message);
        }
    );
    ($condition:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($condition, concat!("unsatisfied precondition: ", stringify!($($arg)*)));
        } else {
            debug_assert!($condition, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a precondition that must
/// be verified by the caller of the function.
#[macro_export]
macro_rules! debug_checked_precondition_eq {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left == $right, concat!("unsatisfied precondition: ", stringify!($left == $right)))
        } else {
            debug_assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left == $right, concat!("unsatisfied precondition: ", stringify!($left == $right), ", ", $message))
        } else {
            debug_assert_eq!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left == $right, concat!("unsatisfied precondition: ", stringify!($left == $right), ", ", stringify!($($arg)*)));
        } else {
            debug_assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_ne! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume the condition at the
/// point where it appears in a function, but to also add it a precondition that must
/// be verified by the caller of the function.
#[macro_export]
macro_rules! debug_checked_precondition_ne {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left != $right, concat!("unsatisfied precondition: ", stringify!($left != $right)))
        } else {
            debug_assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left != $right, concat!("unsatisfied precondition: ", stringify!($left != $right), ", ", $message))
        } else {
            debug_assert_ne!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_precondition_start();
            mirai_annotations::mirai_precondition($left != $right, concat!("unsatisfied precondition: ", stringify!($left != $right), ", ", stringify!($($arg)*)));
        } else {
            debug_assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Terminates the program with a panic that is tagged as being an unrecoverable error.
/// Use this for errors that arise in correct programs due to external factors.
/// For example, if a file that is essential for running cannot be found for some reason.
#[macro_export]
macro_rules! unrecoverable {
    ($fmt:expr) => (
        panic!(concat!("unrecoverable: ", stringify!($fmt)));
    );
    ($fmt:expr, $($arg:tt)+) => (
        panic!(concat!("unrecoverable: ", stringify!($fmt)), $($arg)+);
    );
}

/// Equivalent to a no op when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to check the condition and
/// emit a diagnostic unless it can prove it to be true.
#[macro_export]
macro_rules! verify {
    ($condition:expr) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($condition, "false verification condition")
        }
    };
}

/// Equivalent to the standard assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to check the condition and
/// emit a diagnostic unless it can prove it to be true.
#[macro_export]
macro_rules! checked_verify {
    ($condition:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($condition, "false verification condition")
        } else {
            assert!($condition);
        }
    );
    ($condition:expr, $message:literal) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($condition, concat!("false verification condition: ", $message))
        } else {
            assert!($condition, $message);
        }
    };
    ($condition:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($condition,  concat!("false verification condition: ", stringify!($($arg)*)));
        } else {
            assert!($condition, $($arg)*);
        }
    );
}

/// Equivalent to the standard assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to check the condition and
/// emit a diagnostic unless it can prove it to be true.
#[macro_export]
macro_rules! checked_verify_eq {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left == $right, concat!("false verification condition: ", stringify!($left == $right)))
        } else {
            assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left == $right, concat!("false verification condition: ", stringify!($left == $right), ", ", $message))
        } else {
            assert_eq!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left == $right, concat!("false verification condition: ", stringify!($left == $right), ", ", stringify!($($arg)*)));
        } else {
            assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to check the condition and
/// emit a diagnostic unless it can prove it to be true.
#[macro_export]
macro_rules! checked_verify_ne {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left != $right, concat!("false verification condition: ", stringify!($left != $right)))
        } else {
            assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left != $right, concat!("false verification condition: ", stringify!($left != $right), ", ", $message))
        } else {
            assert_ne!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left != $right, concat!("false verification condition: ", stringify!($left != $right), ", ", stringify!($($arg)*)));
        } else {
            assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to check the condition and
/// emit a diagnostic unless it can prove it to be true.
#[macro_export]
macro_rules! debug_checked_verify {
    ($condition:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($condition, "false verification condition")
        } else {
            debug_assert!($condition);
        }
    );
    ($condition:expr, $message:literal) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($condition, concat!("false verification condition: ", $message))
        } else {
            debug_assert!($condition, $message);
        }
    };
    ($condition:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($condition,  concat!("false verification condition: ", stringify!($($arg)*)));
        } else {
            debug_assert!($condition, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_eq! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to check the condition and
/// emit a diagnostic unless it can prove it to be true.
#[macro_export]
macro_rules! debug_checked_verify_eq {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left == $right, concat!("false verification condition: ", stringify!($left == $right)))
        } else {
            debug_assert_eq!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left == $right, concat!("false verification condition: ", stringify!($left == $right), ", ", $message))
        } else {
            debug_assert_eq!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left == $right, concat!("false verification condition: ", stringify!($left == $right), ", ", stringify!($($arg)*)));
        } else {
            debug_assert_eq!($left, $right, $($arg)*);
        }
    );
}

/// Equivalent to the standard debug_assert_ne! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to check the condition and
/// emit a diagnostic unless it can prove it to be true.
#[macro_export]
macro_rules! debug_checked_verify_ne {
    ($left:expr, $right:expr) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left != $right, concat!("false verification condition: ", stringify!($left != $right)))
        } else {
            debug_assert_ne!($left, $right);
        }
    );
    ($left:expr, $right:expr, $message:literal) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left != $right, concat!("false verification condition: ", stringify!($left != $right), ", ", $message))
        } else {
            debug_assert_ne!($left, $right, $message);
        }
    );
    ($left:expr, $right:expr, $($arg:tt)*) => (
        if cfg!(mirai) {
            mirai_annotations::mirai_verify($left != $right, concat!("false verification condition: ", stringify!($left != $right), ", ", stringify!($($arg)*)));
        } else {
            debug_assert_ne!($left, $right, $($arg)*);
        }
    );
}

/// Retrieves the value of the specified model field, or the given default value if the model field
/// is not set.
/// This function has no meaning outside of a verification
/// condition and should not be used with checked or debug_checked conditions.
/// For example: precondition!(get_model_field!(x, f) > 1).
#[macro_export]
macro_rules! get_model_field {
    ($target:expr, $field_name:ident, $default_value:expr) => {
        mirai_annotations::mirai_get_model_field($target, stringify!($field_name), $default_value)
    };
}

/// Provides a way to refer to the result value of an abstract or contract function without
/// specifying an actual value anywhere.
/// This macro expands to unimplemented!() unless the program is compiled with MIRAI.
/// It result should therefore not be assigned to a variable unless the assignment is contained
/// inside a specification macro argument list.
/// It may, however, be the return value of the function, which should never be called and
/// therefore unimplemented!() is the right behavior for it at runtime.
#[macro_export]
macro_rules! result {
    () => {
        if cfg!(mirai) {
            mirai_annotations::mirai_result()
        } else {
            unimplemented!()
        }
    };
}

/// Sets the value of the specified model field.
/// A model field does not exist at runtime and is invisible to the Rust compiler.
/// This macro expands to nothing unless the program is compiled with MIRAI.
#[macro_export]
macro_rules! set_model_field {
    ($target:expr, $field_name:ident, $value:expr) => {
        if cfg!(mirai) {
            mirai_annotations::mirai_set_model_field($target, stringify!($field_name), $value);
        }
    };
}

/// Equivalent to unreachable! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to assume that the annotation statement cannot be reached.
#[macro_export]
macro_rules! assume_unreachable {
    () => {
        if cfg!(mirai) {
            unreachable!()
        } else {
            unreachable!()
        }
    };
    ($message:literal) => {
        if cfg!(mirai) {
            unreachable!()
        } else {
            unreachable!($message)
        }
    };
    ($msg:expr,) => ({
        if cfg!(mirai) {
            unreachable!()
        } else {
            unreachable!($msg)
        }
    });
    ($fmt:expr, $($arg:tt)*) => {
        if cfg!(mirai) {
            unreachable!()
        } else {
            unreachable!($fmt, $($arg)*)
        }
    };
}

/// Equivalent to unreachable! when used with an unmodified Rust compiler.
/// When compiled with MIRAI, this causes MIRAI to verify that the annotation statement cannot be reached.
#[macro_export]
macro_rules! verify_unreachable {
    () => {
        if cfg!(mirai) {
            panic!("statement is reachable");
        } else {
            unreachable!()
        }
    };
    ($message:literal) => {
        if cfg!(mirai) {
            panic!($message);
        } else {
            unreachable!($message)
        }
    };
    ($msg:expr,) => ({
        if cfg!(mirai) {
            panic!($message)
        } else {
            unreachable!($msg)
        }
    });
    ($fmt:expr, $($arg:tt)*) => {
        if cfg!(mirai) {
            panic!($fmt, $($arg)*);
        } else {
            unreachable!($fmt, $($arg)*)
        }
    };
}

// Helper function for MIRAI. Should only be called via the result! macro.
#[doc(hidden)]
pub fn mirai_abstract_value<T>(x: T) -> T {
    x
}

// Helper function for MIRAI. Should only be called via the assume macros.
#[doc(hidden)]
pub fn mirai_assume(_condition: bool) {}

// Helper function for MIRAI. Should only be called via the assume_precondition macro.
#[doc(hidden)]
pub fn mirai_assume_preconditions() {}

// Helper function for MIRAI. Should only be called via the postcondition macros.
#[doc(hidden)]
pub fn mirai_postcondition(_condition: bool, _assumed: bool, _message: &str) {}

// Helper function for MIRAI. Should only be called via the precondition macros.
#[doc(hidden)]
pub fn mirai_precondition_start() {}

// Helper function for MIRAI. Should only be called via the precondition macros.
#[doc(hidden)]
pub fn mirai_precondition(_condition: bool, _message: &str) {}

// Helper function for MIRAI. Should only be called via the verify macros.
#[doc(hidden)]
pub fn mirai_verify(_condition: bool, _message: &str) {}

// Helper function for MIRAI. Should only be called via the get_model_field macro.
#[doc(hidden)]
pub fn mirai_get_model_field<T, V>(_target: T, _field_name: &str, default_value: V) -> V {
    default_value
}

// Helper function for MIRAI. Should only be called via the result! macro.
#[doc(hidden)]
pub fn mirai_result<T>() -> T {
    unreachable!()
}

// Helper function for MIRAI. Should only be called via the set_model_field macro.
#[doc(hidden)]
pub fn mirai_set_model_field<T, V>(_target: T, _field_name: &str, _value: V) {}
