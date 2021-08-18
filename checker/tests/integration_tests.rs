// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// In an ideal world there would be a stable well documented set of crates containing a specific
// version of the Rust compiler along with its sources and debug information. We'd then just get
// those from crate.io and merely go on our way as just another Rust application. Rust compiler
// upgrades will be non events for Mirai until it is ready to jump to another release and old
// versions of Mirai will continue to work just as before.
//
// In the current world, however, we have to use the following hacky feature to get access to a
// private and not very stable set of APIs from whatever compiler is in the path when we run Mirai.
// While pretty bad, it is a lot less bad than having to write our own compiler, so here goes.
#![feature(rustc_private)]
#![feature(box_syntax)]

extern crate mirai;
extern crate rustc_ast;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_rayon;
extern crate tempfile;

use mirai::call_graph::{CallGraphConfig, CallGraphReduction};
use mirai::callbacks;
use mirai::options::{DiagLevel, Options};
use mirai::utils;
use mirai_annotations::unrecoverable;
use regex::Regex;
use rustc_rayon::iter::IntoParallelIterator;
use rustc_rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::fs::read_to_string;
use std::iter::FromIterator;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tempfile::TempDir;
use walkdir::WalkDir;

// Run the tests in the tests/run-pass directory.
// Eventually, there will be separate test cases for other directories such as compile-fail.
#[test]
fn run_pass() {
    let extern_deps = vec![
        (
            "mirai_annotations",
            find_extern_library("mirai_annotations"),
        ),
        ("contracts", find_extern_library("contracts")),
    ];
    let mut run_pass_path = PathBuf::from_str("tests/run-pass").unwrap();
    if !run_pass_path.exists() {
        run_pass_path = PathBuf::from_str("checker/tests/run-pass").unwrap();
    }
    let files = run_directory(run_pass_path);
    let result = invoke_driver_on_files(
        files,
        extern_deps,
        &(start_driver as fn(DriverConfig) -> usize),
    );
    assert_eq!(result, 0);
}

// Run the tests in the tests/call_graph directory.
#[test]
fn run_call_graph_tests() {
    let mut call_graph_tests_path = PathBuf::from_str("tests/call_graph").unwrap();
    if !call_graph_tests_path.exists() {
        call_graph_tests_path = PathBuf::from_str("checker/tests/call_graph").unwrap();
    }
    let files = run_directory(call_graph_tests_path);
    let result = invoke_driver_on_files(
        files,
        Vec::<(&str, String)>::new(),
        &(start_driver_call_graph as fn(DriverConfig) -> usize),
    );
    assert_eq!(result, 0);
}

fn find_extern_library(base_name: &str) -> String {
    let mut deps_path = PathBuf::from_str("../target/debug").unwrap();
    if !deps_path.exists() {
        deps_path = PathBuf::from_str("target/debug").unwrap();
    }

    for entry in WalkDir::new(deps_path)
        .contents_first(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        };
        let file_name = entry.file_name().to_str().unwrap_or("");
        // On Windows we have either lib{base_name}.rlib or {base_name}.dll. We match any form.
        if !file_name.starts_with(format!("lib{}", base_name).as_str())
            && !file_name.starts_with(base_name)
        {
            continue;
        }
        if entry.path().to_str().unwrap().contains(".dSYM/") {
            // There might be a directory .dSYM which contains the same library file
            // but for different purpose. Skip this.
            continue;
        }
        if !file_name.ends_with(".rlib")
            && !file_name.ends_with(".dylib")
            && !file_name.ends_with(".so")
            && !file_name.ends_with(".dll")
        {
            continue;
        }
        println!("resolving {}", entry.path().to_str().unwrap());
        return entry.path().to_str().unwrap().to_string();
    }
    unreachable!("could not find the `{}` library", base_name);
}

// Iterates through the files in the directory at the given path and runs each as a separate test
// case. For each case, a temporary output directory is created. The cases are then iterated in
// parallel and run via invoke_driver.
fn run_directory(directory_path: PathBuf) -> Vec<(String, String)> {
    let mut files_and_temp_dirs = Vec::new();
    let error_msg = format!("failed to read {:?}", directory_path);
    for entry in fs::read_dir(directory_path).expect(error_msg.as_str()) {
        let entry = entry.unwrap();
        if !entry.file_type().unwrap().is_file() {
            continue;
        };
        let file_path = entry.path();
        let file_name = entry.file_name();
        let temp_dir = TempDir::new().expect("failed to create a temp dir");
        let temp_dir_path_buf = temp_dir.into_path();
        let output_dir_path_buf = temp_dir_path_buf.join(file_name.into_string().unwrap());
        fs::create_dir(output_dir_path_buf.as_path()).expect("failed to create test output dir");
        files_and_temp_dirs.push((
            file_path.into_os_string().into_string().unwrap(),
            output_dir_path_buf.into_os_string().into_string().unwrap(),
        ));
    }
    files_and_temp_dirs
}

fn build_options() -> Options {
    let mut options = Options::default();
    options.parse_from_str(""); // get defaults
    options.diag_level = DiagLevel::Paranoid; // override default
    options.max_analysis_time_for_body = 40;
    options
}

// Partial call graph config to be read from the
// test file
#[derive(Deserialize)]
struct CallGraphTestConfig {
    reductions: Vec<CallGraphReduction>,
    included_crates: Vec<Box<str>>,
}

// Write a call graph configuration file for the current test case
fn generate_call_graph_config(
    file_name: &String,
    temp_dir_path: &String,
) -> (CallGraphConfig, String) {
    let test_case_data =
        fs::read_to_string(Path::new(&file_name)).expect("Failed to read test case");
    let call_graph_test_config: CallGraphTestConfig;
    let config_regex = Regex::new(r"(/\* CONFIG)([\S\s]*?)(\*/)").unwrap();
    if let Some(captures) = config_regex.captures(&test_case_data) {
        assert!(captures.len() == 4);
        call_graph_test_config = serde_json::from_str(&captures[2].to_owned())
            .expect("Failed to deserialize test config");
    } else {
        unrecoverable!("Could not find a call graph config in test file");
    }
    let call_graph_config = CallGraphConfig::new(
        Some(format!("{}/graph.dot", temp_dir_path).into_boxed_str()),
        Some(format!("{}/graph.dat", temp_dir_path).into_boxed_str()),
        Some(format!("{}/types.json", temp_dir_path).into_boxed_str()),
        None,
        call_graph_test_config.reductions,
        call_graph_test_config.included_crates,
    );
    let call_graph_config_path = format!("{}/call_graph_config.json", temp_dir_path);
    let call_graph_config_str =
        serde_json::to_string(&call_graph_config).expect("Failed to serialize config");
    fs::write(Path::new(&call_graph_config_path), call_graph_config_str)
        .expect("Failed to write call graph config");
    (call_graph_config, call_graph_config_path)
}

struct DriverConfig {
    file_name: String,
    temp_dir_path: String,
    extern_deps: Vec<(&'static str, String)>,
}

fn invoke_driver_on_files(
    files_and_temp_dirs: Vec<(String, String)>,
    extern_deps: Vec<(&'static str, String)>,
    driver: &fn(DriverConfig) -> usize,
) -> usize {
    if option_env!("MIRAI_SINGLE").is_some() {
        files_and_temp_dirs
            .into_iter()
            .fold(0, |acc, (file_name, temp_dir_path)| {
                println!("{}", file_name);
                acc + driver(DriverConfig {
                    file_name,
                    temp_dir_path,
                    extern_deps: extern_deps.clone(),
                })
            })
    } else {
        files_and_temp_dirs
            .into_par_iter()
            .fold(
                || 0,
                |acc, (file_name, temp_dir_path)| {
                    acc + driver(DriverConfig {
                        file_name,
                        temp_dir_path,
                        extern_deps: extern_deps.clone(),
                    })
                },
            )
            .reduce(|| 0, |acc, code| acc + code)
    }
}

// Runs the single test case found in file_name, using temp_dir_path as the place
// to put compiler output, which for Mirai includes the persistent summary store.
fn invoke_driver(
    file_name: String,
    temp_dir_path: String,
    sys_root: String,
    extern_deps: Vec<(&str, String)>,
    mut options: Options,
) -> usize {
    let mut rustc_args = vec![]; // any arguments after `--` for rustc
    {
        let file_content = read_to_string(&Path::new(&file_name)).unwrap();
        let options_re = Regex::new(r"(?m)^\s*//\s*MIRAI_FLAGS\s(?P<flags>.*)$").unwrap();
        if let Some(captures) = options_re.captures(&file_content) {
            rustc_args = options.parse_from_str(&captures["flags"]); // override based on test source
        }
    }

    // Setup rustc call.
    let mut command_line_arguments: Vec<String> = vec![
        String::from("--crate-name mirai"),
        file_name.clone(),
        String::from("--crate-type"),
        String::from("lib"),
        String::from("--edition=2018"),
        String::from("-C"),
        String::from("debuginfo=2"),
        String::from("--out-dir"),
        temp_dir_path,
        String::from("--sysroot"),
        sys_root,
        String::from("-Z"),
        String::from("span_free_formats"),
        String::from("-Z"),
        String::from("mir-emit-retag"),
    ];
    command_line_arguments.extend(rustc_args);
    if options.test_only {
        // #[test] will be ignored unless we enable this.
        command_line_arguments.push("--test".to_string());
    }
    for extern_dep in extern_deps {
        command_line_arguments.push("--extern".to_string());
        command_line_arguments.push(format!("{}={}", extern_dep.0, extern_dep.1));
    }

    let mut call_backs = callbacks::MiraiCallbacks::test_runner(options);
    let result = std::panic::catch_unwind(move || {
        let compiler = rustc_driver::RunCompiler::new(&command_line_arguments, &mut call_backs);
        compiler.run()
    });
    match result {
        Ok(_) => 0,
        Err(_) => {
            println!("{} failed", file_name);
            1
        }
    }
}

// Parse expected or actual output into a map
// from trimmed non-emptylines to counts.
fn build_output_counter(output: &String) -> HashMap<&str, u32> {
    let items: Vec<&str> = Vec::from_iter(
        output
            .split('\n')
            .collect::<Vec<&str>>()
            .iter()
            .filter(|x| x.len() > 0)
            .map(|x| x.trim()),
    );
    let mut counter = HashMap::<&str, u32>::new();
    for item in items.iter() {
        *counter.entry(item).or_insert(0) += 1;
    }
    counter
}

// Two outputs are considered equivalent if they
// have the same lines (and counts of each line),
// order-independent.
fn compare_lines(actual: &String, expected: &String) -> bool {
    let actual_counter = build_output_counter(actual);
    let expected_counter = build_output_counter(expected);
    actual_counter == expected_counter
}

// Checked call graph output types
#[derive(Debug, PartialEq)]
enum CallGraphOutputType {
    Dot,
    Ddlog,
    TypeMap,
}

// Check the call graph output files against
// the expected output from the test case file.
fn check_call_graph_output(
    file_name: &String,
    call_graph_config: &CallGraphConfig,
    output_type: CallGraphOutputType,
) -> usize {
    let test_case_data =
        fs::read_to_string(Path::new(&file_name)).expect("Failed to read test case");
    // Check that the expected and actual output files match
    let expected: String;
    let expected_regex = match output_type {
        CallGraphOutputType::Dot => Regex::new(r"(/\* EXPECTED:DOT)([\S\s]*?)(\*/)").unwrap(),
        CallGraphOutputType::Ddlog => Regex::new(r"(/\* EXPECTED:DDLOG)([\S\s]*?)(\*/)").unwrap(),
        CallGraphOutputType::TypeMap => {
            Regex::new(r"(/\* EXPECTED:TYPEMAP)([\S\s]*?)(\*/)").unwrap()
        }
    };
    if let Some(captures) = expected_regex.captures(&test_case_data) {
        assert!(captures.len() == 4);
        expected = captures[2].to_owned();
    } else {
        unrecoverable!("Could not find a call graph config in test file");
    }
    let output_path = match output_type {
        CallGraphOutputType::Dot => call_graph_config.get_dot_path().unwrap(),
        CallGraphOutputType::Ddlog => call_graph_config.get_ddlog_path().unwrap(),
        CallGraphOutputType::TypeMap => call_graph_config.get_type_map_path().unwrap(),
    };
    if let Ok(actual) = fs::read_to_string(output_path) {
        if compare_lines(&expected, &actual) {
            0
        } else {
            println!("{} failed {:?} output", file_name, output_type);
            println!("Expected:\n{}", expected);
            println!("Actual:\n{}", actual);
            1
        }
    } else {
        println!("{} failed dot output", file_name);
        return 1;
    }
}

// Default test driver
fn start_driver(config: DriverConfig) -> usize {
    let sys_root = utils::find_sysroot();
    let options = build_options();
    self::invoke_driver(
        config.file_name,
        config.temp_dir_path,
        sys_root.clone(),
        config.extern_deps,
        options,
    )
}

// Test driver for call graph generation;
// sets up call graph configuration.
fn start_driver_call_graph(config: DriverConfig) -> usize {
    let sys_root = utils::find_sysroot();
    let mut options = build_options();
    let (call_graph_config, call_graph_config_path) =
        generate_call_graph_config(&config.file_name, &config.temp_dir_path);
    options.call_graph_config = Some(call_graph_config_path);
    let result = self::invoke_driver(
        config.file_name.clone(),
        config.temp_dir_path.clone(),
        sys_root.clone(),
        config.extern_deps,
        options,
    );
    if result == 0 {
        check_call_graph_output(
            &config.file_name,
            &call_graph_config,
            CallGraphOutputType::Dot,
        ) + check_call_graph_output(
            &config.file_name,
            &call_graph_config,
            CallGraphOutputType::Ddlog,
        ) + check_call_graph_output(
            &config.file_name,
            &call_graph_config,
            CallGraphOutputType::TypeMap,
        )
    } else {
        result
    }
}
