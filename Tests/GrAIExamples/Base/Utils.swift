//
// Utils.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 19/10/2022.
//

import XCTest
import Foundation
import Cocoa
import PythonKit
import GrAIdient

/// Python library default path.
let PYTHON_LIB =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/miniconda3/envs/graiexamples/lib/libpython3.9.dylib"
let PYTHON_PACKAGES =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/miniconda3/envs/graiexamples/bin/lib/python3.9/site-packages"

/// Environment variable holding the root directory of the pretrained weights.
let WEIGHTS_DIR_VARIABLE = "GRAI_WEIGHTS_DIR"

/// Set the Python library path.
func setPythonLib()
{
    // Keep a library path already provided by the environment.
    if ProcessInfo.processInfo.environment["PYTHON_LIBRARY"] == nil
    {
        setenv("PYTHON_LIBRARY", PYTHON_LIB, 1)
    }
    let sys = Python.import("sys")
    sys.path.insert(0, PYTHON_PACKAGES)
}

///
/// Get the root directory of the pretrained weights.
///
/// The test is skipped when the environment variable is not set.
///
/// - Returns: The root directory.
///
func weightsRoot() throws -> String
{
    guard let root = ProcessInfo.processInfo
        .environment[WEIGHTS_DIR_VARIABLE] else
    {
        throw XCTSkip("\(WEIGHTS_DIR_VARIABLE) is not set.")
    }
    return root
}

///
/// Get the directory of a pretrained model.
///
/// The test is skipped when the environment variable is not set
/// or when the model is not found under the root directory.
///
/// - Parameter model: Name of the model's folder under the root directory.
/// - Returns: The model's directory, with a trailing slash.
///
func modelWeightsPath(model: String) throws -> String
{
    let path = try weightsRoot() + "/" + model + "/"
    guard FileManager.default.fileExists(atPath: path) else
    {
        throw XCTSkip("Weights not found: \(path)")
    }
    return path
}
