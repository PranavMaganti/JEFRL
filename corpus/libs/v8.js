// Copyright 2008 the V8 project authors. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//     * Neither the name of Google Inc. nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function MjsUnitAssertionError(message) {
    this.message = message;
    // Temporarily install a custom stack trace formatter and restore the
    // previous value.
    let prevPrepareStackTrace = Error.prepareStackTrace;
    try {
        Error.prepareStackTrace = MjsUnitAssertionError.prepareStackTrace;
        // This allows fetching the stack trace using TryCatch::StackTrace.
        this.stack = new Error("MjsUnitAssertionError").stack;
    } finally {
        Error.prepareStackTrace = prevPrepareStackTrace;
    }
}

/*
 * This file is included in all mini jsunit test cases.  The test
 * framework expects lines that signal failed tests to start with
 * the f-word and ignore all other lines.
 */

MjsUnitAssertionError.prototype.toString = function () {
    return this.message + "\n\nStack: " + this.stack;
};

// Expected and found values the same objects, or the same primitive
// values.
// For known primitive values, please use assertEquals.
var assertSame;

// Inverse of assertSame.
var assertNotSame;

// Expected and found values are identical primitive values or functions
// or similarly structured objects (checking internal properties
// of, e.g., Number and Date objects, the elements of arrays
// and the properties of non-Array objects).
var assertEquals;

// Deep equality predicate used by assertEquals.
var deepEquals;

// Expected and found values are not identical primitive values or functions
// or similarly structured objects (checking internal properties
// of, e.g., Number and Date objects, the elements of arrays
// and the properties of non-Array objects).
var assertNotEquals;

// The difference between expected and found value is within certain tolerance.
var assertEqualsDelta;

// The found object is an Array with the same length and elements
// as the expected object. The expected object doesn't need to be an Array,
// as long as it's "array-ish".
var assertArrayEquals;

// The found object must have the same enumerable properties as the
// expected object. The type of object isn't checked.
var assertPropertiesEqual;

// Assert that the string conversion of the found value is equal to
// the expected string. Only kept for backwards compatibility, please
// check the real structure of the found value.
var assertToStringEquals;

// Checks that the found value is true. Use with boolean expressions
// for tests that doesn't have their own assertXXX function.
var assertTrue;

// Checks that the found value is false.
var assertFalse;

// Checks that the found value is null. Kept for historical compatibility,
// please just use assertEquals(null, expected).
var assertNull;

// Checks that the found value is *not* null.
var assertNotNull;

// Assert that the passed function or eval code throws an exception.
// The optional second argument is an exception constructor that the
// thrown exception is checked against with "instanceof".
// The optional third argument is a message type string that is compared
// to the type property on the thrown exception.
var assertThrows;

// Assert that the passed function throws an exception.
// The exception is checked against the second argument using assertEquals.
var assertThrowsEquals;

// Assert that the passed function or eval code does not throw an exception.
var assertDoesNotThrow;

// Asserts that the found value is an instance of the constructor passed
// as the second argument.
var assertInstanceof;

// Assert that this code is never executed (i.e., always fails if executed).
var assertUnreachable;

// Assert that the function code is (not) optimized.  If "no sync" is passed
// as second argument, we do not wait for the concurrent optimization thread to
// finish when polling for optimization status.
// Only works with --allow-natives-syntax.
var assertOptimized;
var assertUnoptimized;

// Assert that a string contains another expected substring.
var assertContains;

// Assert that a string matches a given regex.
var assertMatches;

// Assert that a promise resolves or rejects.
// Parameters:
// {promise} - the promise
// {success} - optional - a callback which is called with the result of the
//             resolving promise.
//  {fail} -   optional - a callback which is called with the result of the
//             rejecting promise. If the promise is rejected but no {fail}
//             callback is set, the error is propagated out of the promise
//             chain.
var assertPromiseResult;

var promiseTestChain;
var promiseTestCount = 0;

// These bits must be in sync with bits defined in Runtime_GetOptimizationStatus
var V8OptimizationStatus = {
    kIsFunction: 1 << 0,
    kNeverOptimize: 1 << 1,
    kAlwaysOptimize: 1 << 2,
    kMaybeDeopted: 1 << 3,
    kOptimized: 1 << 4,
    kTurboFanned: 1 << 5,
    kInterpreted: 1 << 6,
    kMarkedForOptimization: 1 << 7,
    kMarkedForConcurrentOptimization: 1 << 8,
    kOptimizingConcurrently: 1 << 9,
    kIsExecuting: 1 << 10,
    kTopmostFrameIsTurboFanned: 1 << 11,
    kLiteMode: 1 << 12,
};

// Returns true if --lite-mode is on and we can't ever turn on optimization.
var isNeverOptimizeLiteMode;

// Returns true if --no-opt mode is on.
var isNeverOptimize;

// Returns true if --always-opt mode is on.
var isAlwaysOptimize;

// Returns true if given function in interpreted.
var isInterpreted;

// Returns true if given function is optimized.
var isOptimized;

// Returns true if given function is compiled by TurboFan.
var isTurboFanned;

// Monkey-patchable all-purpose failure handler.
var failWithMessage;

// Returns the formatted failure text.  Used by test-async.js.
var formatFailureText;

// Returns a pretty-printed string representation of the passed value.
var prettyPrinted;

(function () {
    // Scope for utility functions.

    var ObjectPrototypeToString = Object.prototype.toString;
    var NumberPrototypeValueOf = Number.prototype.valueOf;
    var BooleanPrototypeValueOf = Boolean.prototype.valueOf;
    var StringPrototypeValueOf = String.prototype.valueOf;
    var DatePrototypeValueOf = Date.prototype.valueOf;
    var RegExpPrototypeToString = RegExp.prototype.toString;
    var ArrayPrototypeForEach = Array.prototype.forEach;
    var ArrayPrototypeJoin = Array.prototype.join;
    var ArrayPrototypeMap = Array.prototype.map;
    var ArrayPrototypePush = Array.prototype.push;

    var BigIntPrototypeValueOf;
    // TODO(neis): Remove try-catch once BigInts are enabled by default.
    try {
        BigIntPrototypeValueOf = BigInt.prototype.valueOf;
    } catch (e) {}

    function classOf(object) {
        // Argument must not be null or undefined.
        var string = ObjectPrototypeToString.call(object);
        // String has format [object <ClassName>].
        return string.substring(8, string.length - 1);
    }

    function ValueOf(value) {
        switch (classOf(value)) {
            case "Number":
                return NumberPrototypeValueOf.call(value);
            case "BigInt":
                return BigIntPrototypeValueOf.call(value);
            case "String":
                return StringPrototypeValueOf.call(value);
            case "Boolean":
                return BooleanPrototypeValueOf.call(value);
            case "Date":
                return DatePrototypeValueOf.call(value);
            default:
                return value;
        }
    }

    prettyPrinted = function prettyPrinted(value) {
        switch (typeof value) {
            case "string":
                return JSON.stringify(value);
            case "bigint":
                return String(value) + "n";
            case "number":
                if (value === 0 && 1 / value < 0) return "-0";
            // FALLTHROUGH.
            case "boolean":
            case "undefined":
            case "function":
            case "symbol":
                return String(value);
            case "object":
                if (value === null) return "null";
                var objectClass = classOf(value);
                switch (objectClass) {
                    case "Number":
                    case "BigInt":
                    case "String":
                    case "Boolean":
                    case "Date":
                        return (
                            objectClass +
                            "(" +
                            prettyPrinted(ValueOf(value)) +
                            ")"
                        );
                    case "RegExp":
                        return RegExpPrototypeToString.call(value);
                    case "Array":
                        var mapped = ArrayPrototypeMap.call(
                            value,
                            prettyPrintedArrayElement
                        );
                        var joined = ArrayPrototypeJoin.call(mapped, ",");
                        return "[" + joined + "]";
                    case "Uint8Array":
                    case "Int8Array":
                    case "Int16Array":
                    case "Uint16Array":
                    case "Uint32Array":
                    case "Int32Array":
                    case "Float32Array":
                    case "Float64Array":
                        var joined = ArrayPrototypeJoin.call(value, ",");
                        return objectClass + "([" + joined + "])";
                    case "Object":
                        break;
                    default:
                        return objectClass + "(" + String(value) + ")";
                }
                // [[Class]] is "Object".
                var name = value.constructor.name;
                if (name) return name + "()";
                return "Object()";
            default:
                return "-- unknown value --";
        }
    };

    function prettyPrintedArrayElement(value, index, array) {
        if (value === undefined && !(index in array)) return "";
        return prettyPrinted(value);
    }

    failWithMessage = function failWithMessage(message) {
        throw new MjsUnitAssertionError(message);
    };

    formatFailureText = function (expectedText, found, name_opt) {
        var message = "Fail" + "ure";
        if (name_opt) {
            // Fix this when we ditch the old test runner.
            message += " (" + name_opt + ")";
        }

        var foundText = prettyPrinted(found);
        if (expectedText.length <= 40 && foundText.length <= 40) {
            message +=
                ": expected <" + expectedText + "> found <" + foundText + ">";
        } else {
            message +=
                ":\nexpected:\n" + expectedText + "\nfound:\n" + foundText;
        }
        return message;
    };

    function fail(expectedText, found, name_opt) {
        return failWithMessage(
            formatFailureText(expectedText, found, name_opt)
        );
    }

    function deepObjectEquals(a, b) {
        var aProps = Object.keys(a);
        aProps.sort();
        var bProps = Object.keys(b);
        bProps.sort();
        if (!deepEquals(aProps, bProps)) {
            return false;
        }
        for (var i = 0; i < aProps.length; i++) {
            if (!deepEquals(a[aProps[i]], b[aProps[i]])) {
                return false;
            }
        }
        return true;
    }

    deepEquals = function deepEquals(a, b) {
        if (a === b) {
            // Check for -0.
            if (a === 0) return 1 / a === 1 / b;
            return true;
        }
        if (typeof a !== typeof b) return false;
        if (typeof a === "number") return isNaN(a) && isNaN(b);
        if (typeof a !== "object" && typeof a !== "function") return false;
        // Neither a nor b is primitive.
        var objectClass = classOf(a);
        if (objectClass !== classOf(b)) return false;
        if (objectClass === "RegExp") {
            // For RegExp, just compare pattern and flags using its toString.
            return (
                RegExpPrototypeToString.call(a) ===
                RegExpPrototypeToString.call(b)
            );
        }
        // Functions are only identical to themselves.
        if (objectClass === "Function") return false;
        if (objectClass === "Array") {
            var elementCount = 0;
            if (a.length !== b.length) {
                return false;
            }
            for (var i = 0; i < a.length; i++) {
                if (!deepEquals(a[i], b[i])) return false;
            }
            return true;
        }
        if (
            objectClass === "String" ||
            objectClass === "Number" ||
            objectClass === "BigInt" ||
            objectClass === "Boolean" ||
            objectClass === "Date"
        ) {
            if (ValueOf(a) !== ValueOf(b)) return false;
        }
        return deepObjectEquals(a, b);
    };

    assertSame = function assertSame(expected, found, name_opt) {
        // TODO(mstarzinger): We should think about using Harmony's egal operator
        // or the function equivalent Object.is() here.
        if (found === expected) {
            if (expected !== 0 || 1 / expected === 1 / found) return;
        } else if (expected !== expected && found !== found) {
            return;
        }
        fail(prettyPrinted(expected), found, name_opt);
    };

    assertNotSame = function assertNotSame(expected, found, name_opt) {
        // TODO(mstarzinger): We should think about using Harmony's egal operator
        // or the function equivalent Object.is() here.
        if (found !== expected) {
            if (expected === 0 || 1 / expected !== 1 / found) return;
        } else if (!(expected !== expected && found !== found)) {
            return;
        }
        fail(prettyPrinted(expected), found, name_opt);
    };

    assertEquals = function assertEquals(expected, found, name_opt) {
        if (!deepEquals(found, expected)) {
            fail(prettyPrinted(expected), found, name_opt);
        }
    };

    assertNotEquals = function assertNotEquals(expected, found, name_opt) {
        if (deepEquals(found, expected)) {
            fail("not equals to " + prettyPrinted(expected), found, name_opt);
        }
    };

    assertEqualsDelta = function assertEqualsDelta(
        expected,
        found,
        delta,
        name_opt
    ) {
        if (Math.abs(expected - found) > delta) {
            fail(
                prettyPrinted(expected) + " +- " + prettyPrinted(delta),
                found,
                name_opt
            );
        }
    };

    assertArrayEquals = function assertArrayEquals(expected, found, name_opt) {
        var start = "";
        if (name_opt) {
            start = name_opt + " - ";
        }
        assertEquals(expected.length, found.length, start + "array length");
        if (expected.length === found.length) {
            for (var i = 0; i < expected.length; ++i) {
                assertEquals(
                    expected[i],
                    found[i],
                    start + "array element at index " + i
                );
            }
        }
    };

    assertPropertiesEqual = function assertPropertiesEqual(
        expected,
        found,
        name_opt
    ) {
        // Check properties only.
        if (!deepObjectEquals(expected, found)) {
            fail(expected, found, name_opt);
        }
    };

    assertToStringEquals = function assertToStringEquals(
        expected,
        found,
        name_opt
    ) {
        if (expected !== String(found)) {
            fail(expected, found, name_opt);
        }
    };

    assertTrue = function assertTrue(value, name_opt) {
        assertEquals(true, value, name_opt);
    };

    assertFalse = function assertFalse(value, name_opt) {
        assertEquals(false, value, name_opt);
    };

    assertNull = function assertNull(value, name_opt) {
        if (value !== null) {
            fail("null", value, name_opt);
        }
    };

    assertNotNull = function assertNotNull(value, name_opt) {
        if (value === null) {
            fail("not null", value, name_opt);
        }
    };

    assertThrows = function assertThrows(code, type_opt, cause_opt) {
        try {
            if (typeof code === "function") {
                code();
            } else {
                eval(code);
            }
        } catch (e) {
            if (typeof type_opt === "function") {
                assertInstanceof(e, type_opt);
            } else if (type_opt !== void 0) {
                failWithMessage(
                    "invalid use of assertThrows, maybe you want assertThrowsEquals"
                );
            }
            if (arguments.length >= 3) {
                if (cause_opt instanceof RegExp) {
                    assertMatches(cause_opt, e.message, "Error message");
                } else {
                    assertEquals(cause_opt, e.message, "Error message");
                }
            }
            // Success.
            return;
        }
        failWithMessage("Did not throw exception");
    };

    assertThrowsEquals = function assertThrowsEquals(fun, val) {
        try {
            fun();
        } catch (e) {
            assertSame(val, e);
            return;
        }
        failWithMessage("Did not throw exception");
    };

    assertInstanceof = function assertInstanceof(obj, type) {
        if (!(obj instanceof type)) {
            var actualTypeName = null;
            var actualConstructor = Object.getPrototypeOf(obj).constructor;
            if (typeof actualConstructor === "function") {
                actualTypeName =
                    actualConstructor.name || String(actualConstructor);
            }
            failWithMessage(
                "Object <" +
                    prettyPrinted(obj) +
                    "> is not an instance of <" +
                    (type.name || type) +
                    ">" +
                    (actualTypeName ? " but of <" + actualTypeName + ">" : "")
            );
        }
    };

    assertDoesNotThrow = function assertDoesNotThrow(code, name_opt) {
        try {
            if (typeof code === "function") {
                return code();
            } else {
                return eval(code);
            }
        } catch (e) {
            failWithMessage("threw an exception: " + (e.message || e));
        }
    };

    assertUnreachable = function assertUnreachable(name_opt) {
        // Fix this when we ditch the old test runner.
        var message = "Fail" + "ure: unreachable";
        if (name_opt) {
            message += " - " + name_opt;
        }
        failWithMessage(message);
    };

    assertContains = function (sub, value, name_opt) {
        if (value == null ? sub != null : value.indexOf(sub) == -1) {
            fail("contains '" + String(sub) + "'", value, name_opt);
        }
    };

    assertMatches = function (regexp, str, name_opt) {
        if (!(regexp instanceof RegExp)) {
            regexp = new RegExp(regexp);
        }
        if (!str.match(regexp)) {
            fail("should match '" + regexp + "'", str, name_opt);
        }
    };

    function concatenateErrors(stack, exception) {
        // If the exception does not contain a stack trace, wrap it in a new Error.
        if (!exception.stack) exception = new Error(exception);

        // If the exception already provides a special stack trace, we do not modify
        // it.
        if (typeof exception.stack !== "string") {
            return exception;
        }
        exception.stack = stack + "\n\n" + exception.stack;
        return exception;
    }

    assertPromiseResult = function (promise, success, fail) {
        const stack = new Error().stack;

        var test_promise = promise.then(
            (result) => {
                try {
                    if (--promiseTestCount == 0) {
                    }
                    if (success) success(result);
                } catch (e) {
                    // Use setTimeout to throw the error again to get out of the promise
                    // chain.
                    setTimeout((_) => {
                        throw concatenateErrors(stack, e);
                    }, 0);
                }
            },
            (result) => {
                try {
                    if (--promiseTestCount == 0) {
                    }
                    if (!fail) throw result;
                    fail(result);
                } catch (e) {
                    // Use setTimeout to throw the error again to get out of the promise
                    // chain.
                    setTimeout((_) => {
                        throw concatenateErrors(stack, e);
                    }, 0);
                }
            }
        );

        if (!promiseTestChain) promiseTestChain = Promise.resolve();
        // waitUntilDone is idempotent.
        ++promiseTestCount;
        return promiseTestChain.then(test_promise);
    };

    var OptimizationStatusImpl = undefined;

    var OptimizationStatus = function (fun, sync_opt) {
        if (OptimizationStatusImpl === undefined) {
            try {
                OptimizationStatusImpl = new Function(
                    "fun",
                    "sync",
                    "return %GetOptimizationStatus(fun, sync);"
                );
            } catch (e) {
                throw new Error("natives syntax not allowed");
            }
        }
        return OptimizationStatusImpl(fun, sync_opt);
    };

    assertUnoptimized = function assertUnoptimized(
        fun,
        sync_opt,
        name_opt,
        skip_if_maybe_deopted = true
    ) {
        if (sync_opt === undefined) sync_opt = "";
        var opt_status = OptimizationStatus(fun, sync_opt);
        // Tests that use assertUnoptimized() do not make sense if --always-opt
        // option is provided. Such tests must add --no-always-opt to flags comment.
        assertFalse(
            (opt_status & V8OptimizationStatus.kAlwaysOptimize) !== 0,
            "test does not make sense with --always-opt"
        );
        assertTrue(
            (opt_status & V8OptimizationStatus.kIsFunction) !== 0,
            name_opt
        );
        if (
            skip_if_maybe_deopted &&
            (opt_status & V8OptimizationStatus.kMaybeDeopted) !== 0
        ) {
            // When --deopt-every-n-times flag is specified it's no longer guaranteed
            // that particular function is still deoptimized, so keep running the test
            // to stress test the deoptimizer.
            return;
        }
        assertFalse(
            (opt_status & V8OptimizationStatus.kOptimized) !== 0,
            name_opt
        );
    };

    assertOptimized = function assertOptimized(
        fun,
        sync_opt,
        name_opt,
        skip_if_maybe_deopted = true
    ) {
        if (sync_opt === undefined) sync_opt = "";
        var opt_status = OptimizationStatus(fun, sync_opt);
        // Tests that use assertOptimized() do not make sense for Lite mode where
        // optimization is always disabled, explicitly exit the test with a warning.
        if (opt_status & V8OptimizationStatus.kLiteMode) {
            print(
                "Warning: Test uses assertOptimized in Lite mode, skipping test."
            );
            quit(0);
        }
        // Tests that use assertOptimized() do not make sense if --no-opt
        // option is provided. Such tests must add --opt to flags comment.
        assertFalse(
            (opt_status & V8OptimizationStatus.kNeverOptimize) !== 0,
            "test does not make sense with --no-opt"
        );
        assertTrue(
            (opt_status & V8OptimizationStatus.kIsFunction) !== 0,
            name_opt
        );
        if (
            skip_if_maybe_deopted &&
            (opt_status & V8OptimizationStatus.kMaybeDeopted) !== 0
        ) {
            // When --deopt-every-n-times flag is specified it's no longer guaranteed
            // that particular function is still optimized, so keep running the test
            // to stress test the deoptimizer.
            return;
        }
        assertTrue(
            (opt_status & V8OptimizationStatus.kOptimized) !== 0,
            name_opt
        );
    };

    isNeverOptimizeLiteMode = function isNeverOptimizeLiteMode() {
        var opt_status = OptimizationStatus(undefined, "");
        return (opt_status & V8OptimizationStatus.kLiteMode) !== 0;
    };

    isNeverOptimize = function isNeverOptimize() {
        var opt_status = OptimizationStatus(undefined, "");
        return (opt_status & V8OptimizationStatus.kNeverOptimize) !== 0;
    };

    isAlwaysOptimize = function isAlwaysOptimize() {
        var opt_status = OptimizationStatus(undefined, "");
        return (opt_status & V8OptimizationStatus.kAlwaysOptimize) !== 0;
    };

    isInterpreted = function isInterpreted(fun) {
        var opt_status = OptimizationStatus(fun, "");
        assertTrue(
            (opt_status & V8OptimizationStatus.kIsFunction) !== 0,
            "not a function"
        );
        return (
            (opt_status & V8OptimizationStatus.kOptimized) === 0 &&
            (opt_status & V8OptimizationStatus.kInterpreted) !== 0
        );
    };

    isOptimized = function isOptimized(fun) {
        var opt_status = OptimizationStatus(fun, "");
        assertTrue(
            (opt_status & V8OptimizationStatus.kIsFunction) !== 0,
            "not a function"
        );
        return (opt_status & V8OptimizationStatus.kOptimized) !== 0;
    };

    isTurboFanned = function isTurboFanned(fun) {
        var opt_status = OptimizationStatus(fun, "");
        assertTrue(
            (opt_status & V8OptimizationStatus.kIsFunction) !== 0,
            "not a function"
        );
        return (
            (opt_status & V8OptimizationStatus.kOptimized) !== 0 &&
            (opt_status & V8OptimizationStatus.kTurboFanned) !== 0
        );
    };

    // Custom V8-specific stack trace formatter that is temporarily installed on
    // the Error object.
    MjsUnitAssertionError.prepareStackTrace = function (error, stack) {
        // Trigger default formatting with recursion.
        try {
            // Filter-out all but the first mjsunit frame.
            let filteredStack = [];
            let inMjsunit = true;
            for (let i = 0; i < stack.length; i++) {
                let frame = stack[i];
                if (inMjsunit) {
                    let file = frame.getFileName();
                    if (!file || !file.endsWith("mjsunit.js")) {
                        inMjsunit = false;
                        // Push the last mjsunit frame, typically containing the assertion
                        // function.
                        if (i > 0)
                            ArrayPrototypePush.call(
                                filteredStack,
                                stack[i - 1]
                            );
                        ArrayPrototypePush.call(filteredStack, stack[i]);
                    }
                    continue;
                }
                ArrayPrototypePush.call(filteredStack, frame);
            }
            stack = filteredStack;

            // Infer function names and calculate {max_name_length}
            let max_name_length = 0;
            ArrayPrototypeForEach.call(stack, (each) => {
                let name = each.getFunctionName();
                if (name == null) name = "";
                if (each.isEval()) {
                    name = name;
                } else if (each.isConstructor()) {
                    name = "new " + name;
                } else if (each.isNative()) {
                    name = "native " + name;
                } else if (!each.isToplevel()) {
                    name = each.getTypeName() + "." + name;
                }
                each.name = name;
                max_name_length = Math.max(name.length, max_name_length);
            });

            // Format stack frames.
            stack = ArrayPrototypeMap.call(stack, (each) => {
                let frame = "    at " + each.name.padEnd(max_name_length);
                let fileName = each.getFileName();
                if (each.isEval()) return frame + " " + each.getEvalOrigin();
                frame += " " + (fileName ? fileName : "");
                let line = each.getLineNumber();
                frame += " " + (line ? line : "");
                let column = each.getColumnNumber();
                frame += column ? ":" + column : "";
                return frame;
            });
            return (
                "" + error.message + "\n" + ArrayPrototypeJoin.call(stack, "\n")
            );
        } catch (e) {}
        return error.stack;
    };
})();

function f() {
    return [];
}
function f0() {
    return true;
}
function f1() {
    return 0.0;
}
function f2(v) {
    return v;
}
let TestCoverage;
let TestCoverageNoGC;

let nop;
let gen;

!(function () {
    function GetCoverage(source) {
        return undefined;
    }

    function TestCoverageInternal(name, source, expectation, collect_garbage) {
        source = source.trim();
        eval(source);
        var covfefe = GetCoverage(source);
        var stringified_result = JSON.stringify(covfefe);
        var stringified_expectation = JSON.stringify(expectation);
        if (stringified_result != stringified_expectation) {
            print(stringified_result.replace(/[}],[{]/g, "},\n {"));
        }
        assertEquals(
            stringified_expectation,
            stringified_result,
            name + " failed"
        );
    }

    TestCoverage = function (name, source, expectation) {
        TestCoverageInternal(name, source, expectation, true);
    };

    TestCoverageNoGC = function (name, source, expectation) {
        TestCoverageInternal(name, source, expectation, false);
    };

    nop = function () {};

    gen = function* () {
        yield 1;
        yield 2;
        yield 3;
    };
})();

function isOneByteString(s) {
    return s[0];
}

const regexp = "/P{Lu}/ui";
const regexpu =
    "/[\0-@[-\xBF\xD7\xDF-\xFF\u0101\u0103\u0105\u0107\u0109\u010B\u010D\u010F\u0111\u0113\u0115\u0117\u0119\u011B\u011D\u011F\u0121\u0123\u0125\u0127\u0129\u012B\u012D\u012F\u0131\u0133\u0135\u0137\u0138\u013A\u013C\u013E\u0140\u0142\u0144\u0146\u0148\u0149\u014B\u014D\u014F\u0151\u0153\u0155\u0157\u0159\u015B\u015D\u015F\u0161\u0163\u0165\u0167\u0169\u016B\u016D\u016F\u0171\u0173\u0175\u0177\u017A\u017C\u017E-\u0180\u0183\u0185\u0188\u018C\u018D\u0192\u0195\u0199-\u019B\u019E\u01A1\u01A3\u01A5\u01A8\u01AA\u01AB\u01AD\u01B0\u01B4\u01B6\u01B9-\u01BB\u01BD-\u01C3\u01C5\u01C6\u01C8\u01C9\u01CB\u01CC\u01CE\u01D0\u01D2\u01D4\u01D6\u01D8\u01DA\u01DC\u01DD\u01DF\u01E1\u01E3\u01E5\u01E7\u01E9\u01EB\u01ED\u01EF\u01F0\u01F2\u01F3\u01F5\u01F9\u01FB\u01FD\u01FF\u0201\u0203\u0205\u0207\u0209\u020B\u020D\u020F\u0211\u0213\u0215\u0217\u0219\u021B\u021D\u021F\u0221\u0223\u0225\u0227\u0229\u022B\u022D\u022F\u0231\u0233-\u0239\u023C\u023F\u0240\u0242\u0247\u0249\u024B\u024D\u024F-\u036F\u0371\u0373-\u0375\u0377-\u037E\u0380-\u0385\u0387\u038B\u038D\u0390\u03A2\u03AC-\u03CE\u03D0\u03D1\u03D5-\u03D7\u03D9\u03DB\u03DD\u03DF\u03E1\u03E3\u03E5\u03E7\u03E9\u03EB\u03ED\u03EF-\u03F3\u03F5\u03F6\u03F8\u03FB\u03FC\u0430-\u045F\u0461\u0463\u0465\u0467\u0469\u046B\u046D\u046F\u0471\u0473\u0475\u0477\u0479\u047B\u047D\u047F\u0481-\u0489\u048B\u048D\u048F\u0491\u0493\u0495\u0497\u0499\u049B\u049D\u049F\u04A1\u04A3\u04A5\u04A7\u04A9\u04AB\u04AD\u04AF\u04B1\u04B3\u04B5\u04B7\u04B9\u04BB\u04BD\u04BF\u04C2\u04C4\u04C6\u04C8\u04CA\u04CC\u04CE\u04CF\u04D1\u04D3\u04D5\u04D7\u04D9\u04DB\u04DD\u04DF\u04E1\u04E3\u04E5\u04E7\u04E9\u04EB\u04ED\u04EF\u04F1\u04F3\u04F5\u04F7\u04F9\u04FB\u04FD\u04FF\u0501\u0503\u0505\u0507\u0509\u050B\u050D\u050F\u0511\u0513\u0515\u0517\u0519\u051B\u051D\u051F\u0521\u0523\u0525\u0527\u0529\u052B\u052D\u052F\u0530\u0557-\u109F\u10C6\u10C8-\u10CC\u10CE-\u139F\u13F6-\u1DFF\u1E01\u1E03\u1E05\u1E07\u1E09\u1E0B\u1E0D\u1E0F\u1E11\u1E13\u1E15\u1E17\u1E19\u1E1B\u1E1D\u1E1F\u1E21\u1E23\u1E25\u1E27\u1E29\u1E2B\u1E2D\u1E2F\u1E31\u1E33\u1E35\u1E37\u1E39\u1E3B\u1E3D\u1E3F\u1E41\u1E43\u1E45\u1E47\u1E49\u1E4B\u1E4D\u1E4F\u1E51\u1E53\u1E55\u1E57\u1E59\u1E5B\u1E5D\u1E5F\u1E61\u1E63\u1E65\u1E67\u1E69\u1E6B\u1E6D\u1E6F\u1E71\u1E73\u1E75\u1E77\u1E79\u1E7B\u1E7D\u1E7F\u1E81\u1E83\u1E85\u1E87\u1E89\u1E8B\u1E8D\u1E8F\u1E91\u1E93\u1E95-\u1E9D\u1E9F\u1EA1\u1EA3\u1EA5\u1EA7\u1EA9\u1EAB\u1EAD\u1EAF\u1EB1\u1EB3\u1EB5\u1EB7\u1EB9\u1EBB\u1EBD\u1EBF\u1EC1\u1EC3\u1EC5\u1EC7\u1EC9\u1ECB\u1ECD\u1ECF\u1ED1\u1ED3\u1ED5\u1ED7\u1ED9\u1EDB\u1EDD\u1EDF\u1EE1\u1EE3\u1EE5\u1EE7\u1EE9\u1EEB\u1EED\u1EEF\u1EF1\u1EF3\u1EF5\u1EF7\u1EF9\u1EFB\u1EFD\u1EFF-\u1F07\u1F10-\u1F17\u1F1E-\u1F27\u1F30-\u1F37\u1F40-\u1F47\u1F4E-\u1F58\u1F5A\u1F5C\u1F5E\u1F60-\u1F67\u1F70-\u1FB7\u1FBC-\u1FC7\u1FCC-\u1FD7\u1FDC-\u1FE7\u1FED-\u1FF7\u1FFC-\u2101\u2103-\u2106\u2108-\u210A\u210E\u210F\u2113\u2114\u2116-\u2118\u211E-\u2123\u2125\u2127\u2129\u212E\u212F\u2134-\u213D\u2140-\u2144\u2146-\u2182\u2184-\u2BFF\u2C2F-\u2C5F\u2C61\u2C65\u2C66\u2C68\u2C6A\u2C6C\u2C71\u2C73\u2C74\u2C76-\u2C7D\u2C81\u2C83\u2C85\u2C87\u2C89\u2C8B\u2C8D\u2C8F\u2C91\u2C93\u2C95\u2C97\u2C99\u2C9B\u2C9D\u2C9F\u2CA1\u2CA3\u2CA5\u2CA7\u2CA9\u2CAB\u2CAD\u2CAF\u2CB1\u2CB3\u2CB5\u2CB7\u2CB9\u2CBB\u2CBD\u2CBF\u2CC1\u2CC3\u2CC5\u2CC7\u2CC9\u2CCB\u2CCD\u2CCF\u2CD1\u2CD3\u2CD5\u2CD7\u2CD9\u2CDB\u2CDD\u2CDF\u2CE1\u2CE3-\u2CEA\u2CEC\u2CEE-\u2CF1\u2CF3-\uA63F\uA641\uA643\uA645\uA647\uA649\uA64B\uA64D\uA64F\uA651\uA653\uA655\uA657\uA659\uA65B\uA65D\uA65F\uA661\uA663\uA665\uA667\uA669\uA66B\uA66D-\uA67F\uA681\uA683\uA685\uA687\uA689\uA68B\uA68D\uA68F\uA691\uA693\uA695\uA697\uA699\uA69B-\uA721\uA723\uA725\uA727\uA729\uA72B\uA72D\uA72F-\uA731\uA733\uA735\uA737\uA739\uA73B\uA73D\uA73F\uA741\uA743\uA745\uA747\uA749\uA74B\uA74D\uA74F\uA751\uA753\uA755\uA757\uA759\uA75B\uA75D\uA75F\uA761\uA763\uA765\uA767\uA769\uA76B\uA76D\uA76F-\uA778\uA77A\uA77C\uA77F\uA781\uA783\uA785\uA787-\uA78A\uA78C\uA78E\uA78F\uA791\uA793-\uA795\uA797\uA799\uA79B\uA79D\uA79F\uA7A1\uA7A3\uA7A5\uA7A7\uA7A9\uA7AE\uA7AF\uA7B5\uA7B7-\uFF20\uFF3B-\u{103FF}\u{10428}-\u{10C7F}\u{10CB3}-\u{1189F}\u{118C0}-\u{1D3FF}\u{1D41A}-\u{1D433}\u{1D44E}-\u{1D467}\u{1D482}-\u{1D49B}\u{1D49D}\u{1D4A0}\u{1D4A1}\u{1D4A3}\u{1D4A4}\u{1D4A7}\u{1D4A8}\u{1D4AD}\u{1D4B6}-\u{1D4CF}\u{1D4EA}-\u{1D503}\u{1D506}\u{1D50B}\u{1D50C}\u{1D515}\u{1D51D}-\u{1D537}\u{1D53A}\u{1D53F}\u{1D545}\u{1D547}-\u{1D549}\u{1D551}-\u{1D56B}\u{1D586}-\u{1D59F}\u{1D5BA}-\u{1D5D3}\u{1D5EE}-\u{1D607}\u{1D622}-\u{1D63B}\u{1D656}-\u{1D66F}\u{1D68A}-\u{1D6A7}\u{1D6C1}-\u{1D6E1}\u{1D6FB}-\u{1D71B}\u{1D735}-\u{1D755}\u{1D76F}-\u{1D78F}\u{1D7A9}-\u{1D7C9}\u{1D7CB}-\u{10FFFF}]/ui";

// Test is split into parts to increase parallelism.
const number_of_tests = 10;
const max_codepoint = 0x10ffff;

function firstCodePointOfRange(i) {
    return Math.floor(i * (max_codepoint / number_of_tests));
}

function testCodePointRange(i) {
    assertTrue(i >= 0 && i < number_of_tests);

    const from = firstCodePointOfRange(i);
    const to =
        i == number_of_tests - 1
            ? max_codepoint + 1
            : firstCodePointOfRange(i + 1);

    for (let codePoint = from; codePoint < to; codePoint++) {
        const string = String.fromCodePoint(codePoint);
        assertEquals(regexp.test(string), regexpu.test(string));
    }
}
if (gc == undefined) {
    function gc() {
        for (let i = 0; i < 10; i++) {
            new ArrayBuffer(1024 * 1024 * 10);
        }
    }
}
if (BigInt == undefined)
    function BigInt(v) {
        return new Number(v);
    }
if (BigInt64Array == undefined)
    function BigInt64Array(v) {
        return new Array(v);
    }
if (BigUint64Array == undefined)
    function BigUint64Array(v) {
        return new Array(v);
    }

if (typeof console == "undefined") {
    console = {
        log: print,
    };
}

if (typeof gc == "undefined") {
    gc = function () {
        for (let i = 0; i < 10; i++) {
            new ArrayBuffer(1024 * 1024 * 10);
        }
    };
}

if (typeof BigInt == "undefined") {
    BigInt = function (v) {
        return new Number(v);
    };
}

if (typeof BigInt64Array == "undefined") {
    BigInt64Array = function (v) {
        return new Array(v);
    };
}

if (typeof BigUint64Array == "undefined") {
    BigUint64Array = function (v) {
        return new Array(v);
    };
}

if (typeof quit == "undefined") {
    quit = function () {};
}

function noInline() {}

function OSRExit() {}

function ensureArrayStorage() {}

function fiatInt52(i) {
    return i;
}

function noDFG() {}

function noOSRExitFuzzing() {}

function isFinalTier() {
    return true;
}

function transferArrayBuffer() {}

function fullGC() {
    gc();
}

function edenGC() {
    gc();
}

function forceGCSlowPaths() {
    gc();
}

function noFTL() {}

function debug(x) {
    console.log(x);
}

function describe(x) {
    console.log(x);
}

function isInt32(i) {
    return typeof i === "number";
}

/* -*- indent-tabs-mode: nil; js-indent-level: 2 -*-
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// NOTE: If you're adding new test harness functionality -- first, should you
//       at all?  Most stuff is better in specific tests, or in nested shell.js
//       or browser.js.  Second, supposing you should, please add it to this
//       IIFE for better modularity/resilience against tests that must do
//       particularly bizarre things that might break the harness.

(function (global) {
    "use strict";

    /**********************************************************************
     * CACHED PRIMORDIAL FUNCTIONALITY (before a test might overwrite it) *
     **********************************************************************/

    var undefined; // sigh

    var Error = global.Error;
    var Function = global.Function;
    var Number = global.Number;
    var RegExp = global.RegExp;
    var String = global.String;
    var Symbol = global.Symbol;
    var TypeError = global.TypeError;

    var ArrayIsArray = global.Array.isArray;
    var MathAbs = global.Math.abs;
    var ObjectCreate = global.Object.create;
    var ObjectDefineProperty = global.Object.defineProperty;
    var ReflectApply = global.Reflect.apply;
    var RegExpPrototypeExec = global.RegExp.prototype.exec;
    var StringPrototypeCharCodeAt = global.String.prototype.charCodeAt;
    var StringPrototypeIndexOf = global.String.prototype.indexOf;
    var StringPrototypeSubstring = global.String.prototype.substring;

    global.Array.prototype.toSource = function () {
        return this.toString();
    };
    var runningInBrowser = typeof global.window !== "undefined";
    if (runningInBrowser) {
        // Certain cached functionality only exists (and is only needed) when
        // running in the browser.  Segregate that caching here.

        var SpecialPowersSetGCZeal = global.SpecialPowers
            ? global.SpecialPowers.setGCZeal
            : undefined;
    }

    var evaluate = global.evaluate;
    var options = global.options;

    /****************************
     * GENERAL HELPER FUNCTIONS *
     ****************************/

    // We *cannot* use Array.prototype.push for this, because that function sets
    // the new trailing element, which could invoke a setter (left by a test) on
    // Array.prototype or Object.prototype.
    function ArrayPush(arr, val) {
        assertEq(
            ArrayIsArray(arr),
            true,
            "ArrayPush must only be used on actual arrays"
        );

        var desc = ObjectCreate(null);
        desc.value = val;
        desc.enumerable = true;
        desc.configurable = true;
        desc.writable = true;
        ObjectDefineProperty(arr, arr.length, desc);
    }

    function StringCharCodeAt(str, index) {
        return ReflectApply(StringPrototypeCharCodeAt, str, [index]);
    }

    function StringSplit(str, delimiter) {
        assertEq(
            typeof str === "string" && typeof delimiter === "string",
            true,
            "StringSplit must be called with two string arguments"
        );
        assertEq(
            delimiter.length > 0,
            true,
            "StringSplit doesn't support an empty delimiter string"
        );

        var parts = [];
        var last = 0;
        while (true) {
            var i = ReflectApply(StringPrototypeIndexOf, str, [
                delimiter,
                last,
            ]);
            if (i < 0) {
                if (last < str.length)
                    ArrayPush(
                        parts,
                        ReflectApply(StringPrototypeSubstring, str, [last])
                    );
                return parts;
            }

            ArrayPush(
                parts,
                ReflectApply(StringPrototypeSubstring, str, [last, i])
            );
            last = i + delimiter.length;
        }
    }

    function shellOptionsClear() {
        assertEq(
            runningInBrowser,
            false,
            "Only called when running in the shell."
        );

        // Return early if no options are set.
        var currentOptions = options ? options() : "";
        if (currentOptions === "") return;

        // Turn off current settings.
        var optionNames = StringSplit(currentOptions, ",");
        for (var i = 0; i < optionNames.length; i++) {
            options(optionNames[i]);
        }
    }

    /****************************
     * TESTING FUNCTION EXPORTS *
     ****************************/

    function SameValue(v1, v2) {
        // We could |return Object.is(v1, v2);|, but that's less portable.
        if (v1 === 0 && v2 === 0) return 1 / v1 === 1 / v2;
        if (v1 !== v1 && v2 !== v2) return true;
        return v1 === v2;
    }

    var assertEq = global.assertEq;
    if (typeof assertEq !== "function") {
        assertEq = function assertEq(actual, expected, message) {
            if (!SameValue(actual, expected)) {
                throw new TypeError(
                    `Assertion failed: got "${actual}", expected "${expected}"` +
                        (message ? ": " + message : "")
                );
            }
        };
        global.assertEq = assertEq;
    }

    function assertEqArray(actual, expected) {
        var len = actual.length;
        assertEq(len, expected.length, "mismatching array lengths");

        var i = 0;
        try {
            for (; i < len; i++)
                assertEq(actual[i], expected[i], "mismatch at element " + i);
        } catch (e) {
            throw new Error(`Exception thrown at index ${i}: ${e}`);
        }
    }
    global.assertEqArray = assertEqArray;

    function assertThrows(f) {
        var ok = false;
        try {
            f();
        } catch (exc) {
            ok = true;
        }
        if (!ok)
            throw new Error(`Assertion failed: ${f} did not throw as expected`);
    }
    global.assertThrows = assertThrows;

    function assertThrowsInstanceOf(f, ctor, msg) {
        var fullmsg;
        try {
            f();
        } catch (exc) {
            if (exc instanceof ctor) return;
            fullmsg = `Assertion failed: expected exception ${ctor.name}, got ${exc}`;
        }

        if (fullmsg === undefined)
            fullmsg = `Assertion failed: expected exception ${ctor.name}, no exception thrown`;
        if (msg !== undefined) fullmsg += " - " + msg;

        throw new Error(fullmsg);
    }
    global.assertThrowsInstanceOf = assertThrowsInstanceOf;

    /****************************
     * UTILITY FUNCTION EXPORTS *
     ****************************/

    var dump = global.dump;
    if (typeof global.dump === "function") {
        // A presumptively-functional |dump| exists, so no need to do anything.
    } else {
        // We don't have |dump|.  Try to simulate the desired effect another way.
        if (runningInBrowser) {
            // We can't actually print to the console: |global.print| invokes browser
            // printing functionality here (it's overwritten just below), and
            // |global.dump| isn't a function that'll dump to the console (presumably
            // because the preference to enable |dump| wasn't set).  Just make it a
            // no-op.
            dump = function () {};
        } else {
            // |print| prints to stdout: make |dump| do likewise.
            dump = global.print;
        }
        global.dump = dump;
    }

    var print;
    if (runningInBrowser) {
        // We're executing in a browser.  Using |global.print| would invoke browser
        // printing functionality: not what tests want!  Instead, use a print
        // function that syncs up with the test harness and console.
        print = function print() {
            var s = "TEST-INFO | ";
            for (var i = 0; i < arguments.length; i++)
                s += String(arguments[i]) + " ";

            // Dump the string to the console for developers and the harness.
            dump(s + "\n");

            // AddPrintOutput doesn't require HTML special characters be escaped.
            global.AddPrintOutput(s);
        };

        global.print = print;
    } else {
        // We're executing in a shell, and |global.print| is the desired function.
        print = global.print;
    }

    var gczeal = global.gczeal;
    if (typeof gczeal !== "function") {
        if (typeof SpecialPowersSetGCZeal === "function") {
            gczeal = function gczeal(z) {
                SpecialPowersSetGCZeal(z);
            };
        } else {
            gczeal = function () {}; // no-op if not available
        }

        global.gczeal = gczeal;
    }

    // Evaluates the given source code as global script code. browser.js provides
    // a different implementation for this function.
    var evaluateScript = global.evaluateScript;
    if (
        typeof evaluate === "function" &&
        typeof evaluateScript !== "function"
    ) {
        evaluateScript = function evaluateScript(code) {
            evaluate(String(code));
        };

        global.evaluateScript = evaluateScript;
    }

    function toPrinted(value) {
        value = String(value);

        var digits = "0123456789ABCDEF";
        var result = "";
        for (var i = 0; i < value.length; i++) {
            var ch = StringCharCodeAt(value, i);
            if (ch === 0x5c && i + 1 < value.length) {
                var d = value[i + 1];
                if (d === "n") {
                    result += "NL";
                    i++;
                } else if (d === "r") {
                    result += "CR";
                    i++;
                } else {
                    result += "\\";
                }
            } else if (ch === 0x0a) {
                result += "NL";
            } else if (ch < 0x20 || ch > 0x7e) {
                var a = digits[ch & 0xf];
                ch >>= 4;
                var b = digits[ch & 0xf];
                ch >>= 4;

                if (ch) {
                    var c = digits[ch & 0xf];
                    ch >>= 4;
                    var d = digits[ch & 0xf];

                    result += "\\u" + d + c + b + a;
                } else {
                    result += "\\x" + b + a;
                }
            } else {
                result += value[i];
            }
        }

        return result;
    }

    /*
     * An xorshift pseudo-random number generator see:
     * https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
     * This generator will always produce a value, n, where
     * 0 <= n <= 255
     */
    function* XorShiftGenerator(seed, size) {
        let x = seed;
        for (let i = 0; i < size; i++) {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            yield x % 256;
        }
    }
    global.XorShiftGenerator = XorShiftGenerator;

    /*************************************************************************
     * HARNESS-CENTRIC EXPORTS (we should generally work to eliminate these) *
     *************************************************************************/

    var PASSED = " PASSED! ";
    var FAILED = " FAILED! ";

    /*
     * Same as `new TestCase(description, expect, actual)`, except it doesn't
     * return the newly created test case object.
     */
    function AddTestCase(description, expect, actual) {
        new TestCase(description, expect, actual);
    }
    global.AddTestCase = AddTestCase;

    var testCasesArray = [];

    function TestCase(d, e, a, r) {
        this.description = d;
        this.expect = e;
        this.actual = a;
        this.passed = getTestCaseResult(e, a);
        this.reason = typeof r !== "undefined" ? String(r) : "";

        ArrayPush(testCasesArray, this);
    }
    global.TestCase = TestCase;

    TestCase.prototype = ObjectCreate(null);
    TestCase.prototype.testPassed = function TestCase_testPassed() {
        return this.passed;
    };
    TestCase.prototype.testFailed = function TestCase_testFailed() {
        return !this.passed;
    };
    TestCase.prototype.testDescription = function TestCase_testDescription() {
        return this.description + " " + this.reason;
    };

    function getTestCaseResult(expected, actual) {
        if (typeof expected !== typeof actual) return false;
        if (typeof expected !== "number")
            // Note that many tests depend on the use of '==' here, not '==='.
            return actual == expected;

        // Distinguish NaN from other values.  Using x !== x comparisons here
        // works even if tests redefine isNaN.
        if (actual !== actual) return expected !== expected;
        if (expected !== expected) return false;

        // Tolerate a certain degree of error.
        if (actual !== expected) return MathAbs(actual - expected) <= 1e-10;

        // Here would be a good place to distinguish 0 and -0, if we wanted
        // to.  However, doing so would introduce a number of failures in
        // areas where they don't seem important.  For example, the WeekDay
        // function in ECMA-262 returns -0 for Sundays before the epoch, but
        // the Date functions in SpiderMonkey specified in terms of WeekDay
        // often don't.  This seems unimportant.
        return true;
    }

    function reportTestCaseResult(description, expected, actual, output) {
        var testcase = new TestCase(description, expected, actual, output);

        // if running under reftest, let it handle result reporting.
        if (!runningInBrowser) {
            if (testcase.passed) {
                print(PASSED + description);
            } else {
                reportFailure(description + " : " + output);
            }
        }
    }

    function getTestCases() {
        return testCasesArray;
    }
    global.getTestCases = getTestCases;

    /*
     * The test driver searches for such a phrase in the test output.
     * If such phrase exists, it will set n as the expected exit code.
     */
    function expectExitCode(n) {
        print("--- NOTE: IN THIS TESTCASE, WE EXPECT EXIT CODE " + n + " ---");
    }
    global.expectExitCode = expectExitCode;

    /*
     * Statuses current section of a test
     */
    function inSection(x) {
        return "Section " + x + " of test - ";
    }
    global.inSection = inSection;

    /*
     * Report a failure in the 'accepted' manner
     */
    function reportFailure(msg) {
        msg = String(msg);
        var lines = StringSplit(msg, "\n");

        for (var i = 0; i < lines.length; i++) print(FAILED + " " + lines[i]);
    }
    global.reportFailure = reportFailure;

    /*
     * Print a non-failure message.
     */
    function printStatus(msg) {
        msg = String(msg);
        var lines = StringSplit(msg, "\n");

        for (var i = 0; i < lines.length; i++) print("STATUS: " + lines[i]);
    }
    global.printStatus = printStatus;

    /*
     * Print a bugnumber message.
     */
    function printBugNumber(num) {
        print("BUGNUMBER: " + num);
    }
    global.printBugNumber = printBugNumber;

    /*
     * Compare expected result to actual result, if they differ (in value and/or
     * type) report a failure.  If description is provided, include it in the
     * failure report.
     */
    function reportCompare(expected, actual, description) {
        var expected_t = typeof expected;
        var actual_t = typeof actual;
        var output = "";

        if (typeof description === "undefined") description = "";

        if (expected_t !== actual_t)
            output += `Type mismatch, expected type ${expected_t}, actual type ${actual_t} `;

        if (expected != actual)
            output += `Expected value '${toPrinted(
                expected
            )}', Actual value '${toPrinted(actual)}' `;

        reportTestCaseResult(description, expected, actual, output);
    }
    global.reportCompare = reportCompare;

    /*
     * Attempt to match a regular expression describing the result to
     * the actual result, if they differ (in value and/or
     * type) report a failure.  If description is provided, include it in the
     * failure report.
     */
    function reportMatch(expectedRegExp, actual, description) {
        var expected_t = "string";
        var actual_t = typeof actual;
        var output = "";

        if (typeof description === "undefined") description = "";

        if (expected_t !== actual_t)
            output += `Type mismatch, expected type ${expected_t}, actual type ${actual_t} `;

        var matches =
            ReflectApply(RegExpPrototypeExec, expectedRegExp, [actual]) !==
            null;
        if (!matches) {
            output += `Expected match to '${toPrinted(
                expectedRegExp
            )}', Actual value '${toPrinted(actual)}' `;
        }

        reportTestCaseResult(description, true, matches, output);
    }
    global.reportMatch = reportMatch;

    function compareSource(expect, actual, summary) {
        // compare source
        var expectP = String(expect);
        var actualP = String(actual);

        print("expect:\n" + expectP);
        print("actual:\n" + actualP);

        reportCompare(expectP, actualP, summary);

        // actual must be compilable if expect is?
        try {
            var expectCompile = "No Error";
            var actualCompile;

            Function(expect);
            try {
                Function(actual);
                actualCompile = "No Error";
            } catch (ex1) {
                actualCompile = ex1 + "";
            }
            reportCompare(
                expectCompile,
                actualCompile,
                summary + ": compile actual"
            );
        } catch (ex) {}
    }
    global.compareSource = compareSource;

    function test() {
        var testCases = getTestCases();
        for (var i = 0; i < testCases.length; i++) {
            var testCase = testCases[i];
            testCase.reason += testCase.passed ? "" : "wrong value ";

            // if running under reftest, let it handle result reporting.
            if (!runningInBrowser) {
                var message = `${testCase.description} = ${testCase.actual} expected: ${testCase.expect}`;
                print((testCase.passed ? PASSED : FAILED) + message);
            }
        }
    }
    global.test = test;

    // This function uses the shell's print function. When running tests in the
    // browser, browser.js overrides this function to write to the page.
    function writeHeaderToLog(string) {
        print(string);
    }
    global.writeHeaderToLog = writeHeaderToLog;

    /************************************
     * PROMISE TESTING FUNCTION EXPORTS *
     ************************************/

    function getPromiseResult(promise) {
        var result,
            error,
            caught = false;
        promise.then(
            (r) => {
                result = r;
            },
            (e) => {
                caught = true;
                error = e;
            }
        );
        if (caught) throw error;
        return result;
    }
    global.getPromiseResult = getPromiseResult;

    function assertEventuallyEq(promise, expected) {
        assertEq(getPromiseResult(promise), expected);
    }
    global.assertEventuallyEq = assertEventuallyEq;

    function assertEventuallyThrows(promise, expectedErrorType) {
        assertThrowsInstanceOf(
            () => getPromiseResult(promise),
            expectedErrorType
        );
    }
    global.assertEventuallyThrows = assertEventuallyThrows;

    function assertEventuallyDeepEq(promise, expected) {
        assertDeepEq(getPromiseResult(promise), expected);
    }
    global.assertEventuallyDeepEq = assertEventuallyDeepEq;

    /*******************************************
     * RUN ONCE CODE TO SETUP ADDITIONAL STATE *
     *******************************************/

    /*
     * completesNormally(CODE) returns true if evaluating CODE (as eval
     * code) completes normally (rather than throwing an exception).
     */
    global.completesNormally = function completesNormally(code) {
        try {
            eval(code);
            return true;
        } catch (exception) {
            return false;
        }
    };

    /*
     * raisesException(EXCEPTION)(CODE) returns true if evaluating CODE (as
     * eval code) throws an exception object that is an instance of EXCEPTION,
     * and returns false if it throws any other error or evaluates
     * successfully. For example: raises(TypeError)("0()") == true.
     */
    global.raisesException = function raisesException(exception) {
        return function (code) {
            try {
                eval(code);
                return false;
            } catch (actual) {
                return actual instanceof exception;
            }
        };
    };

    /*
     * Return true if A is equal to B, where equality on arrays and objects
     * means that they have the same set of enumerable properties, the values
     * of each property are deep_equal, and their 'length' properties are
     * equal. Equality on other types is ==.
     */
    global.deepEqual = function deepEqual(a, b) {
        if (typeof a != typeof b) return false;

        if (typeof a == "object") {
            var props = {};
            // For every property of a, does b have that property with an equal value?
            for (var prop in a) {
                if (!deepEqual(a[prop], b[prop])) return false;
                props[prop] = true;
            }
            // Are all of b's properties present on a?
            for (var prop in b) if (!props[prop]) return false;
            // length isn't enumerable, but we want to check it, too.
            return a.length == b.length;
        }

        if (a === b) {
            // Distinguish 0 from -0, even though they are ===.
            return a !== 0 || 1 / a === 1 / b;
        }

        // Treat NaNs as equal, even though NaN !== NaN.
        // NaNs are the only non-reflexive values, i.e., if a !== a, then a is a NaN.
        // isNaN is broken: it converts its argument to number, so isNaN("foo") => true
        return a !== a && b !== b;
    };

    /** Make an iterator with a return method. */
    global.makeIterator = function makeIterator(overrides) {
        var throwMethod;
        if (overrides && overrides.throw) throwMethod = overrides.throw;
        var iterator = {
            throw: throwMethod,
            next: function (x) {
                if (overrides && overrides.next) return overrides.next(x);
                return { done: false };
            },
            return: function (x) {
                if (overrides && overrides.ret) return overrides.ret(x);
                return { done: true };
            },
        };

        return function () {
            return iterator;
        };
    };

    /** Yield every permutation of the elements in some array. */
    global.Permutations = function* Permutations(items) {
        if (items.length == 0) {
            yield [];
        } else {
            items = items.slice(0);
            for (let i = 0; i < items.length; i++) {
                let swap = items[0];
                items[0] = items[i];
                items[i] = swap;
                for (let e of Permutations(items.slice(1, items.length)))
                    yield [items[0]].concat(e);
            }
        }
    };

    if (typeof global.assertThrowsValue === "undefined") {
        global.assertThrowsValue = function assertThrowsValue(f, val, msg) {
            var fullmsg;
            try {
                f();
            } catch (exc) {
                if (
                    (exc === val) === (val === val) &&
                    (val !== 0 || 1 / exc === 1 / val)
                )
                    return;
                fullmsg =
                    "Assertion failed: expected exception " +
                    val +
                    ", got " +
                    exc;
            }
            if (fullmsg === undefined)
                fullmsg =
                    "Assertion failed: expected exception " +
                    val +
                    ", no exception thrown";
            if (msg !== undefined) fullmsg += " - " + msg;
            throw new Error(fullmsg);
        };
    }

    if (typeof global.assertThrowsInstanceOf === "undefined") {
        global.assertThrowsInstanceOf = function assertThrowsInstanceOf(
            f,
            ctor,
            msg
        ) {
            var fullmsg;
            try {
                f();
            } catch (exc) {
                if (exc instanceof ctor) return;
                fullmsg = `Assertion failed: expected exception ${ctor.name}, got ${exc}`;
            }

            if (fullmsg === undefined)
                fullmsg = `Assertion failed: expected exception ${ctor.name}, no exception thrown`;
            if (msg !== undefined) fullmsg += " - " + msg;

            throw new Error(fullmsg);
        };
    }

    global.assertDeepEq = (function () {
        var call = Function.prototype.call,
            Array_isArray = ArrayIsArray,
            Map_ = Map,
            Error_ = Error,
            Symbol_ = Symbol,
            Map_has = call.bind(Map.prototype.has),
            Map_get = call.bind(Map.prototype.get),
            Map_set = call.bind(Map.prototype.set),
            Object_toString = call.bind(Object.prototype.toString),
            Function_toString = call.bind(Function.prototype.toString),
            Object_getPrototypeOf = Object.getPrototypeOf,
            Object_hasOwnProperty = call.bind(Object.prototype.hasOwnProperty),
            Object_getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor,
            Object_isExtensible = Object.isExtensible,
            Object_getOwnPropertyNames = Object.getOwnPropertyNames,
            uneval_ = global.uneval;

        // Return true iff ES6 Type(v) isn't Object.
        // Note that `typeof document.all === "undefined"`.
        function isPrimitive(v) {
            return (
                v === null ||
                v === undefined ||
                typeof v === "boolean" ||
                typeof v === "number" ||
                typeof v === "string" ||
                typeof v === "symbol"
            );
        }

        function assertSameValue(a, b, msg) {
            try {
                assertEq(a, b);
            } catch (exc) {
                throw Error_(exc.message + (msg ? " " + msg : ""));
            }
        }

        function assertSameClass(a, b, msg) {
            var ac = Object_toString(a),
                bc = Object_toString(b);
            assertSameValue(ac, bc, msg);
            switch (ac) {
                case "[object Function]":
                    if (
                        typeof isProxy !== "undefined" &&
                        !isProxy(a) &&
                        !isProxy(b)
                    )
                        assertSameValue(
                            Function_toString(a),
                            Function_toString(b),
                            msg
                        );
            }
        }

        function at(prevmsg, segment) {
            return prevmsg ? prevmsg + segment : "at _" + segment;
        }

        // Assert that the arguments a and b are thoroughly structurally equivalent.
        //
        // For the sake of speed, we cut a corner:
        //    var x = {}, y = {}, ax = [x];
        //    assertDeepEq([ax, x], [ax, y]);  // passes (?!)
        //
        // Technically this should fail, since the two object graphs are different.
        // (The graph of [ax, y] contains one more object than the graph of [ax, x].)
        //
        // To get technically correct behavior, pass {strictEquivalence: true}.
        // This is slower because we have to walk the entire graph, and Object.prototype
        // is big.
        //
        return function assertDeepEq(a, b, options) {
            var strictEquivalence = options ? options.strictEquivalence : false;

            function assertSameProto(a, b, msg) {
                check(
                    Object_getPrototypeOf(a),
                    Object_getPrototypeOf(b),
                    at(msg, ".__proto__")
                );
            }

            function failPropList(na, nb, msg) {
                throw Error_(
                    "got own properties " +
                        uneval_(na) +
                        ", expected " +
                        uneval_(nb) +
                        (msg ? " " + msg : "")
                );
            }

            function assertSameProps(a, b, msg) {
                var na = Object_getOwnPropertyNames(a),
                    nb = Object_getOwnPropertyNames(b);
                if (na.length !== nb.length) failPropList(na, nb, msg);

                // Ignore differences in whether Array elements are stored densely.
                if (Array_isArray(a)) {
                    na.sort();
                    nb.sort();
                }

                for (var i = 0; i < na.length; i++) {
                    var name = na[i];
                    if (name !== nb[i]) failPropList(na, nb, msg);
                    var da = Object_getOwnPropertyDescriptor(a, name),
                        db = Object_getOwnPropertyDescriptor(b, name);
                    var pmsg = at(
                        msg,
                        /^[_$A-Za-z0-9]+$/.test(name)
                            ? /0|[1-9][0-9]*/.test(name)
                                ? "[" + name + "]"
                                : "." + name
                            : "[" + uneval_(name) + "]"
                    );
                    assertSameValue(
                        da.configurable,
                        db.configurable,
                        at(pmsg, ".[[Configurable]]")
                    );
                    assertSameValue(
                        da.enumerable,
                        db.enumerable,
                        at(pmsg, ".[[Enumerable]]")
                    );
                    if (Object_hasOwnProperty(da, "value")) {
                        if (!Object_hasOwnProperty(db, "value"))
                            throw Error_(
                                "got data property, expected accessor property" +
                                    pmsg
                            );
                        check(da.value, db.value, pmsg);
                    } else {
                        if (Object_hasOwnProperty(db, "value"))
                            throw Error_(
                                "got accessor property, expected data property" +
                                    pmsg
                            );
                        check(da.get, db.get, at(pmsg, ".[[Get]]"));
                        check(da.set, db.set, at(pmsg, ".[[Set]]"));
                    }
                }
            }

            var ab = new Map_();
            var bpath = new Map_();

            function check(a, b, path) {
                if (typeof a === "symbol") {
                    // Symbols are primitives, but they have identity.
                    // Symbol("x") !== Symbol("x") but
                    // assertDeepEq(Symbol("x"), Symbol("x")) should pass.
                    if (typeof b !== "symbol") {
                        throw Error_(
                            "got " +
                                uneval_(a) +
                                ", expected " +
                                uneval_(b) +
                                " " +
                                path
                        );
                    } else if (uneval_(a) !== uneval_(b)) {
                        // We lamely use uneval_ to distinguish well-known symbols
                        // from user-created symbols. The standard doesn't offer
                        // a convenient way to do it.
                        throw Error_(
                            "got " +
                                uneval_(a) +
                                ", expected " +
                                uneval_(b) +
                                " " +
                                path
                        );
                    } else if (Map_has(ab, a)) {
                        assertSameValue(Map_get(ab, a), b, path);
                    } else if (Map_has(bpath, b)) {
                        var bPrevPath = Map_get(bpath, b) || "_";
                        throw Error_(
                            "got distinct symbols " +
                                at(path, "") +
                                " and " +
                                at(bPrevPath, "") +
                                ", expected the same symbol both places"
                        );
                    } else {
                        Map_set(ab, a, b);
                        Map_set(bpath, b, path);
                    }
                } else if (isPrimitive(a)) {
                    assertSameValue(a, b, path);
                } else if (isPrimitive(b)) {
                    throw Error_(
                        "got " +
                            Object_toString(a) +
                            ", expected " +
                            uneval_(b) +
                            " " +
                            path
                    );
                } else if (Map_has(ab, a)) {
                    assertSameValue(Map_get(ab, a), b, path);
                } else if (Map_has(bpath, b)) {
                    var bPrevPath = Map_get(bpath, b) || "_";
                    throw Error_(
                        "got distinct objects " +
                            at(path, "") +
                            " and " +
                            at(bPrevPath, "") +
                            ", expected the same object both places"
                    );
                } else {
                    Map_set(ab, a, b);
                    Map_set(bpath, b, path);
                    if (a !== b || strictEquivalence) {
                        assertSameClass(a, b, path);
                        assertSameProto(a, b, path);
                        assertSameProps(a, b, path);
                        assertSameValue(
                            Object_isExtensible(a),
                            Object_isExtensible(b),
                            at(path, ".[[Extensible]]")
                        );
                    }
                }
            }

            check(a, b, "");
        };
    })();

    const msPerDay = 1000 * 60 * 60 * 24;
    const msPerHour = 1000 * 60 * 60;
    global.msPerHour = msPerHour;

    // Offset of tester's time zone from UTC.
    const TZ_DIFF = GetRawTimezoneOffset();
    global.TZ_ADJUST = TZ_DIFF * msPerHour;

    const UTC_01_JAN_1900 = -2208988800000;
    const UTC_01_JAN_2000 = 946684800000;
    const UTC_29_FEB_2000 = UTC_01_JAN_2000 + 31 * msPerDay + 28 * msPerDay;
    const UTC_01_JAN_2005 =
        UTC_01_JAN_2000 +
        TimeInYear(2000) +
        TimeInYear(2001) +
        TimeInYear(2002) +
        TimeInYear(2003) +
        TimeInYear(2004);
    global.UTC_01_JAN_1900 = UTC_01_JAN_1900;
    global.UTC_01_JAN_2000 = UTC_01_JAN_2000;
    global.UTC_29_FEB_2000 = UTC_29_FEB_2000;
    global.UTC_01_JAN_2005 = UTC_01_JAN_2005;

    /*
     * Originally, the test suite used a hard-coded value TZ_DIFF = -8.
     * But that was only valid for testers in the Pacific Standard Time Zone!
     * We calculate the proper number dynamically for any tester. We just
     * have to be careful not to use a date subject to Daylight Savings Time...
     */
    function GetRawTimezoneOffset() {
        let t1 = new Date(2000, 1, 1).getTimezoneOffset();
        let t2 = new Date(2000, 1 + 6, 1).getTimezoneOffset();

        // 1) Time zone without daylight saving time.
        // 2) Northern hemisphere with daylight saving time.
        if (t1 - t2 >= 0) return -t1 / 60;

        // 3) Southern hemisphere with daylight saving time.
        return -t2 / 60;
    }

    function DaysInYear(y) {
        return y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0) ? 366 : 365;
    }

    function TimeInYear(y) {
        return DaysInYear(y) * msPerDay;
    }

    function getDefaultTimeZone() {
        return "EST5EDT";
    }

    function getDefaultLocale() {
        // If the default locale looks like a BCP-47 language tag, return it.
        var locale = global.getDefaultLocale();
        if (locale.match(/^[a-z][a-z0-9\-]+$/i)) return locale;

        // Otherwise use undefined to reset to the default locale.
        return undefined;
    }

    let defaultTimeZone = null;
    let defaultLocale = null;

    // Run the given test in the requested time zone.
    function inTimeZone(tzname, fn) {
        if (defaultTimeZone === null) defaultTimeZone = getDefaultTimeZone();

        try {
            fn();
        } finally {
        }
    }
    global.inTimeZone = inTimeZone;

    // Run the given test with the requested locale.
    function withLocale(locale, fn) {
        if (defaultLocale === null) defaultLocale = getDefaultLocale();

        setDefaultLocale(locale);
        try {
            fn();
        } finally {
            setDefaultLocale(defaultLocale);
        }
    }
    global.withLocale = withLocale;

    const Month = {
        January: 0,
        February: 1,
        March: 2,
        April: 3,
        May: 4,
        June: 5,
        July: 6,
        August: 7,
        September: 8,
        October: 9,
        November: 10,
        December: 11,
    };
    global.Month = Month;

    const weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].join(
        "|"
    );
    const months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ].join("|");
    const datePart = String.raw`(?:${weekdays}) (?:${months}) \d{2}`;
    const timePart = String.raw`\d{4,6} \d{2}:\d{2}:\d{2} GMT[+-]\d{4}`;
    const dateTimeRE = new RegExp(
        String.raw`^(${datePart} ${timePart})(?: \((.+)\))?$`
    );

    function assertDateTime(date, expected, ...alternativeTimeZones) {
        let actual = date.toString();
        assertEq(dateTimeRE.test(expected), true, `${expected}`);
        assertEq(dateTimeRE.test(actual), true, `${actual}`);

        let [, expectedDateTime, expectedTimeZone] = dateTimeRE.exec(expected);
        let [, actualDateTime, actualTimeZone] = dateTimeRE.exec(actual);

        assertEq(actualDateTime, expectedDateTime);

        // The time zone identifier is optional, so only compare its value if
        // it's present in |actual| and |expected|.
        if (expectedTimeZone !== undefined && actualTimeZone !== undefined) {
            // Test against the alternative time zone identifiers if necessary.
            if (actualTimeZone !== expectedTimeZone) {
                for (let alternativeTimeZone of alternativeTimeZones) {
                    if (actualTimeZone === alternativeTimeZone) {
                        expectedTimeZone = alternativeTimeZone;
                        break;
                    }
                }
            }
            assertEq(actualTimeZone, expectedTimeZone);
        }
    }
    global.assertDateTime = assertDateTime;

    global.testRegExp = function testRegExp(
        statuses,
        patterns,
        strings,
        actualmatches,
        expectedmatches
    ) {
        var status = "";
        var pattern = new RegExp();
        var string = "";
        var actualmatch = new Array();
        var expectedmatch = new Array();
        var state = "";
        var lActual = -1;
        var lExpect = -1;
        var actual = new Array();

        for (var i = 0; i != patterns.length; i++) {
            status = statuses[i];
            pattern = patterns[i];
            string = strings[i];
            actualmatch = actualmatches[i];
            expectedmatch = expectedmatches[i];

            if (actualmatch) {
                actual = formatArray(actualmatch);
                if (expectedmatch) {
                    // expectedmatch and actualmatch are arrays -
                    lExpect = expectedmatch.length;
                    lActual = actualmatch.length;

                    var expected = formatArray(expectedmatch);

                    if (lActual != lExpect) {
                        reportCompare(
                            lExpect,
                            lActual,
                            state +
                                ERR_LENGTH +
                                MSG_EXPECT +
                                expected +
                                MSG_ACTUAL +
                                actual +
                                CHAR_NL
                        );
                        continue;
                    }

                    // OK, the arrays have same length -
                    if (expected != actual) {
                        reportCompare(
                            expected,
                            actual,
                            state +
                                ERR_MATCH +
                                MSG_EXPECT +
                                expected +
                                MSG_ACTUAL +
                                actual +
                                CHAR_NL
                        );
                    } else {
                        reportCompare(expected, actual, state);
                    }
                } //expectedmatch is null - that is, we did not expect a match -
                else {
                    expected = expectedmatch;
                    reportCompare(
                        expected,
                        actual,
                        state +
                            ERR_UNEXP_MATCH +
                            MSG_EXPECT +
                            expectedmatch +
                            MSG_ACTUAL +
                            actual +
                            CHAR_NL
                    );
                }
            } // actualmatch is null
            else {
                if (expectedmatch) {
                    actual = actualmatch;
                    reportCompare(
                        expected,
                        actual,
                        state +
                            ERR_NO_MATCH +
                            MSG_EXPECT +
                            expectedmatch +
                            MSG_ACTUAL +
                            actualmatch +
                            CHAR_NL
                    );
                } // we did not expect a match
                else {
                    // Being ultra-cautious. Presumably expectedmatch===actualmatch===null
                    expected = expectedmatch;
                    actual = actualmatch;
                    reportCompare(expectedmatch, actualmatch, state);
                }
            }
        }
    };

    function clone_object_check(b, desc) {
        function classOf(obj) {
            return Object.prototype.toString.call(obj);
        }

        function ownProperties(obj) {
            return Object.getOwnPropertyNames(obj).map(function (p) {
                return [p, Object.getOwnPropertyDescriptor(obj, p)];
            });
        }

        function isArrayLength(obj, pair) {
            return Array.isArray(obj) && pair[0] == "length";
        }

        function isCloneable(obj, pair) {
            return (
                isArrayLength(obj, pair) ||
                (typeof pair[0] === "string" && pair[1].enumerable)
            );
        }

        function notIndex(p) {
            var u = p >>> 0;
            return !("" + u == p && u != 0xffffffff);
        }

        function assertIsCloneOf(a, b, path) {
            assertEq(a === b, false);

            var ca = classOf(a);
            assertEq(ca, classOf(b), path);

            assertEq(
                Object.getPrototypeOf(a),
                ca == "[object Object]" ? Object.prototype : Array.prototype,
                path
            );

            // 'b', the original object, may have non-enumerable or XMLName
            // properties; ignore them (except .length, if it's an Array).
            // 'a', the clone, should not have any non-enumerable properties
            // (except .length, if it's an Array) or XMLName properties.
            var pb = ownProperties(b).filter(function (element) {
                return isCloneable(b, element);
            });
            var pa = ownProperties(a);
            for (var i = 0; i < pa.length; i++) {
                assertEq(
                    typeof pa[i][0],
                    "string",
                    "clone should not have E4X properties " + path
                );
                if (!isCloneable(a, pa[i])) {
                    throw new Error(
                        "non-cloneable clone property " +
                            uneval(pa[i][0]) +
                            " " +
                            path
                    );
                }
            }

            // Check that, apart from properties whose names are array indexes,
            // the enumerable properties appear in the same order.
            var aNames = pa
                .map(function (pair) {
                    return pair[1];
                })
                .filter(notIndex);
            var bNames = pa
                .map(function (pair) {
                    return pair[1];
                })
                .filter(notIndex);
            assertEq(aNames.join(","), bNames.join(","), path);

            // Check that the lists are the same when including array indexes.
            function byName(a, b) {
                a = a[0];
                b = b[0];
                return a < b ? -1 : a === b ? 0 : 1;
            }
            pa.sort(byName);
            pb.sort(byName);
            assertEq(
                pa.length,
                pb.length,
                "should see the same number of properties " + path
            );
            for (var i = 0; i < pa.length; i++) {
                var aName = pa[i][0];
                var bName = pb[i][0];
                assertEq(aName, bName, path);

                var path2 = path + "." + aName;
                var da = pa[i][1];
                var db = pb[i][1];
                if (!isArrayLength(a, pa[i])) {
                    assertEq(da.configurable, true, path2);
                }
                assertEq(da.writable, true, path2);
                assertEq("value" in da, true, path2);
                var va = da.value;
                var vb = b[pb[i][0]];
                if (typeof va === "object" && va !== null)
                    queue.push([va, vb, path2]);
                else assertEq(va, vb, path2);
            }
        }

        var banner = "while testing clone of " + (desc || uneval(b));
        var a = deserialize(serialize(b));
        var queue = [[a, b, banner]];
        while (queue.length) {
            var triple = queue.shift();
            assertIsCloneOf(triple[0], triple[1], triple[2]);
        }

        return a; // for further testing
    }
    global.clone_object_check = clone_object_check;

    global.testLenientAndStrict = function testLenientAndStrict(
        code,
        lenient_pred,
        strict_pred
    ) {
        return strict_pred("'use strict'; " + code) && lenient_pred(code);
    };

    /*
     * parsesSuccessfully(CODE) returns true if CODE parses as function
     * code without an error.
     */
    global.parsesSuccessfully = function parsesSuccessfully(code) {
        try {
            Function(code);
            return true;
        } catch (exception) {
            return false;
        }
    };

    /*
     * parseRaisesException(EXCEPTION)(CODE) returns true if parsing CODE
     * as function code raises EXCEPTION.
     */
    global.parseRaisesException = function parseRaisesException(exception) {
        return function (code) {
            try {
                Function(code);
                return false;
            } catch (actual) {
                return exception.prototype.isPrototypeOf(actual);
            }
        };
    };

    /*
     * returns(VALUE)(CODE) returns true if evaluating CODE (as eval code)
     * completes normally (rather than throwing an exception), yielding a value
     * strictly equal to VALUE.
     */
    global.returns = function returns(value) {
        return function (code) {
            try {
                return eval(code) === value;
            } catch (exception) {
                return false;
            }
        };
    };

    const { apply: Reflect_apply, construct: Reflect_construct } = Reflect;
    const { get: WeakMap_prototype_get, has: WeakMap_prototype_has } =
        WeakMap.prototype;

    const sharedConstructors = new WeakMap();

    // Synthesize a constructor for a shared memory array from the constructor
    // for unshared memory. This has "good enough" fidelity for many uses. In
    // cases where it's not good enough, call isSharedConstructor for local
    // workarounds.
    function sharedConstructor(baseConstructor) {
        // Create SharedTypedArray as a subclass of %TypedArray%, following the
        // built-in %TypedArray% subclasses.
        class SharedTypedArray extends Object.getPrototypeOf(baseConstructor) {
            constructor(...args) {
                var array = Reflect_construct(baseConstructor, args);
                var { buffer, byteOffset, length } = array;
                var sharedBuffer = new SharedArrayBuffer(buffer.byteLength);
                var sharedArray = Reflect_construct(
                    baseConstructor,
                    [sharedBuffer, byteOffset, length],
                    new.target
                );
                for (var i = 0; i < length; i++) sharedArray[i] = array[i];
                assertEq(sharedArray.buffer, sharedBuffer);
                return sharedArray;
            }
        }

        // 22.2.5.1 TypedArray.BYTES_PER_ELEMENT
        Object.defineProperty(SharedTypedArray, "BYTES_PER_ELEMENT", {
            __proto__: null,
            value: baseConstructor.BYTES_PER_ELEMENT,
        });

        // 22.2.6.1 TypedArray.prototype.BYTES_PER_ELEMENT
        Object.defineProperty(SharedTypedArray.prototype, "BYTES_PER_ELEMENT", {
            __proto__: null,
            value: baseConstructor.BYTES_PER_ELEMENT,
        });

        // Share the same name with the base constructor to avoid calling
        // isSharedConstructor() in multiple places.
        Object.defineProperty(SharedTypedArray, "name", {
            __proto__: null,
            value: baseConstructor.name,
        });

        sharedConstructors.set(SharedTypedArray, baseConstructor);

        return SharedTypedArray;
    }

    /**
     * Returns `true` if `constructor` is a TypedArray constructor for shared
     * memory.
     */
    function isSharedConstructor(constructor) {
        return Reflect_apply(WeakMap_prototype_has, sharedConstructors, [
            constructor,
        ]);
    }

    /**
     * All TypedArray constructors for unshared memory.
     */
    const typedArrayConstructors = Object.freeze([
        Int8Array,
        Uint8Array,
        Uint8ClampedArray,
        Int16Array,
        Uint16Array,
        Int32Array,
        Uint32Array,
        Float32Array,
        Float64Array,
    ]);
    /**
     * All TypedArray constructors for shared memory.
     */
    const sharedTypedArrayConstructors = Object.freeze(
        typeof SharedArrayBuffer === "function"
            ? typedArrayConstructors.map(sharedConstructor)
            : []
    );

    /**
     * All TypedArray constructors for unshared and shared memory.
     */
    const anyTypedArrayConstructors = Object.freeze([
        ...typedArrayConstructors,
        ...sharedTypedArrayConstructors,
    ]);
    global.typedArrayConstructors = typedArrayConstructors;
    global.sharedTypedArrayConstructors = sharedTypedArrayConstructors;
    global.anyTypedArrayConstructors = anyTypedArrayConstructors;
    /**
     * Returns `true` if `constructor` is a TypedArray constructor for shared
     * or unshared memory, with an underlying element type of either Float32 or
     * Float64.
     */
    function isFloatConstructor(constructor) {
        if (isSharedConstructor(constructor))
            constructor = Reflect_apply(
                WeakMap_prototype_get,
                sharedConstructors,
                [constructor]
            );
        return constructor == Float32Array || constructor == Float64Array;
    }

    global.isSharedConstructor = isSharedConstructor;
    global.isFloatConstructor = isFloatConstructor;
})(this);

var DESCRIPTION;

function arraysEqual(a1, a2) {
    return (
        a1.length === a2.length &&
        a1.every(function (v, i) {
            return v === a2[i];
        })
    );
}

function SameValue(v1, v2) {
    if (v1 === 0 && v2 === 0) return 1 / v1 === 1 / v2;
    if (v1 !== v1 && v2 !== v2) return true;
    return v1 === v2;
}

function arraysEqual(a1, a2) {
    var len1 = a1.length,
        len2 = a2.length;
    if (len1 !== len2) return false;
    for (var i = 0; i < len1; i++) {
        if (!SameValue(a1[i], a2[i])) return false;
    }
    return true;
}

var evalInFrame = function (f) {
    return eval(f);
};

function globalPrototypeChainIsMutable() {
    return false;
}

if (typeof assertIteratorResult === "undefined") {
    var assertIteratorResult = function assertIteratorResult(
        result,
        value,
        done
    ) {
        assertEq(typeof result, "object");
        var expectedProps = ["done", "value"];
        var actualProps = Object.getOwnPropertyNames(result);
        actualProps.sort(), expectedProps.sort();
        assertDeepEq(actualProps, expectedProps);
        assertDeepEq(result.value, value);
        assertDeepEq(result.done, done);
    };
}

if (typeof assertIteratorNext === "undefined") {
    var assertIteratorNext = function assertIteratorNext(iter, value) {
        assertIteratorResult(iter.next(), value, false);
    };
}

if (typeof assertIteratorDone === "undefined") {
    var assertIteratorDone = function assertIteratorDone(iter, value) {
        assertIteratorResult(iter.next(), value, true);
    };
}

var appendToActual = function (s) {
    actual += s + ",";
};

if (!("gczeal" in this)) {
    gczeal = function () {};
}

if (!("schedulegc" in this)) {
    schedulegc = function () {};
}

if (!("gcslice" in this)) {
    gcslice = function () {};
}

if (!("selectforgc" in this)) {
    selectforgc = function () {};
}

if (!("verifyprebarriers" in this)) {
    verifyprebarriers = function () {};
}

if (!("verifypostbarriers" in this)) {
    verifypostbarriers = function () {};
}

if (!("gcPreserveCode" in this)) {
    gcPreserveCode = function () {};
}

if (typeof isHighSurrogate === "undefined") {
    var isHighSurrogate = function isHighSurrogate(s) {
        var c = s.charCodeAt(0);
        return c >= 0xd800 && c <= 0xdbff;
    };
}

if (typeof isLowSurrogate === "undefined") {
    var isLowSurrogate = function isLowSurrogate(s) {
        var c = s.charCodeAt(0);
        return c >= 0xdc00 && c <= 0xdfff;
    };
}

if (typeof isSurrogatePair === "undefined") {
    var isSurrogatePair = function isSurrogatePair(s) {
        return s.length == 2 && isHighSurrogate(s[0]) && isLowSurrogate(s[1]);
    };
}
var newGlobal = function () {
    newGlobal.eval = eval;
    return this;
};

function assertThrowsValue(f) {
    f();
}
function evalcx(f) {
    eval(f);
}
function gcparam() {}
function uneval(f) {
    return f.toString();
}
function oomTest(f) {
    f();
}
function evaluate(f) {
    return eval(f);
}
function inIon() {
    return true;
}
function byteSizeOfScript(f) {
    return f.toString().length;
}

var Match = (function () {
    function Pattern(template) {
        // act like a constructor even as a function
        if (!(this instanceof Pattern)) return new Pattern(template);

        this.template = template;
    }

    Pattern.prototype = {
        match: function (act) {
            return match(act, this.template);
        },

        matches: function (act) {
            try {
                return this.match(act);
            } catch (e) {
                if (!(e instanceof MatchError)) throw e;
                return false;
            }
        },

        assert: function (act, message) {
            try {
                return this.match(act);
            } catch (e) {
                if (!(e instanceof MatchError)) throw e;
                throw new Error((message || "failed match") + ": " + e.message);
            }
        },

        toString: () => "[object Pattern]",
    };

    Pattern.ANY = new Pattern();
    Pattern.ANY.template = Pattern.ANY;

    Pattern.NUMBER = new Pattern();
    Pattern.NUMBER.match = function (act) {
        if (typeof act !== "number") {
            throw new MatchError("Expected number, got: " + quote(act));
        }
    };

    Pattern.NATURAL = new Pattern();
    Pattern.NATURAL.match = function (act) {
        if (typeof act !== "number" || act !== Math.floor(act) || act < 0) {
            throw new MatchError("Expected natural number, got: " + quote(act));
        }
    };

    var quote = uneval;

    function MatchError(msg) {
        this.message = msg;
    }

    MatchError.prototype = {
        toString: function () {
            return "match error: " + this.message;
        },
    };

    function isAtom(x) {
        return (
            typeof x === "number" ||
            typeof x === "string" ||
            typeof x === "boolean" ||
            x === null ||
            x === undefined ||
            (typeof x === "object" && x instanceof RegExp) ||
            typeof x === "bigint"
        );
    }

    function isObject(x) {
        return x !== null && typeof x === "object";
    }

    function isFunction(x) {
        return typeof x === "function";
    }

    function isArrayLike(x) {
        return isObject(x) && "length" in x;
    }

    function matchAtom(act, exp) {
        if (typeof exp === "number" && isNaN(exp)) {
            if (typeof act !== "number" || !isNaN(act))
                throw new MatchError("expected NaN, got: " + quote(act));
            return true;
        }

        if (exp === null) {
            if (act !== null)
                throw new MatchError("expected null, got: " + quote(act));
            return true;
        }

        if (exp instanceof RegExp) {
            if (!(act instanceof RegExp) || exp.source !== act.source)
                throw new MatchError(
                    "expected " + quote(exp) + ", got: " + quote(act)
                );
            return true;
        }

        switch (typeof exp) {
            case "string":
            case "undefined":
                if (act !== exp)
                    throw new MatchError(
                        "expected " + quote(exp) + ", got " + quote(act)
                    );
                return true;
            case "boolean":
            case "number":
            case "bigint":
                if (exp !== act)
                    throw new MatchError(
                        "expected " + exp + ", got " + quote(act)
                    );
                return true;
        }

        throw new Error("bad pattern: " + exp.toSource());
    }

    function matchObject(act, exp) {
        if (!isObject(act))
            throw new MatchError("expected object, got " + quote(act));

        for (var key in exp) {
            if (!(key in act))
                throw new MatchError(
                    "expected property " +
                        quote(key) +
                        " not found in " +
                        quote(act)
                );
            match(act[key], exp[key]);
        }

        return true;
    }

    function matchFunction(act, exp) {
        if (!isFunction(act))
            throw new MatchError("expected function, got " + quote(act));

        if (act !== exp)
            throw new MatchError(
                "expected function: " +
                    exp +
                    "\nbut got different function: " +
                    act
            );
    }

    function matchArray(act, exp) {
        if (!isObject(act) || !("length" in act))
            throw new MatchError(
                "expected array-like object, got " + quote(act)
            );

        var length = exp.length;
        if (act.length !== exp.length)
            throw new MatchError(
                "expected array-like object of length " +
                    length +
                    ", got " +
                    quote(act)
            );

        for (var i = 0; i < length; i++) {
            if (i in exp) {
                if (!(i in act))
                    throw new MatchError(
                        "expected array property " +
                            i +
                            " not found in " +
                            quote(act)
                    );
                match(act[i], exp[i]);
            }
        }

        return true;
    }

    function match(act, exp) {
        if (exp === Pattern.ANY) return true;

        if (exp instanceof Pattern) return exp.match(act);

        if (isAtom(exp)) return matchAtom(act, exp);

        if (isArrayLike(exp)) return matchArray(act, exp);

        if (isFunction(exp)) return matchFunction(act, exp);

        if (isObject(exp)) return matchObject(act, exp);

        throw new Error("bad pattern: " + exp.toSource());
    }

    return { Pattern: Pattern, MatchError: MatchError };
})();

function serialize(f) {
    return f.toString();
}
function isLatin1() {
    return true;
}
function deserialize(f) {
    return f;
}

function assertErrorMessage(f) {
    f();
}
function cacheEntry(f) {
    return eval(f);
}

function resolvePromise(p, s) {
    return p.resolve(s);
}

function isConstructor(o) {
    try {
        new new Proxy(o, { construct: () => ({}) })();
        return true;
    } catch (e) {
        return false;
    }
}

var InternalError = new Error();
function wrapWithProto(p, v) {
    p.proto = v;
    return p;
}

function objectGlobal(v) {
    return v;
}
function saveStack() {
    return "";
}
function callFunctionWithAsyncStack(f) {
    f();
}
function readlineBuf(v) {
    if (v) {
        v = "a";
    }
}
function inJit() {
    return true;
}
function isRelazifiableFunction(f) {
    return f;
}
function bailout(f) {}
function ReadableStream() {
    return {};
}
function evalWithCache(f) {
    return eval(f);
}
function offThreadDecodeScript(f) {
    return eval(f);
}
function isLazyFunction(f) {
    if (typeof f == "function") return true;
    return false;
}
var generation = 0;

function Disjunction(alternatives) {
    return {
        type: "Disjunction",
        alternatives: alternatives,
    };
}

function Alternative(nodes) {
    return {
        type: "Alternative",
        nodes: nodes,
    };
}

function Empty() {
    return {
        type: "Empty",
    };
}

function Text(elements) {
    return {
        type: "Text",
        elements: elements,
    };
}

function Assertion(type) {
    return {
        type: "Assertion",
        assertion_type: type,
    };
}

function Atom(data) {
    return {
        type: "Atom",
        data: data,
    };
}

const kInfinity = 0x7fffffff;
function Quantifier(min, max, type, body) {
    return {
        type: "Quantifier",
        min: min,
        max: max,
        quantifier_type: type,
        body: body,
    };
}

function Lookahead(body) {
    return {
        type: "Lookahead",
        is_positive: true,
        body: body,
    };
}

function NegativeLookahead(body) {
    return {
        type: "Lookahead",
        is_positive: false,
        body: body,
    };
}

function BackReference(index) {
    return {
        type: "BackReference",
        index: index,
    };
}

function CharacterClass(ranges) {
    return {
        type: "CharacterClass",
        is_negated: false,
        ranges: ranges.map(([from, to]) => ({ from, to })),
    };
}

function NegativeCharacterClass(ranges) {
    return {
        type: "CharacterClass",
        is_negated: true,
        ranges: ranges.map(([from, to]) => ({ from, to })),
    };
}

function Capture(index, body) {
    return {
        type: "Capture",
        index: index,
        body: body,
    };
}

function AllSurrogateAndCharacterClass(ranges) {
    return Disjunction([
        CharacterClass(ranges),
        Alternative([
            CharacterClass([["\uD800", "\uDBFF"]]),
            NegativeLookahead(CharacterClass([["\uDC00", "\uDFFF"]])),
        ]),
        Alternative([
            Assertion("NOT_AFTER_LEAD_SURROGATE"),
            CharacterClass([["\uDC00", "\uDFFF"]]),
        ]),
        Text([
            CharacterClass([["\uD800", "\uDBFF"]]),
            CharacterClass([["\uDC00", "\uDFFF"]]),
        ]),
    ]);
}

// testing functions

var all_flags = ["", "i", "m", "u", "im", "iu", "mu", "imu"];

var no_unicode_flags = ["", "i", "m", "im"];

var unicode_flags = ["u", "iu", "mu", "imu"];

var no_multiline_flags = ["", "i", "u", "iu"];

var multiline_flags = ["m", "im", "mu", "imu"];

function test_flags(pattern, flags, match_only, expected) {
    return true;
}

function make_mix(tree) {
    if (tree.type == "Atom") {
        return Atom("X" + tree.data + "Y");
    }
    if (tree.type == "CharacterClass") {
        return Text([Atom("X"), tree, Atom("Y")]);
    }
    if (tree.type == "Alternative") {
        return Alternative([Atom("X"), ...tree.nodes, Atom("Y")]);
    }
    return Alternative([Atom("X"), tree, Atom("Y")]);
}

function test_mix(pattern, flags, expected) {
    test_flags(pattern, flags, false, expected);
    test_flags("X" + pattern + "Y", flags, false, make_mix(expected));
}

function test(pattern, flags, expected) {
    test_flags(pattern, flags, false, expected);
}

function test_match_only(pattern, flags, expected) {
    test_flags(pattern, flags, true, expected);
}
if (gc == undefined) {
    function gc() {
        for (let i = 0; i < 10; i++) {
            new ArrayBuffer(1024 * 1024 * 10);
        }
    }
}
function minorgc() {
    gc();
}
function detachArrayBuffer() {}
function newRope(a, b) {
    return a + b;
}
function oomAfterAllocations(v) {
    return v;
}
function assertJitStackInvariants() {}
function withSourceHook(hook, f) {
    f();
}

function orTestHelper(a, b, n) {
    var k = 0;
    for (var i = 0; i < n; i++) {
        if (a || b) k += i;
    }
    return k;
}

var lazy = 0;
function clone(f) {
    return f;
}
function shapeOf(f) {
    return {};
}
function getMaxArgs() {
    return 0xffffffff;
}

// The nearest representable values to +1.0.
const ONE_PLUS_EPSILON = 1 + Math.pow(2, -52); // 0.9999999999999999
const ONE_MINUS_EPSILON = 1 - Math.pow(2, -53); // 1.0000000000000002

{
    const fail = function (msg) {
        var exc = new Error(msg);
        try {
            // Try to improve on exc.fileName and .lineNumber; leave exc.stack
            // alone. We skip two frames: fail() and its caller, an assertX()
            // function.
            var frames = exc.stack.trim().split("\n");
            if (frames.length > 2) {
                var m = /@([^@:]*):([0-9]+)$/.exec(frames[2]);
                if (m) {
                    exc.fileName = m[1];
                    exc.lineNumber = +m[2];
                }
            }
        } catch (ignore) {
            throw ignore;
        }
        throw exc;
    };

    let ENDIAN; // 0 for little-endian, 1 for big-endian.

    // Return the difference between the IEEE 754 bit-patterns for a and b.
    //
    // This is meaningful when a and b are both finite and have the same
    // sign. Then the following hold:
    //
    //   * If a === b, then diff(a, b) === 0.
    //
    //   * If a !== b, then diff(a, b) === 1 + the number of representable values
    //                                         between a and b.
    //
    const f = new Float64Array([0, 0]);
    const u = new Uint32Array(f.buffer);
    const diff = function (a, b) {
        f[0] = a;
        f[1] = b;
        //print(u[1].toString(16) + u[0].toString(16) + " " + u[3].toString(16) + u[2].toString(16));
        return Math.abs(
            (u[3 - ENDIAN] - u[1 - ENDIAN]) * 0x100000000 +
                u[2 + ENDIAN] -
                u[0 + ENDIAN]
        );
    };

    // Set ENDIAN to the platform's endianness.
    ENDIAN = 0; // try little-endian first
    if (diff(2, 4) === 0x100000)
        // exact wrong answer we'll get on a big-endian platform
        ENDIAN = 1;

    var assertNear = function assertNear(a, b, tolerance = 1) {
        if (!Number.isFinite(b)) {
            fail(
                "second argument to assertNear (expected value) must be a finite number"
            );
        } else if (Number.isNaN(a)) {
            fail("got NaN, expected a number near " + b);
        } else if (!Number.isFinite(a)) {
            if (b * Math.sign(a) < Number.MAX_VALUE)
                fail("got " + a + ", expected a number near " + b);
        } else {
            // When the two arguments do not have the same sign bit, diff()
            // returns some huge number. So if b is positive or negative 0,
            // make target the zero that has the same sign bit as a.
            var target = b === 0 ? a * 0 : b;
            var err = diff(a, target);
            if (err > tolerance) {
                fail(
                    "got " +
                        a +
                        ", expected a number near " +
                        b +
                        " (relative error: " +
                        err +
                        ")"
                );
            }
        }
    };
}
function newExternalString(s) {
    return String(s);
}
function unboxedObjectsEnabled() {
    return true;
}
function unwrappedObjectsHaveSameShape() {
    return true;
}
function relazifyFunctions(f) {}
function isUnboxedObject() {}
function ensureFlatString(s) {
    return s;
}
function finalizeCount() {
    return 1;
}
var mandelbrotImageDataFuzzyResult = 0;
function evalReturningScope(f) {
    return eval(f);
}
function getAllocationMetadata(v) {
    return {};
}
function displayName(f) {
    return f.name;
}
function getBuildConfiguration() {
    this.debug = true;
    return this;
}
function dumpStringRepresentation() {}
function getLastWarning() {
    return null;
}
function grayRoot() {
    return new Array();
}
function byteSize(v) {
    return v.length;
}
function assertThrownErrorContains(thunk, substr) {
    try {
        thunk();
    } catch (e) {
        if (e.message.indexOf(substr) !== -1) return;
        throw new Error("Expected error containing " + substr + ", got " + e);
    }
    throw new Error(
        "Expected error containing " + substr + ", no exception thrown"
    );
}

function formatArray(arr) {
    try {
        return arr.toSource();
    } catch (e) {
        return arr.toString();
    }
}

var document = {};
function options() {}
function setTimeout() {}

function assertFalse(a) {
    assertEq(a, false);
}
function assertTrue(a) {
    assertEq(a, true);
}
function assertNotEq(found, not_expected) {
    assertEq(Object.is(found, not_expected), false);
}
function assertIteratorResult(result, value, done) {
    assertDeepEq(result.value, value);
    assertEq(result.done, done);
}
function assertIteratorNext(iter, value) {
    assertIteratorResult(iter.next(), value, false);
}
function assertIteratorDone(iter, value) {
    assertIteratorResult(iter.next(), value, true);
}

function hasPipeline() {
    try {
        Function("a |> a");
    } catch (e) {
        return false;
    }

    return true;
}

var SOME_PRIMITIVE_VALUES = [
    undefined,
    null,
    false,
    -Infinity,
    -1.6e99,
    -1,
    -0,
    0,
    Math.pow(2, -1074),
    1,
    4294967295,
    Number.MAX_SAFE_INTEGER,
    Number.MAX_SAFE_INTEGER + 1,
    1.6e99,
    Infinity,
    NaN,
    "",
    "Phaedo",
    Symbol(),
    Symbol("iterator"),
    Symbol.for("iterator"),
    Symbol.iterator,
];

function runtest(f) {
    f();
}

var bufferGlobal = [];

WScript = {
    _jscGC: gc,
    _jscLoad: function () {},
    _jscPrint: print,
    // _jscQuit: quit,
    _convertPathname: function (dosStylePath) {
        return dosStylePath.replace(/\\/g, "/");
    },
    Arguments: ["summary"],
    Echo: function () {
        WScript._jscPrint.apply(this, arguments);
    },
    LoadScriptFile: function (path) {
        WScript._jscLoad(WScript._convertPathname(path));
    },
    Quit: function () {
        // WScript._jscQuit();
    },
    Platform: {
        BUILD_TYPE: "Debug",
    },
};

function CollectGarbage() {
    WScript._jscGC();
}

function $ERROR(e) {}
