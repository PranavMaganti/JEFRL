(function testRestIndex() {
}());
var strictTest = function () {
    'use strict';
    return (a, b, ...c) => {
        var expectedLength = a === undefined ? 0 : a - 2;
        for (var i = 2; i < a; ++i) {
        }
    };
}();
var sloppyTest = (a, b, ...c) => {
    var expectedLength = a === undefined ? 0 : a - 2;
    for (var i = 2; i < a; ++i) {
    }
};
var O = {
    strict: strictTest,
    sloppy: sloppyTest
};
(function testStrictRestParamArity() {
}());
(function testRestParamsStrictMode() {
    strictTest();
    strictTest(2, 1);
    strictTest(6, 5, 4, 3, 2, 1);
    strictTest(3, 2, 1);
    O.strict();
    O.strict(2, 1);
    O.strict(6, 5, 4, 3, 2, 1);
    O.strict(3, 2, 1);
}());
(function testRestParamsStrictModeApply() {
    strictTest.apply(null, []);
    strictTest.apply(null, [
        2,
        1
    ]);
    strictTest.apply(null, [
        6,
        5,
        4,
        3,
        2,
        1
    ]);
    strictTest.apply(null, [
        3,
        2,
        1
    ]);
    O.strict.apply(O, []);
    O.strict.apply(O, [
        2,
        1
    ]);
    O.strict.apply(O, [
        6,
        5,
        4,
        3,
        2,
        1
    ]);
    O.strict.apply(O, [
        3,
        2,
        1
    ]);
}());
(function testRestParamsStrictModeCall() {
    strictTest.call(null);
    strictTest.call(null, 2, 1);
    strictTest.call(null, 6, 5, 4, 3, 2, 1);
    strictTest.call(null, 3, 2, 1);
    O.strict.call(O);
    O.strict.call(O, 2, 1);
    O.strict.call(O, 6, 5, 4, 3, 2, 1);
    O.strict.call(O, 3, 2, 1);
}());
(function testsloppyRestParamArity() {
}());
(function testRestParamsSloppyMode() {
    sloppyTest();
    sloppyTest(2, 1);
    sloppyTest(6, 5, 4, 3, 2, 1);
    sloppyTest(3, 2, 1);
    O.sloppy();
    O.sloppy(2, 1);
    O.sloppy(6, 5, 4, 3, 2, 1);
    O.sloppy(3, 2, 1);
}());
(function testRestParamssloppyModeApply() {
    sloppyTest.apply(null, []);
    sloppyTest.apply(null, [
        2,
        1
    ]);
    sloppyTest.apply(null, [
        6,
        5,
        4,
        3,
        2,
        1
    ]);
    sloppyTest.apply(null, [
        3,
        2,
        1
    ]);
    O.sloppy.apply(O, []);
    O.sloppy.apply(O, [
        2,
        1
    ]);
    O.sloppy.apply(O, [
        6,
        5,
        4,
        3,
        2,
        1
    ]);
    O.sloppy.apply(O, [
        3,
        2,
        1
    ]);
}());
(function testRestParamssloppyModeCall() {
    sloppyTest.call(null);
    sloppyTest.call(null, 2, 1);
    sloppyTest.call(null, 6, 5, 4, 3, 2, 1);
    sloppyTest.call(null, 3, 2, 1);
    O.sloppy.call(O);
    O.sloppy.call(O, 2, 1);
    O.sloppy.call(O, 6, 5, 4, 3, 2, 1);
    O.sloppy.call(O, 3, 2, 1);
}());
(function testUnmappedArguments() {
}());
var y = new zeros();