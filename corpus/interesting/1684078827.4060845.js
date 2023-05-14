var builtInPropertyNames = [
    'prototype',
    'length',
    0,
    1,
    '$1',
    'name',
    'message',
    'constructor'
];
var builtInPropertyNamesMayThrow = [
    'caller',
    'arguments'
];
function getAnException() {
    try {
        'str'();
    } catch (e) {
        return e;
    }
}
function getSpecialObjects() {
    return [
        function () {
        },
        [
            1,
            2,
            3
        ],
        /xxx/,
        RegExp,
        'blah',
        9,
        new Date(),
        getAnException()
    ];
}
var object = {};
var fun = function () {
};
var someException = getAnException();
var someDate = new Date();
var objects = [
    [
        1,
        Number.prototype
    ],
    [
        'foo',
        String.prototype
    ],
    [
        true,
        Boolean.prototype
    ],
    [
        object,
        object
    ],
    [
        fun,
        fun
    ],
    [
        someException,
        someException
    ],
    [
        someDate,
        someDate
    ]
];
function runTest(fun) {
    for (var i in objects) {
        var obj = objects[i][0];
        var chain = objects[i][1];
        var specialObjects = getSpecialObjects();
        for (var j in specialObjects) {
            var special = specialObjects[j];
            chain.__proto__ = special;
            for (var k in builtInPropertyNames) {
                var propertyName = builtInPropertyNames[k];
                fun(obj, propertyName);
            }
            for (var k in builtInPropertyNamesMayThrow) {
                var propertyName = builtInPropertyNamesMayThrow[k];
                try {
                    fun(obj, propertyName);
                } catch (e) {
                }
            }
        }
    }
}
runTest(function (obj, name) {
    return obj[name];
});
runTest(function (obj, name) {
    return obj[name] = {};
});
while (i < 107) {
    ++i;
}