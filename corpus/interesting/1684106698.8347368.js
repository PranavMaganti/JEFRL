function ID(x) {
    return x;
}
(function TestBasicsString() {
    var object = {
        a: 'A',
        ['b']: 'B',
        c: 'C',
        [ID('d')]: 'D'
    };
}());
(function TestBasicsNumber() {
    var object = {
        a: 'A',
        [1]: 'B',
        c: 'C',
        [ID(2)]: 'D'
    };
}());
(function TestBasicsSymbol() {
    var sym1 = Symbol();
    var sym2 = Symbol();
    var object = {
        a: 'A',
        [sym1]: 'B',
        c: 'C',
        [ID(sym2)]: 'D'
    };
}());
(function TestToNameSideEffects() {
    var counter = 0;
    var key1 = {
        toString: function () {
            return 'b';
        }
    };
    var key2 = {
        toString: function () {
            return 'd';
        }
    };
    var object = {
        a: 'A',
        [key1]: 'B',
        c: 'C',
        [key2]: 'D'
    };
}());
(function TestToNameSideEffectsNumbers() {
    var counter = 0;
    var key1 = {
        valueOf: function () {
            return 1;
        },
        toString: null
    };
    var key2 = {
        valueOf: function () {
            return 2;
        },
        toString: null
    };
    var object = {
        a: 'A',
        [key1]: 'B',
        c: 'C',
        [key2]: 'D'
    };
}());
(function TestDoubleName() {
    var object = {
        [1.2]: 'A',
        [10000000000000000102350670204085511496304388135324745728]: 'B',
        [1e-06]: 'C',
        [-0]: 'D',
        [NaN]: 'G'
    };
}());
(function TestGetter() {
    var object = {
        get ['a']() {
            return 'A';
        }
    };
    object = {
        get b() {
        },
        get ['b']() {
            return 'B';
        }
    };
    object = {
        get c() {
        },
        get ['c']() {
        },
        get ['c']() {
            return 'C';
        }
    };
    object = {
        get ['d']() {
        },
        get d() {
            return 'D';
        }
    };
}());
(function TestSetter() {
    var calls = 0;
    var object = {
        set ['a'](_) {
            calls++;
        }
    };
    object.a = 'A';
    calls = 0;
    object = {
        set b(_) {
        },
        set ['b'](_) {
            calls++;
        }
    };
    object.b = 'B';
    calls = 0;
    object = {
        set c(_) {
        },
        set ['c'](_) {
        },
        set ['c'](_) {
            calls++;
        }
    };
    object.c = 'C';
    calls = 0;
    object = {
        set ['d'](_) {
        },
        set d(_) {
            calls++;
        }
    };
    object.d = 'D';
}());
(function TestDuplicateKeys() {
    'use strict';
    var object = {
        a: 1,
        ['a']: 2
    };
}());
(function TestProto() {
    var proto = {};
    var object = { __proto__: proto };
    object = { '__proto__': proto };
    object = { ['__proto__']: proto };
    object = {
        [ID('x')]: 'X',
        __proto__: proto
    };
}());
(function TestExceptionInName() {
    function MyError() {
    }
    ;
    function throwMyError() {
        throw new MyError();
    }
}());
(function TestNestedLiterals() {
    var array = [
        42,
        {
            a: 'A',
            ['b']: 'B',
            c: 'C',
            [ID('d')]: 'D'
        },
        43
    ];
    var object = {
        outer: 42,
        inner: {
            a: 'A',
            ['b']: 'B',
            c: 'C',
            [ID('d')]: 'D'
        },
        outer2: 43
    };
    var object = {
        outer: 42,
        array: [
            43,
            {
                a: 'A',
                ['b']: 'B',
                c: 'C',
                [ID('d')]: 'D'
            },
            44
        ],
        outer2: 45
    };
}());
label:
    for (var i = 0; i < 10; ++i) {
        let x = 'middle' + i;
        for (var j = 0; j < 10; ++j) {
            let x = 'inner' + j;
            continue label;
        }
    }
L: {
    try {
        x++;
        if (false)
            return -1;
        break L;
    } catch (o) {
        x--;
    }
}