var x = 0;
function Funky(a, b, c) {
    return 7;
}
Number.prototype.__proto__ = Funky;
Number.prototype.__proto__ = [
    1,
    2,
    3
];
with ({ e: 42 }) {
    return x;
}