public namespace math {
    public inline func round(x, places) {
        return _asm_("round", x, places);
    }
    public inline func floor(x, places) {
        return _asm_("floor", x, places);
    }
    public inline func ceil(x, places) {
        return _asm_("ceil", x, places);
    }
    public inline func sin(x) {
        return _asm_("sin", x);
    }
    public inline func cos(x) {
        return _asm_("cos", x);
    }
    public inline func sqrt(x) {
        return _asm_("sqrt", x);
    }
    public inline func atan2(x, y) {
        return _asm_("atan2", x, y);
    }
    public inline func negate(x) {
        if (x > 0)
            return 0 - x;
        return x;
    }
    public inline func min(a, b) {
        if (a < b) {
            return a;
        }
        return b;
    }
    public inline func max(a, b) {
        if (a > b) {
            return a;
        }
        return b;
    }
    public inline func inRange(x, a, b) {
        return x > a && x < b;
    }
    public inline func abs(x) {
        if (x < 0)
            return (0 - x);
        return x;
    }
}
