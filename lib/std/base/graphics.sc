

public namespace graphics {
    public inline func putPixel(x, y) {
        _asm_("putPixel", x, y);
    }
    public inline func drawLine(x0, y0, x1, y1) {
        _asm_("putLine", x0, y0, x1, y1);
    }
    public inline func fillRect(x, y, width, height) {
        _asm_("putRect", x, y, width, height);
    }
    public inline func setColor(color) {
        _asm_("setColor", color);
    }
    public inline func setStrokeWidth(strokeWidth) {
        _asm_("setStrokeWidth", strokeWidth);
    }
    public inline func clear() {
        _asm_("clg");
    }
    public inline func drawString(string) {
        _asm_("drawText", string);
    }
    public inline func drawStringLine(line) {
        _asm_("drawText", line);
        _asm_("newLine");
    }
    public inline func createColor(r, g, b) {
        return _asm_("createColor", r, g, b);
    }
    public inline func flip() {
        _asm_("graphicsFlip");
    }
    public inline func newLine() {
        _asm_("newLine");
    }
    public inline func goto(x, y) {
        _asm_("goto", x, y);
    }
}
