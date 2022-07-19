

public namespace input {
    public inline func mouseX() {
        return _asm_("mouseX");
    }
    public inline func mouseY() {
        return _asm_("mouseY");
    }
    public inline func isMouseDown() {
        return _asm_("mouseDown");
    }
    public inline func isKeyPressed(key) {
        return _asm_("isKeyPressed", key);
    }
    public inline func ask(message) {
        return _asm_("ask", message);
    }
}
