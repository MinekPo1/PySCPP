# PySCPP

[![Documentation Status](https://readthedocs.org/projects/pyscpp/badge/?version=latest)](https://pyscpp.readthedocs.io/en/latest/?badge=latest)

This is a python implementation of both the SCPP compiler and the SLVM. See [here for SCPP and SLVM spec](https://www.github.com/Its-Jakey/SCPP).

Note: This is a work in progress.

Progress:

- 🟠 Compiler
  - 🟢 Parser & Lexer
  - 🟠 Code optimization
  - ⚫ Assembly
- 🟠 VM
  - 🟢 instructions
  - ⚫ graphics

## Non spec behavior

### Not supported features

- main function being called

    If you need this, stop and go get some help. Touching grass is also a good idea.

### Non backwards compatible features

- you can access and edit addresses of variables.

        var a = 1;
        var $b = $a;
        b = 2;
        print(a); // 2

- multi layer pointers are supported.

        var a = 1;
        var b = $a;
        var c = $b;
        ~~~c = 2;
        print(a); // 2

- functions may be inline

    Inline functions cannot be recursive, however they can be more efficient than regular functions.

        inline func foo() {
                println("foo");
        }
        func bar() {
                println("bar");
        }
        foo();
        bar();

    This is compiles to roughly the following:

        func bar() {
                println("bar");
        }
        println("foo");
        bar();

- Arrays cannot be placed in arrays.

- `_asm_` and `_getValuOfA_` are keywords instead of methods and can be used without parentheses.

- functions, variables and namespaces can be explicitly made private with the `private` keyword.

- namespaces can be placed in namespaces, can be public or private and can have a default access descriptor.

        namespace foo { // public, default access is private
            namespace public bar { // private
                private namespace baz { // public, inherits default access from bar
                    var a; // public
                }
                var b; // public
            }
            var c // private
        }

- concatenation operator (`..`) is supported.
