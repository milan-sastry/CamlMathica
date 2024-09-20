# Assignment 3: Map and Caml-Mathica

## Quick Links:

- [Part 1](#part-1-higher-order-functions-mapreduceml)
- [Part 2](#part-2-a-language-for-symbolic-differentiation-expressionml)
- [Compiling](#how-to-compile-and-test-part-2)
- [Problem 2.1](#problem-21-expression-evaluation)
- [Problem 2.2](#problem-22-derivatives)
- [Problem 2.3](#problem-23-zero-finding)
- [Problem 2.4](#problem-24-symbolic-zero-finding)
- [Problem 2.5 (optional)](#optional-problem-25-forward-mode-automatic-differentiation)
<!-- - [Problem 2.5 (optional)](#optional-problem-25-the-amazing-effectiveness-of-newtons-method) -->

## Objectives

In this assignment, you will be given the opportunity to practice functional
decomposition of complex problems into smaller, simpler ones. More specifically,
you will look at functional decomposition of list processing problems in to
subproblems that can be implemented by map, reduce, fold and filter functions.
Functional programmers do this all the time in day-to-day programming tasks, but
it is also the basis for programming parallel applications in frameworks like
Hadoop or Google's Map-Reduce (it's no coincidence the names are the same!)

In addition, you will write your own language for symbolic differentiation using
OCaml. This will illustrate just how easy it is to develop your own mini-
language inside a functional language like OCaml using recursive data types and
pattern matching. A key idea here is that programs can be represented very
effectively using OCaml datatypes. Once we have represented programs using
datatypes, we can analyze them and transform them in various ways. This is what
compilers of all kinds do (like the OCaml compiler itself). In this case our
"programs" are small arithmetic expressions, but the idea generalizes to
arbitrarily complex programming languages.

## Getting Started

You should see several files in this repository. The main ones you need to
concern yourself with are:
- `mapreduce.ml` which is a self-contained series of exercises involving higher
order functions.
- `expression.ml` in which you will develop functions to perform symbolic
differentiation of algebraic expressions.
- Support code can be found in `ast.ml` and `expressionLibrary.ml`.
- `signature.txt` in which you will record your sources.

A few important things to remember before you start:
- This assignment must be done individually.
- Make sure that the functions you are asked to write have the correct names and
    the number and type of arguments specified in the assignment. Programs that
    do not compile will be subject to penalties on top of regular deductions.
- Style is naturally subjective, but the
    [COS 326 style guide](https://www.cs.princeton.edu/courses/archive/fall23/cos326/style.php)
    provides some good rules of thumb.
- As always, think first; write second; test and revise third. Aim for elegance.
- Compilation: `dune build`

## Part 1: Higher Order Functions (`mapreduce.ml`)

Map, filter and fold are functions that capture extremely common *recursion
patterns* over lists. A good functional programmer uses these functions to
construct solutions to interesting problems using very little code. In this
part, you will get practice with higher-order functions by using map and fold to
write a number of functions.
- map is implemented in OCaml by the function `List.map`.
- filter is implemented in OCaml by the function `List.filter`.
- fold is implemented in OCaml by the function `List.fold_right`. However, the
    standard OCaml library function has its arguments in an inconvenient order.
    Thus, we have provided you with the function `reduce` which computes
    identically to `fold_right` but takes arguments in a different order
    (discuss with your preceptor why this ordering is more useful).

All instructions for Part 1 can be found in `mapreduce.ml`, but as a reminder,
the `rec` keyword and other explicit recursive solutions are prohibited for this
part. You should be using higher-order functions to complete your tasks.

## Part 2: A Language for Symbolic Differentiation (`expression.ml`)

In the Summer of 1958,
[John McCarthy](https://en.wikipedia.org/wiki/John_McCarthy_%28computer_scientist%29)
(recipient of the [Turing Award](https://en.wikipedia.org/wiki/Turing_Award) in
1971) made a major contribution to the field of programming languages. With the
objective of writing a program that performed symbolic differentiation of
algebraic expressions in a effective way, he noticed that some features that
would have helped him to accomplish this task were absent in the programming
languages of that time. This led him to the invention of LISP (published in
Communications of the ACM in 1960) and other ideas, such as list processing (the
name Lisp derives from "List Processing"), recursion, and garbage collection,
which are essential to modern programming languages. Nowadays, symbolic
differentiation of algebraic expressions is a task that can be conveniently
accomplished on modern mathematical packages, such as Mathematica and Maple.

The objective of this part is to build a language that can differentiate and
evaluate symbolically represented mathematical expressions that are functions of
a single variable. Symbolic expressions consist of numbers, variables, and a
subset of the standard math functions (plus, minus, and times).

**Conceptual Overview**

To get you started, we have provided the datatype that defines the abstract
syntax tree for such expressions in `ast.ml`.

```
(* abstract syntax tree *)

(* Binary operators. *)
type binop = Add | Sub | Mul

type expression =
  | Num of float
  | Var
  | Binop of binop * expression * expression
```

`Var` presents an occurrence of the single variable "x". `Binop(Add, Var, Num
3.0)` represents the expression that adds 3.0 to x. The operators `Sub` and
`Mul` refer to subtraction and multiplication, respectively. More complicated
mathematical expressions involving addition, subtraction, multiplication,
constants and the variable x can be constructed using combinations of the
constructors in the above datatype definition. For example,

`"2.0*x + (x*x - 3.0)"`

can be represented as:

```
Binop(Add, Binop(Mul, Num(2.0), Var), Binop(Sub, Binop(Mul,Var,Var), Num(3.0))))
```

Each such expression represents a tree where nodes are the constructors and the
children of each node are the specific operator to use and the arguments of that
constructor. Such a tree is called an abstract syntax tree (or AST for short).

## How to Compile and Test Part 2

The code you will be editing is in `expression.ml`. We have used modules to keep
this file clean and easy to navigate. There are several ways to compile and test
your code, but this is the easiest:
- Edit your code until you are ready to test some part.
- Type `make expression` in your shell, which will identify any compilation
    errors.
- Run the compiled code with `./expression.byte`, which will invoke any tests
    you have constructed.
-  Evaluate the results of your tests and repeat.

## Provided Infrastructure

We have provided some functions to create and manipulate `expression` values.
`checkexp` is contained in `expression.ml`. The others are contained in
`expressionLibrary.ml`.
- `parse`: translates a string in infix form (such as `"x*x - 3.0*x + 2.5"`)
    into an `expression` (treating "x" as the variable). The `parse` function
    parses according to the standard order of operations&mdash;so `"5+x*8"` will
    be read as `"5+(x*8)"`.
- `to_string`: prints expressions in a readable form, using infix notation. This
    function adds parentheses around every binary operation so that the output
    is completely unambiguous.
- `make_exp`: takes in a length $\ell$ and returns a randomly generated
    expression of length at most $2^\ell$.
- `rand_exp_str`: takes in a length $\ell$ and returns a string representation
    of length at most $2^\ell$.
- `checkexp`: takes in a string expression and an x value and prints the results
    of calling every function to be tested except `find_zero`.

## Problem Instructions

For this portion of the assignment, you should fill in function definitions in
`expression.ml`.

### Problem 2.1: Expression Evaluation

Your first take is to write a function that will evaluate (i.e., interpret) an
expression. Given an expression `e` and a floating point value `v` for `x`,
produce the floating point result of evaluating expression `e` when `x` is `v`.
For example,

```
evaluate (parse "x*x + 3") 2.0 = 7.0
```

### Problem 2.2: Derivatives

Next, we want to develop a function that takes an expression `e` as its argument
and returns an expression `e'` representing the derivative of the expression
with respect to x. This process is referred to as symbolic differentiation.

If you don't remember your calculus, here's the necessary crib sheet, some
formulae for computing derivatives that you will use:

```
derivative(f + g)(x) = derivative(f)(x) + derivative(g)(x)
derivative(f - g)(x) = derivative(f)(x) - derivative(g)(x)
derivative(f * g)(x) = derivative(f)(x) * g(x) + f(x) * derivative(g)(x)
```

In addition, the derivative of any constant value is 0 and the derivative of a
variable x is 1.

Your task is to implement the `derivative` function. The type of this function
is `expression -> expression`. The result of your function must be correct, but
need not be expressed in the simplest form. Take advantage of this in order to
keep the code in this part as short as possible. You can implement this function
in as little as 20&ndash;30 lines of code.

To help you, we provide a function, `checkexp`, which checks parts 2.1-2.3 for a
given input. The portions of the function that require your attention read
`failwith "Not implemented"`. Do not attempt to run the function until you have
replaced all of the `failwith` expressions with valid code.

### Problem 2.3: Zero Finding

One application of the derivative of a function is to find zeros of a function.
One way to do so is
[Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) The function
should take an expression, a guess for the zero, a precision requirement, and a
limit to the number of iterations. If a guess `g` (to include the starting
guess) is sufficient to find a zero within the required precision, it should
immediately return `Some g`. Otherwise, so long as the limit has not been
reached, it should produce a new guess and try again. It should return `None` if
no zero was found within the desired precision by the time the limit was
reached.

Your task is to implement the `find_zero : expression -> float -> float -> int
-> float option` function. <!-- Note that there are cases where Newton's method will
fail to produce a zero, such as for the function $x^{1/3}$. You are not
responsible for finding a zero in those cases, but just for the correct
implementation of Newton's method.
-->

**Note:** If the expression that `find_zero` is operating on is `"f(x)"` and the
precision is `epsilon`, we are asking you to find a value `x` such that `|f(x)|
< epsilon`. That is, the value that the expression evaluates to at `x` is
"within `epsilon`" of 0.

We are **not** requiring you to find an `x` such that `|x - x₀| < epsilon` for
some `x₀` for which `f(x₀) = 0`.

### Problem 2.4: Symbolic Zero-Finding

The function you wrote above allows you to find the zero (or a zero) of most
functions that can be represented with our AST. We could also extend our little
language with other operations like sin, cos, exponentiation, logarithms, etc.,
and this method would continue to work. This makes it quite powerful. However,
in addition to numeric solving like this, Mathematica and many similar programs
can perform symbolic algebra. These programs can solve equations using
techniques similar to those you learned in middle and high school (as well as
more advanced techniques for more complex equations) to get *exact*, rather than
approximate answers.

Performing symbolic manipulation on complex expressions is quite difficult, and
we do not expect you to do it. However, there is one type of expression for
which this is not so difficult. These are polynomial expressions that can be
simplified to the form *ax + b*. You likely learned how to solve equations of
the form *ax + b=0* years ago, and can apply the same skills in writing a
program to solve these.

More specifically, for the purposes of this question, a *degree-one expression*
is one that can be simplified to the form *ax + b*.

Your task is to write the function `find_zero_exact`, which will exactly find
the zero of those degree-one expressions that do have zeros. More specifically,
for degree-one expressions that do have zeros your function should return `Some`
of an expression that
- contains no variables
- evaluates to the zero of the given expression and
- is exact.

If the expression is not degree one or has no zero, return `None`.

Note: degree one expressions need not be as simple as *ax + b*. Something like
*5x - 3 + 2(x - 8)* is also a degree-one expression since it can be turned into
*ax + b* by distributing and simplifying. Likewise *x\*x - x\*x + x* can be
simplified to *x* since the quadratic expression cancels.

You may also assume idealized floating point arithmetic (`+.`, `-.`, `*.` never
overflow and result in exact values). If you are given an expression that is
equivalent to *ax + b* where *a* is equal to 0.0, you should return `None`, as
this is not a degree-one expression (it is a degree-zero expression).

You will need to think about how to handle various forms of degree one
expressions. You will also need to think about how to determine whether an
expression is degree one in the first place. You should aim for a very general
solution&mdash;trying to solve this problem with a list of special cases for
particular kinds of expressions (like *x\*x - x\*x*) will not lead to a general
solution. We suggest converting each expression into some kind of standard form,
and then manipulating those standard forms. In particular, consider how you
might represent and program with polynomials. Unlike previous questions, you
will do best here if you build yourself a collection of useful helper functions.

How you handle these operations must be, at the core, symbolic manipulation
instead of a series of evaluation steps. An example of the latter would be first
confirming degree-one by evaluating the second derivative and comparing with 0,
and then evaluating the function at 0 and 1 to determine the two coefficients of
the closed-form solution `x = -a₀/a₁`. Do not do this, as computational
solutions like this will receive no credit.

### Optional Problem 2.5: Forward-mode automatic differentiation

In `expression.ml` you'll find the optional problem `evaluate2`: implement
*forward-mode automatic differentiation*.

The implementation is *easy*. The hard part is figuring out what is forward-mode
automatic differentiation. It's explained in section 3.1 (and Table 2)
of
[Automatic Differentiation in Machine Learning: A Survey](https://jmlr.org/papers/volume18/17-468/17-468.pdf).

Automatic Differentiation (AD) is really important now in machine learning, as
an efficient algorithm for training neural nets.

## Handin Instructions and Grading Information

This problem set is to be done individually.

Your assignment will be automatically submitted every time you push your changes
to your GitHub repository. Within a couple minutes of your submission, the
autograder will make a comment on your commit listing the output of our testing
suite when run against your code. **Note that you will be graded only on your
changes to `mapreduce.ml` and `expression.ml`**, and not on your changes to any
other files.

You may submit and receive feedback in this way as many times as you like,
whenever you like.  When you have submitted your final version, make sure to click `Check Submitted Files` because that output is what the graders will start with (along with reading your .ml files).

It is much better to turn in an assignment that compiles but is incomplete than to turn in an assignment that does not compile. If you do not have a compiling version of a
function, please comment out all your code for that function rather than
reverting to the `failwith "Not implemented"` line; this way your files will
still compile and will not disrupt the automatic grader. Document in a comment
anything you tried or any problems you had with a particular portion of the
assignment.

There are 30 total points in this assignment, broken down as 8 for problem 1.1,
9 for problem 1.2, and 13 for problem 2. Further subdivisions are as follows:
- Each subproblem 1.1.a-1.1.d: 2
- Each subproblem 1.2.a-1.2.c: 3
- Problem 2.1: 2
- Problem 2.2: 3
- Problem 2.3: 3
- Problem 2.4: 5
