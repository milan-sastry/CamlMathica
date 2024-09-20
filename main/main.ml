(* Main program for assignment 3.  You can put whatever you want in here.
 * You can run it by "dune exec main"
 *)

open A3lib
(* You can say "open Mapreduce", etc. to avoid having to use fully
    qualified names *)

let _ = assert ( Mapreduce.negate_all [1; -2; 0] = [-1; 2; 0] )

let _ = assert (Expression.evaluate (ExpressionLibrary.parse "x*x + 3") 2.0 = 7.0 )
