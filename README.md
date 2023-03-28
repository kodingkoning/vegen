# To build
You need llvm 12.0.1 to build VeGen. That can be found at: https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/clang+llvm-12.0.1-x86\_64-linux-gnu-ubuntu-16.04.tar.xz. This is compatible with gcc 7.2.0.
You need `cmake` to build VeGen. VeGen depends on LLVM, and therefore you will
also need the *same* compiler that you used to compile LLVM to build VeGen for ABI compatibility.
```bash
export CXX=<the same c++ compiler you used to build llvm>
mkdir build
cd build
cmake $path_to_vegen
```

# To use
After building VeGen, you can in principle use `vegen-clang` and `vegen-clang++` as a drop-in replacement for clang.

# Directory structure
`/gslp` (`gslp` stands for Generalized SLP) 
  contains the vectorization heuristic and the code generation implementation.
`/sema` contains the semantics handling logic.
`/gslp/target-sema` and `gslp/target-wrappers` contains the code generated from `/sema`.
