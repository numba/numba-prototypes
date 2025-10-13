module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @func(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg9, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg10, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.insertvalue %arg11, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.insertvalue %arg12, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.insertvalue %arg15, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.insertvalue %arg13, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.insertvalue %arg16, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.insertvalue %arg14, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.insertvalue %arg17, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = builtin.unrealized_conversion_cast %9 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x5x6xf64>
    %11 = builtin.unrealized_conversion_cast %10 : memref<1x5x6xf64> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %13 = llvm.insertvalue %arg0, %12[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.insertvalue %arg1, %13[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %arg2, %14[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.insertvalue %arg3, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.insertvalue %arg6, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %arg4, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %arg7, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %arg5, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %arg8, %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = builtin.unrealized_conversion_cast %21 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x5x6xf64>
    %23 = builtin.unrealized_conversion_cast %22 : memref<1x5x6xf64> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %24 = builtin.unrealized_conversion_cast %10 : memref<1x5x6xf64> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %25 = builtin.unrealized_conversion_cast %22 : memref<1x5x6xf64> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(5 : index) : i64
    %28 = llvm.mlir.constant(6 : index) : i64
    %29 = llvm.mlir.constant(2 : index) : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.constant(12 : index) : i64
    %32 = llvm.mlir.constant(60 : index) : i64
    %33 = llvm.mlir.constant(60 : index) : i64
    %34 = llvm.mlir.zero : !llvm.ptr
    %35 = llvm.getelementptr %34[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.mlir.constant(64 : index) : i64
    %38 = llvm.add %36, %37 : i64
    %39 = llvm.call @malloc(%38) : (i64) -> !llvm.ptr
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.sub %37, %41 : i64
    %43 = llvm.add %40, %42 : i64
    %44 = llvm.urem %43, %37  : i64
    %45 = llvm.sub %43, %44 : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr
    %47 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %48 = llvm.insertvalue %39, %47[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.insertvalue %46, %48[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.mlir.constant(0 : index) : i64
    %51 = llvm.insertvalue %50, %49[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %52 = llvm.insertvalue %26, %51[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.insertvalue %27, %52[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %54 = llvm.insertvalue %28, %53[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.insertvalue %29, %54[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.insertvalue %32, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.insertvalue %31, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.insertvalue %29, %57[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.insertvalue %30, %58[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x5x6x2xf64>
    %61 = llvm.extractvalue %25[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %62 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %63 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %64 = llvm.insertvalue %61, %63[0] : !llvm.struct<(ptr, ptr, i64)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64)> 
    %68 = llvm.extractvalue %25[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.extractvalue %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.extractvalue %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.extractvalue %25[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %72 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.extractvalue %25[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = llvm.extractvalue %25[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %75 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %76 = llvm.extractvalue %67[0] : !llvm.struct<(ptr, ptr, i64)> 
    %77 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %78 = llvm.insertvalue %76, %75[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.insertvalue %77, %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(0 : index) : i64
    %81 = llvm.insertvalue %80, %79[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(5 : index) : i64
    %83 = llvm.insertvalue %82, %81[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.mlir.constant(6 : index) : i64
    %85 = llvm.insertvalue %84, %83[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.mlir.constant(6 : index) : i64
    %87 = llvm.insertvalue %86, %85[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.mlir.constant(1 : index) : i64
    %89 = llvm.insertvalue %88, %87[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %91 = llvm.extractvalue %59[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %92 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %93 = llvm.insertvalue %91, %90[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %92, %93[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.mlir.constant(0 : index) : i64
    %96 = llvm.insertvalue %95, %94[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.mlir.constant(5 : index) : i64
    %98 = llvm.insertvalue %97, %96[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.mlir.constant(12 : index) : i64
    %100 = llvm.insertvalue %99, %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mlir.constant(6 : index) : i64
    %102 = llvm.insertvalue %101, %100[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.mlir.constant(2 : index) : i64
    %104 = llvm.insertvalue %103, %102[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.intr.stacksave : !llvm.ptr
    %106 = llvm.mlir.constant(2 : i64) : i64
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.alloca %107 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %89, %108 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %109 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %110 = llvm.insertvalue %106, %109[0] : !llvm.struct<(i64, ptr)> 
    %111 = llvm.insertvalue %108, %110[1] : !llvm.struct<(i64, ptr)> 
    %112 = llvm.mlir.constant(2 : i64) : i64
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.alloca %113 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %104, %114 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %115 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %116 = llvm.insertvalue %112, %115[0] : !llvm.struct<(i64, ptr)> 
    %117 = llvm.insertvalue %114, %116[1] : !llvm.struct<(i64, ptr)> 
    %118 = llvm.mlir.constant(1 : index) : i64
    %119 = llvm.alloca %118 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %111, %119 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %120 = llvm.alloca %118 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %117, %120 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %121 = llvm.mlir.zero : !llvm.ptr
    %122 = llvm.getelementptr %121[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %123 = llvm.ptrtoint %122 : !llvm.ptr to i64
    llvm.call @memrefCopy(%123, %119, %120) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %105 : !llvm.ptr
    %124 = llvm.extractvalue %24[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %125 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %126 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %127 = llvm.insertvalue %124, %126[0] : !llvm.struct<(ptr, ptr, i64)> 
    %128 = llvm.insertvalue %125, %127[1] : !llvm.struct<(ptr, ptr, i64)> 
    %129 = llvm.mlir.constant(0 : index) : i64
    %130 = llvm.insertvalue %129, %128[2] : !llvm.struct<(ptr, ptr, i64)> 
    %131 = llvm.extractvalue %24[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %132 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %133 = llvm.extractvalue %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %134 = llvm.extractvalue %24[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %135 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %136 = llvm.extractvalue %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %137 = llvm.extractvalue %24[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %139 = llvm.extractvalue %130[0] : !llvm.struct<(ptr, ptr, i64)> 
    %140 = llvm.extractvalue %130[1] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = llvm.insertvalue %139, %138[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.insertvalue %140, %141[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.mlir.constant(0 : index) : i64
    %144 = llvm.insertvalue %143, %142[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.mlir.constant(5 : index) : i64
    %146 = llvm.insertvalue %145, %144[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.mlir.constant(6 : index) : i64
    %148 = llvm.insertvalue %147, %146[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.mlir.constant(6 : index) : i64
    %150 = llvm.insertvalue %149, %148[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.insertvalue %151, %150[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %154 = llvm.extractvalue %59[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %155 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %156 = llvm.insertvalue %154, %153[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.insertvalue %155, %156[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.mlir.constant(1 : index) : i64
    %159 = llvm.insertvalue %158, %157[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(5 : index) : i64
    %161 = llvm.insertvalue %160, %159[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.mlir.constant(12 : index) : i64
    %163 = llvm.insertvalue %162, %161[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.mlir.constant(6 : index) : i64
    %165 = llvm.insertvalue %164, %163[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.mlir.constant(2 : index) : i64
    %167 = llvm.insertvalue %166, %165[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.intr.stacksave : !llvm.ptr
    %169 = llvm.mlir.constant(2 : i64) : i64
    %170 = llvm.mlir.constant(1 : index) : i64
    %171 = llvm.alloca %170 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %152, %171 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %172 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %173 = llvm.insertvalue %169, %172[0] : !llvm.struct<(i64, ptr)> 
    %174 = llvm.insertvalue %171, %173[1] : !llvm.struct<(i64, ptr)> 
    %175 = llvm.mlir.constant(2 : i64) : i64
    %176 = llvm.mlir.constant(1 : index) : i64
    %177 = llvm.alloca %176 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %167, %177 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %178 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %179 = llvm.insertvalue %175, %178[0] : !llvm.struct<(i64, ptr)> 
    %180 = llvm.insertvalue %177, %179[1] : !llvm.struct<(i64, ptr)> 
    %181 = llvm.mlir.constant(1 : index) : i64
    %182 = llvm.alloca %181 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %174, %182 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %183 = llvm.alloca %181 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %180, %183 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %184 = llvm.mlir.zero : !llvm.ptr
    %185 = llvm.getelementptr %184[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %186 = llvm.ptrtoint %185 : !llvm.ptr to i64
    llvm.call @memrefCopy(%186, %182, %183) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %168 : !llvm.ptr
    llvm.return %59 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
  llvm.func @_mlir_ciface_func(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %10[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %10[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.extractvalue %10[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.call @func(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %20, %arg0 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @global_init() attributes {llvm.emit_c_interface} {
    llvm.return
  }
  llvm.func @_mlir_ciface_global_init() attributes {llvm.emit_c_interface} {
    llvm.call @global_init() : () -> ()
    llvm.return
  }
}

