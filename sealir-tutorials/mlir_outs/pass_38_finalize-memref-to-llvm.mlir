module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @func(%arg0: memref<1x5x6xf64>, %arg1: memref<1x5x6xf64>) -> memref<1x5x6x2xf64> attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<1x5x6xf64> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<1x5x6xf64> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(5 : index) : i64
    %4 = llvm.mlir.constant(6 : index) : i64
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(12 : index) : i64
    %8 = llvm.mlir.constant(60 : index) : i64
    %9 = llvm.mlir.constant(60 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.sub %13, %17 : i64
    %19 = llvm.add %16, %18 : i64
    %20 = llvm.urem %19, %13  : i64
    %21 = llvm.sub %19, %20 : i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %23 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %24 = llvm.insertvalue %15, %23[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %2, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %3, %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.insertvalue %4, %29[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %5, %30[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %8, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %7, %32[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %5, %33[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.insertvalue %6, %34[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = builtin.unrealized_conversion_cast %35 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x5x6x2xf64>
    %37 = llvm.extractvalue %1[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %40 = llvm.insertvalue %37, %39[0] : !llvm.struct<(ptr, ptr, i64)> 
    %41 = llvm.insertvalue %38, %40[1] : !llvm.struct<(ptr, ptr, i64)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr, ptr, i64)> 
    %44 = llvm.extractvalue %1[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.extractvalue %1[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.extractvalue %1[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = llvm.extractvalue %1[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %48 = llvm.extractvalue %1[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.extractvalue %1[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %50 = llvm.extractvalue %1[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.extractvalue %43[0] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = llvm.insertvalue %52, %51[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %53, %54[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mlir.constant(0 : index) : i64
    %57 = llvm.insertvalue %56, %55[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mlir.constant(5 : index) : i64
    %59 = llvm.insertvalue %58, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.mlir.constant(6 : index) : i64
    %61 = llvm.insertvalue %60, %59[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mlir.constant(6 : index) : i64
    %63 = llvm.insertvalue %62, %61[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.insertvalue %64, %63[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.extractvalue %35[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.insertvalue %67, %66[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %68, %69[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mlir.constant(0 : index) : i64
    %72 = llvm.insertvalue %71, %70[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.mlir.constant(5 : index) : i64
    %74 = llvm.insertvalue %73, %72[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.mlir.constant(12 : index) : i64
    %76 = llvm.insertvalue %75, %74[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.mlir.constant(6 : index) : i64
    %78 = llvm.insertvalue %77, %76[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mlir.constant(2 : index) : i64
    %80 = llvm.insertvalue %79, %78[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.intr.stacksave : !llvm.ptr
    %82 = llvm.mlir.constant(2 : i64) : i64
    %83 = llvm.mlir.constant(1 : index) : i64
    %84 = llvm.alloca %83 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %65, %84 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %85 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %86 = llvm.insertvalue %82, %85[0] : !llvm.struct<(i64, ptr)> 
    %87 = llvm.insertvalue %84, %86[1] : !llvm.struct<(i64, ptr)> 
    %88 = llvm.mlir.constant(2 : i64) : i64
    %89 = llvm.mlir.constant(1 : index) : i64
    %90 = llvm.alloca %89 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %80, %90 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %91 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %92 = llvm.insertvalue %88, %91[0] : !llvm.struct<(i64, ptr)> 
    %93 = llvm.insertvalue %90, %92[1] : !llvm.struct<(i64, ptr)> 
    %94 = llvm.mlir.constant(1 : index) : i64
    %95 = llvm.alloca %94 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %87, %95 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %96 = llvm.alloca %94 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %93, %96 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %97 = llvm.mlir.zero : !llvm.ptr
    %98 = llvm.getelementptr %97[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
    llvm.call @memrefCopy(%99, %95, %96) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %81 : !llvm.ptr
    %100 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %101 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %102 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %103 = llvm.insertvalue %100, %102[0] : !llvm.struct<(ptr, ptr, i64)> 
    %104 = llvm.insertvalue %101, %103[1] : !llvm.struct<(ptr, ptr, i64)> 
    %105 = llvm.mlir.constant(0 : index) : i64
    %106 = llvm.insertvalue %105, %104[2] : !llvm.struct<(ptr, ptr, i64)> 
    %107 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %108 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %109 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %110 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %111 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %112 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %113 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %114 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %115 = llvm.extractvalue %106[0] : !llvm.struct<(ptr, ptr, i64)> 
    %116 = llvm.extractvalue %106[1] : !llvm.struct<(ptr, ptr, i64)> 
    %117 = llvm.insertvalue %115, %114[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.insertvalue %116, %117[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.mlir.constant(0 : index) : i64
    %120 = llvm.insertvalue %119, %118[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.mlir.constant(5 : index) : i64
    %122 = llvm.insertvalue %121, %120[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.mlir.constant(6 : index) : i64
    %124 = llvm.insertvalue %123, %122[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.mlir.constant(6 : index) : i64
    %126 = llvm.insertvalue %125, %124[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.constant(1 : index) : i64
    %128 = llvm.insertvalue %127, %126[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.extractvalue %35[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %131 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %132 = llvm.insertvalue %130, %129[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %131, %132[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.mlir.constant(1 : index) : i64
    %135 = llvm.insertvalue %134, %133[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(5 : index) : i64
    %137 = llvm.insertvalue %136, %135[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mlir.constant(12 : index) : i64
    %139 = llvm.insertvalue %138, %137[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.mlir.constant(6 : index) : i64
    %141 = llvm.insertvalue %140, %139[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(2 : index) : i64
    %143 = llvm.insertvalue %142, %141[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.intr.stacksave : !llvm.ptr
    %145 = llvm.mlir.constant(2 : i64) : i64
    %146 = llvm.mlir.constant(1 : index) : i64
    %147 = llvm.alloca %146 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %128, %147 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %148 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %149 = llvm.insertvalue %145, %148[0] : !llvm.struct<(i64, ptr)> 
    %150 = llvm.insertvalue %147, %149[1] : !llvm.struct<(i64, ptr)> 
    %151 = llvm.mlir.constant(2 : i64) : i64
    %152 = llvm.mlir.constant(1 : index) : i64
    %153 = llvm.alloca %152 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %143, %153 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %154 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %155 = llvm.insertvalue %151, %154[0] : !llvm.struct<(i64, ptr)> 
    %156 = llvm.insertvalue %153, %155[1] : !llvm.struct<(i64, ptr)> 
    %157 = llvm.mlir.constant(1 : index) : i64
    %158 = llvm.alloca %157 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %150, %158 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %159 = llvm.alloca %157 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %156, %159 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %160 = llvm.mlir.zero : !llvm.ptr
    %161 = llvm.getelementptr %160[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %162 = llvm.ptrtoint %161 : !llvm.ptr to i64
    llvm.call @memrefCopy(%162, %158, %159) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %144 : !llvm.ptr
    return %36 : memref<1x5x6x2xf64>
  }
  func.func @global_init() attributes {llvm.emit_c_interface} {
    return
  }
}

