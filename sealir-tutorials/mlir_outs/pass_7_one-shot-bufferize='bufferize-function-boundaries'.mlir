module {
  func.func @func(%arg0: memref<1x5x6xf64>, %arg1: memref<1x5x6xf64>) -> memref<1x5x6x2xf64> attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5x6x2xf64>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<1x5x6xf64> into memref<5x6xf64>
    %subview = memref.subview %alloc[0, 0, 0, 0] [1, 5, 6, 1] [1, 1, 1, 2] : memref<1x5x6x2xf64> to memref<5x6xf64, strided<[12, 2]>>
    memref.copy %collapse_shape, %subview : memref<5x6xf64> to memref<5x6xf64, strided<[12, 2]>>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<1x5x6xf64> into memref<5x6xf64>
    %subview_1 = memref.subview %alloc[0, 0, 0, 1] [1, 5, 6, 1] [1, 1, 1, 2] : memref<1x5x6x2xf64> to memref<5x6xf64, strided<[12, 2], offset: 1>>
    memref.copy %collapse_shape_0, %subview_1 : memref<5x6xf64> to memref<5x6xf64, strided<[12, 2], offset: 1>>
    %cast = memref.cast %alloc : memref<1x5x6x2xf64> to memref<1x5x6x2xf64, strided<[?, ?, ?, ?], offset: ?>>
    return %alloc : memref<1x5x6x2xf64>
  }
  func.func @global_init() attributes {llvm.emit_c_interface} {
    return
  }
}

