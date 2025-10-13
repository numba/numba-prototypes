module {
  func.func @func(%arg0: memref<1x5x6xf64>, %arg1: memref<1x5x6xf64>) -> memref<1x5x6x2xf64, strided<[?, ?, ?, ?], offset: ?>> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<1x5x6x2xf64>
    %1 = bufferization.to_tensor %arg0 restrict : memref<1x5x6xf64>
    %inserted_slice = tensor.insert_slice %1 into %0[0, 0, 0, 0] [1, 5, 6, 1] [1, 1, 1, 2] : tensor<1x5x6xf64> into tensor<1x5x6x2xf64>
    %2 = bufferization.to_tensor %arg1 restrict : memref<1x5x6xf64>
    %inserted_slice_0 = tensor.insert_slice %2 into %inserted_slice[0, 0, 0, 1] [1, 5, 6, 1] [1, 1, 1, 2] : tensor<1x5x6xf64> into tensor<1x5x6x2xf64>
    %3 = bufferization.to_memref %inserted_slice_0 : memref<1x5x6x2xf64>
    %cast = memref.cast %3 : memref<1x5x6x2xf64> to memref<1x5x6x2xf64, strided<[?, ?, ?, ?], offset: ?>>
    return %cast : memref<1x5x6x2xf64, strided<[?, ?, ?, ?], offset: ?>>
  }
  func.func @global_init() attributes {llvm.emit_c_interface} {
    return
  }
}

