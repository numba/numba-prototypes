module {
  func.func @func(%arg0: memref<1x5x6xf64>, %arg1: memref<1x5x6xf64>) -> memref<1x5x6x2xf64> attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5x6x2xf64>
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<1x5x6xf64> -> memref<f64>, index, index, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [5, 6], strides: [6, 1] : memref<f64> to memref<5x6xf64>
    %reinterpret_cast_0 = memref.reinterpret_cast %alloc to offset: [0], sizes: [5, 6], strides: [12, 2] : memref<1x5x6x2xf64> to memref<5x6xf64, strided<[12, 2]>>
    memref.copy %reinterpret_cast, %reinterpret_cast_0 : memref<5x6xf64> to memref<5x6xf64, strided<[12, 2]>>
    %base_buffer_1, %offset_2, %sizes_3:3, %strides_4:3 = memref.extract_strided_metadata %arg1 : memref<1x5x6xf64> -> memref<f64>, index, index, index, index, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [0], sizes: [5, 6], strides: [6, 1] : memref<f64> to memref<5x6xf64>
    %reinterpret_cast_6 = memref.reinterpret_cast %alloc to offset: [1], sizes: [5, 6], strides: [12, 2] : memref<1x5x6x2xf64> to memref<5x6xf64, strided<[12, 2], offset: 1>>
    memref.copy %reinterpret_cast_5, %reinterpret_cast_6 : memref<5x6xf64> to memref<5x6xf64, strided<[12, 2], offset: 1>>
    return %alloc : memref<1x5x6x2xf64>
  }
  func.func @global_init() attributes {llvm.emit_c_interface} {
    return
  }
}

