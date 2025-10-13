"builtin.module"() ({
  "func.func"() <{function_type = (memref<30x40x50xf64>, memref<30x5x50xf64>) -> memref<f64>, sym_name = "func"}> ({
  ^bb0(%arg0: memref<30x40x50xf64>, %arg1: memref<30x5x50xf64>):
    %0 = "bufferization.to_tensor"(%arg1) <{restrict}> : (memref<30x5x50xf64>) -> tensor<30x5x50xf64>
    %1 = "bufferization.to_tensor"(%arg0) <{restrict}> : (memref<30x40x50xf64>) -> tensor<30x40x50xf64>
    %2 = "tensor.insert_slice"(%0, %1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 30, 5, 50>, static_strides = array<i64: 1, 1, 1>}> : (tensor<30x5x50xf64>, tensor<30x40x50xf64>) -> tensor<30x40x50xf64>
    "func.return"() : () -> ()
  }) {llvm.emit_c_interface} : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "global_init"}> ({
    "func.return"() : () -> ()
  }) {llvm.emit_c_interface} : () -> ()
}) : () -> ()
