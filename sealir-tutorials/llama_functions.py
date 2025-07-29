
from mlir.ir import *
from mlir.dialects import arith, memref, scf, func, linalg, math
import numpy as np
from mlir.passmanager import PassManager
import mlir.execution_engine as execution_engine
import mlir.runtime as runtime
import ctypes

_DEBUG = False

class Backend:
    fn_counter = {}

    def run_passes(self, module):
        if _DEBUG:
            module.dump()

        if _DEBUG:
            module.context.enable_multithreading(False)
        if _DEBUG:
            # notebook may hang if ir_printing is enabled and and MLIR failed.
            pass_man.enable_ir_printing()

        pass_man = PassManager(context=module.context)
        pass_man.add("convert-linalg-to-loops")
        pass_man.add("convert-scf-to-cf")
        pass_man.add("finalize-memref-to-llvm")
        pass_man.add("convert-math-to-libm")
        pass_man.add("convert-func-to-llvm")
        pass_man.add("convert-index-to-llvm")
        pass_man.add("reconcile-unrealized-casts")
        pass_man.enable_verifier(True)
        pass_man.run(module.operation)
        # Output LLVM-dialect MLIR
        if _DEBUG:
            module.dump()
        return module

    def jit_compile_extra(
        self,
        llmod,
        input_types,
        output_types,
        function_name="func",
        exec_engine=None,
        is_ufunc=False,
        **execution_engine_params,
    ):
        # Converts the MLIR module into a JIT-callable function.
        # Use MLIR's own internal execution engine
        if exec_engine is None:
            engine = execution_engine.ExecutionEngine(
                llmod, **execution_engine_params
            )
        else:
            engine = exec_engine

        assert (
            len(output_types) == 1
        ), "Execution of functions with output arguments > 1 not supported"
        nout = len(output_types)

        # Build a wrapper function
        def jit_func(*args):
            if is_ufunc:
                input_args = args[:-nout]
                output_args = args[-nout:]
            else:
                input_args = args
                output_args = [None]
            assert len(input_args) == len(input_types)
            for arg, arg_ty in zip(input_args, input_types):
                # assert isinstance(arg, arg_ty)
                # TODO: Assert types here
                pass
            # Transform the input arguments into C-types
            # with their respective values. All inputs to
            # the internal execution engine should
            # be C-Type pointers.
            input_exec_ptrs = [
                self.get_exec_ptr(ty, val)[0]
                for ty, val in zip(input_types, input_args)
            ]
            # Invokes the function that we built, internally calls
            # _mlir_ciface_function_name as a void pointer with the given
            # input pointers, there can only be one resulting pointer
            # appended to the end of all input pointers in the invoke call.
            res_ptr, res_val = self.get_exec_ptr(
                output_types[0], output_args[0]
            )
            engine.invoke(function_name, *input_exec_ptrs, res_ptr)

            return self.get_out_val(res_ptr, res_val)

        return jit_func


    @classmethod
    def get_exec_ptr(self, mlir_ty, val):
        """Get Execution Pointer

        Convert MLIR types to C-types and allocate memory for the value.
        """
        if isinstance(mlir_ty, IntegerType):
            val = 0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_int64(val))        
        elif isinstance(mlir_ty, IndexType):
            val = 0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_int64(val))
        elif isinstance(mlir_ty, F32Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_float(val))
        elif isinstance(mlir_ty, F64Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_double(val))
        elif isinstance(mlir_ty, MemRefType):
            if isinstance(mlir_ty.element_type, F64Type):
                np_dtype = np.float64
            elif isinstance(mlir_ty.element_type, F32Type):
                np_dtype = np.float32
            else:
                raise TypeError(
                    "The current array element type is not supported"
                )

            if val is None:
                if not mlir_ty.has_static_shape:
                    raise ValueError(f"{mlir_ty} does not have static shape")
                val = np.zeros(mlir_ty.shape, dtype=np_dtype)

            ptr = ctypes.pointer(
                ctypes.pointer(runtime.get_ranked_memref_descriptor(val))
            )

        return ptr, val

    @classmethod
    def get_out_val(cls, res_ptr, res_val):
        if isinstance(res_val, np.ndarray):
            return res_val
        else:
            return res_ptr.contents.value
    
    def gen_fn_name(self, name_string):
        curr_counter = self.fn_counter.get(name_string, 0)
        fn_name = f"{name_string}_{str(curr_counter)}"
        self.fn_counter[name_string] = curr_counter + 1
        return fn_name

    def generate_arange(self, module, dtype):
        fn_name = self.gen_fn_name("arange")

        with module.context:
            with InsertionPoint(module.body), Location.unknown():
                element_type = F64Type.get()
                memref_type = MemRefType.get([ShapedType.get_dynamic_size()], element_type)
                func_type = FunctionType.get([element_type, element_type, element_type, memref_type], [])

                func_op = func.FuncOp(fn_name, func_type)
                func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

                with InsertionPoint(func_op.add_entry_block()):
                    start, stop, step, dynamic_memref = func_op.arguments

                    indexing_maps = ArrayAttr.get([
                        AffineMapAttr.get(
                            AffineMap.get(
                                1, 0, [AffineExpr.get_dim(0)]
                            )
                        )
                    ])
                    iterator_types = ArrayAttr.get([Attribute.parse("#linalg.iterator_type<parallel>")])
                    
                    generic_op = linalg.GenericOp(
                        result_tensors=[],
                        inputs=[],
                        outputs=[dynamic_memref],
                        indexing_maps=indexing_maps,
                        iterator_types=iterator_types
                    )
                    body = generic_op.regions[0].blocks.append(
                        element_type
                    )
                    with InsertionPoint(body):
                        loop_index = linalg.IndexOp(0).result
                        index_i64 = arith.IndexCastOp(IntegerType.get_signless(64), loop_index).result
                        index_times_step = arith.MulFOp(arith.SIToFPOp(element_type, index_i64).result, step).result
                        result_value = arith.AddFOp(start, index_times_step).result
                        linalg.YieldOp([result_value])

                    func.ReturnOp([])

        return fn_name

    def gen_array_unary(self, module, num_axis, op, dtype):
        fn_name = self.gen_fn_name("unary")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * num_axis, element_type)
            func_type = FunctionType.get([memref_type, memref_type], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments

                generic_op = linalg.GenericOp(
                    result_tensors=[],
                    inputs=[input_memref],
                    outputs=[output_memref],
                    indexing_maps=[
                        AffineMap.get_identity(num_axis),
                        AffineMap.get_identity(num_axis)
                    ],
                    iterator_types=ArrayAttr.get([Attribute.parse("#linalg.iterator_type<parallel>")])
                )

                body = generic_op.regions[0].blocks.append(
                    element_type, element_type
                )
        
                with InsertionPoint(body):
                    linalg.YieldOp([op(body.arguments[0])])

                func.ReturnOp([])

        return fn_name

    def gen_array_binary(self, module, num_axis_lhs, num_axis_rhs, op, dtype):
        # TODO: Differently shaped arrays misbehave and needs handling/broadcast implementations
        fn_name = self.gen_fn_name("binary")
        
        if num_axis_lhs != num_axis_rhs:
            raise NotImplementedError("brodcasting for different dimensional arrays is not supported")

        num_axis_res = max(num_axis_lhs, num_axis_rhs)

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type_lhs = MemRefType.get([ShapedType.get_dynamic_size()] * num_axis_lhs, element_type)
            memref_type_rhs = MemRefType.get([ShapedType.get_dynamic_size()] * num_axis_rhs, element_type)
            memref_type_res = MemRefType.get([ShapedType.get_dynamic_size()] * num_axis_res, element_type)
            func_type = FunctionType.get([memref_type_lhs, memref_type_rhs, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref_lhs, input_memref_rhs, output_memref = func_op.arguments

                generic_op = linalg.GenericOp(
                    result_tensors=[],
                    inputs=[input_memref_lhs, input_memref_rhs],
                    outputs=[output_memref],
                    indexing_maps=[
                        AffineMap.get_identity(num_axis_res),
                        AffineMap.get_identity(num_axis_res),
                        AffineMap.get_identity(num_axis_res)
                    ],
                    iterator_types=ArrayAttr.get([Attribute.parse("#linalg.iterator_type<parallel>")]*2)
                )

                body = generic_op.regions[0].blocks.append(
                    element_type, element_type, element_type
                )
        
                with InsertionPoint(body):
                    linalg.YieldOp([op(body.arguments[0], body.arguments[1])])

                func.ReturnOp([])

        return fn_name

    def gen_array_reduce(self, module, num_axis, reduce_axes, op, dtype):
        fn_name = self.gen_fn_name("reduce")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type_inp = MemRefType.get([ShapedType.get_dynamic_size()] * num_axis, element_type)
            memref_type_out = MemRefType.get([ShapedType.get_dynamic_size()] * (num_axis - len(reduce_axes)), element_type)
            func_type = FunctionType.get([memref_type_inp, memref_type_out], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments

                reduce_op = linalg.ReduceOp(
                    result=[],
                    inputs=[input_memref],
                    inits=[output_memref],
                    dimensions=list(reduce_axes)
                )

                body = reduce_op.regions[0].blocks.append(
                    element_type, element_type
                )
        
                with InsertionPoint(body):
                    linalg.YieldOp([op(body.arguments[0], body.arguments[1])])

                func.ReturnOp([])

        return fn_name

    def gen_array_outer_product(self, module, dims, dtype):
        fn_name = self.gen_fn_name("outer_prod")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type_inp = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            memref_type_res = MemRefType.get([ShapedType.get_dynamic_size()] * (dims + 1), element_type)
            func_type = FunctionType.get([memref_type_inp, memref_type_inp, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref_lhs, input_memref_rhs, output_memref = func_op.arguments

                generic_op = linalg.GenericOp(
                    result_tensors=[],
                    inputs=[input_memref_lhs, input_memref_rhs],
                    outputs=[output_memref],
                    indexing_maps=ArrayAttr.get([
                        AffineMapAttr.get(AffineMap.get(dims + 1, 0, [AffineDimExpr.get(i) for i in range(dims)])),
                        AffineMapAttr.get(AffineMap.get(dims + 1, 0, [AffineDimExpr.get(i) for i in range(dims-1)] + [AffineDimExpr.get(dims)])), 
                        AffineMapAttr.get(AffineMap.get(dims + 1, 0, [AffineDimExpr.get(i) for i in range(dims + 1)]))
                    ]),
                    iterator_types=ArrayAttr.get([Attribute.parse("#linalg.iterator_type<parallel>")] * 2)
                )

                body = generic_op.regions[0].blocks.append(
                    element_type, element_type, element_type
                )
        
                with InsertionPoint(body):
                    linalg.YieldOp([arith.mulf(body.arguments[0], body.arguments[1])])

                func.ReturnOp([])

        return fn_name

    def gen_array_reshape(self, module, dims, reshape_tuple):
        fn_name = self.gen_fn_name("reshape")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            memref_type_res = MemRefType.get(list(reshape_tuple), element_type)
            func_type = FunctionType.get([memref_type, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                func.ReturnOp([memref.reshape(memref_type_res, func_op.arguments[0])])

        return fn_name

    def gen_array_broadcast(self, module, dims, new_shape, broadcast_along):
        fn_name = self.gen_fn_name("broadcast")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            memref_type_res = MemRefType.get(list(new_shape), element_type)
            func_type = FunctionType.get([memref_type, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments
                linalg.broadcast(
                    input_memref,
                    outs=[output_memref],
                    dimensions=broadcast_along
                )
                func.ReturnOp([])

        return fn_name

    def gen_array_reshape_runtime(self, module):
        fn_name = self.gen_fn_name("reshape_runtime")

        return fn_name

    def gen_array_index(self, module):
        fn_name = self.gen_fn_name("index")

        return fn_name

    def gen_array_stack(self, module):
        fn_name = self.gen_fn_name("stack")

        return fn_name

    def gen_array_concat(self, module):
        fn_name = self.gen_fn_name("concat")

        return fn_name

    def gen_array_expand_dims(self, module, dims, axes):
        fn_name = self.gen_fn_name("expand_dims")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            result_shape = [ShapedType.get_dynamic_size()] * (dims + len(axes))
            for i in axes:
                result_shape[i] = [arith.constant(IndexType.get(), 1)]

            memref_type_res = MemRefType.get(result_shape, element_type)
            func_type = FunctionType.get([memref_type, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                func.ReturnOp([memref.ExpandShapeOp(memref_type_res, func_op.arguments[0], [])])

        return fn_name

    def gen_array_argmax(self, module, dims, axis):
        fn_name = self.gen_fn_name("argmax")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type_inp = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            memref_type_aux = MemRefType.get([ShapedType.get_dynamic_size()] * (dims - 1), element_type)
            memref_type_res = MemRefType.get([ShapedType.get_dynamic_size()] * (dims - 1), index_type)
            func_type = FunctionType.get([memref_type_inp, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments
               
                output_axes = [i for i in range(dims)]
                output_axes.pop(axis)
                max_memref = memref.AllocOp(memref_type_aux, [memref.dim(output_memref, arith.constant(index_type, i)) for i in range(dims - 1)], symbolOperands=[])
                iterator_types = [Attribute.parse("#linalg.iterator_type<parallel>")] * dims
                iterator_types[axis] = Attribute.parse("#linalg.iterator_type<reduction>")
                generic_op = linalg.GenericOp(
                    result_tensors=[],
                    inputs=[input_memref],
                    outputs=[max_memref, output_memref],
                    indexing_maps=ArrayAttr.get([
                        AffineMapAttr.get(AffineMap.get(dims, 0, [AffineDimExpr.get(i) for i in range(dims)])),
                        AffineMapAttr.get(AffineMap.get(dims, 0, [AffineDimExpr.get(i) for i in output_axes])),
                        AffineMapAttr.get(AffineMap.get(dims, 0, [AffineDimExpr.get(i) for i in output_axes]))
                    ]),
                    iterator_types=ArrayAttr.get(iterator_types)
                )

                body = generic_op.regions[0].blocks.append(
                    element_type, element_type, index_type
                )
        
                with InsertionPoint(body):
                    input_val, max_val, arg_val = body.arguments

                    reduction_idx = linalg.IndexOp(axis)
                    is_greater = arith.CmpFOp(arith.CmpFPredicate.OGT, 
                                            input_val, max_val)
                    new_max = arith.SelectOp(is_greater, input_val, max_val)
                    new_arg = arith.SelectOp(is_greater, reduction_idx.result, arg_val)
                    linalg.YieldOp([new_max.result, new_arg.result])
                func.ReturnOp([])

        return fn_name

    def gen_array_matmul(self, module, dims, dtype):
        fn_name = self.gen_fn_name("matmul")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            func_type = FunctionType.get([memref_type, memref_type, memref_type], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref_lhs, input_memref_rhs, output_memref = func_op.arguments
                linalg.matmul(input_memref_lhs, input_memref_rhs, outs=[output_memref])
                func.ReturnOp([])

        return fn_name

    def gen_array_transpose(self, module, dims, dtype):
        fn_name = self.gen_fn_name("transpose")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            func_type = FunctionType.get([memref_type, memref_type], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments
                linalg.transpose(input_memref, outs=[output_memref], permutation=[(dims-1-i) for i in range(dims)])
                func.ReturnOp([])

        return fn_name

    def gen_array_fill_value(self, module, dims, dtype):
        fn_name = self.gen_fn_name("fill_value")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            func_type = FunctionType.get([element_type, memref_type], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                (value, input_memref) = func_op.arguments
                linalg.fill(value, outs=[input_memref])
                func.ReturnOp([])

        return fn_name

    def gen_array_triu(self, module, dims, dtype):
        fn_name = self.gen_fn_name("triu")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            func_type = FunctionType.get([memref_type, index_type,  memref_type], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, k_value, output_memref = func_op.arguments
                zero = arith.ConstantOp(element_type, 0.0)

                generic_op = linalg.GenericOp(
                    result_tensors=[],
                    inputs=[input_memref],
                    outputs=[output_memref],
                    indexing_maps=[
                        AffineMap.get_identity(dims),
                        AffineMap.get_identity(dims)
                    ],
                    iterator_types=ArrayAttr.get([Attribute.parse("#linalg.iterator_type<parallel>")] * dims)
                )

                body = generic_op.regions[0].blocks.append(
                    element_type, element_type
                )
        
                with InsertionPoint(body):
                    i = arith.IndexCastOp(IndexType.get(), linalg.IndexOp(dims - 2))
                    j = arith.IndexCastOp(IndexType.get(), linalg.IndexOp(dims - 1))
                    condition = arith.CmpIOp(arith.CmpIPredicate.sge, j, arith.AddIOp(i, k_value))
                    linalg.YieldOp([arith.SelectOp(condition, body.arguments[0], zero)])
                func.ReturnOp([])

        return fn_name


context = Context()
context.load_all_available_dialects()

with context:
    module = Module.create(loc=Location.unknown())

backend = Backend()

input_ndim = 3
# We can have dynamic shapes over here too, but will require adjustment to the broadcast op defined below
softmax_input_shape = (3, 4, 6)

arr_max_reduce = backend.gen_array_reduce(module, input_ndim, (input_ndim - 1,), arith.maximumf, None)
arr_sub = backend.gen_array_binary(module, input_ndim, input_ndim, arith.subf, None)
arr_exp = backend.gen_array_unary(module, input_ndim, math.exp, None)
arr_sum_reduce = backend.gen_array_reduce(module, input_ndim, (input_ndim - 1,), arith.addf, None)
arr_div = backend.gen_array_binary(module, input_ndim, input_ndim, arith.divf, None)
arr_broadcast = backend.gen_array_broadcast(module, input_ndim - 1, softmax_input_shape, broadcast_along=[2])


print("Generated MLIR:")
print(str(module))
print("\n" + "="*50 + "\n")

backend.run_passes(module)

print("After lowering to LLVM:")
print(str(module))
print("\n" + "="*50 + "\n")

# Compile the functions, this will be responsiblity of the backend
with InsertionPoint(module.body), Location.unknown():
    element_type = F64Type.get()
    memref_type = MemRefType.get(softmax_input_shape, element_type)
    memref_type_reduced = MemRefType.get(softmax_input_shape[:-1], element_type)

arr_max_reduce_jit = backend.jit_compile_extra(module, (memref_type,), (memref_type_reduced,), arr_max_reduce, is_ufunc=True)
arr_sub_jit = backend.jit_compile_extra(module, (memref_type, memref_type), (memref_type,), arr_sub, is_ufunc=True)
arr_exp_jit = backend.jit_compile_extra(module, (memref_type,), (memref_type,), arr_exp, is_ufunc=True)
arr_sum_reduce_jit = backend.jit_compile_extra(module, (memref_type,), (memref_type_reduced,), arr_sum_reduce, is_ufunc=True)
arr_div_jit = backend.jit_compile_extra(module, (memref_type, memref_type), (memref_type,), arr_div, is_ufunc=True)
arr_broadcast_jit = backend.jit_compile_extra(module, (memref_type_reduced,), (memref_type,), arr_broadcast, is_ufunc=True)

print("Function compiled successfully!")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_mlir(x):
    # We need placeholders to store the results because our current mlir
    # python bindings don't allow us to return a internally allocated memref 
    # so we have to pre-allocate the memref over here and pass that in as a 
    # placeholder for storing the results of each result.

    x_placeholder = np.zeros_like(x)
    x_placeholder_2 = np.zeros_like(x)
    x_placeholder_3 = np.zeros_like(x)
    x_placeholder_4 = np.zeros_like(x)
    x_placeholder_5 = np.zeros_like(x)
    reduced_placeholder = np.zeros(x.shape[:-1])
    reduced_placeholder_1 = np.zeros(x.shape[:-1])

    e_x = arr_exp_jit(arr_sub_jit(x, arr_broadcast_jit(arr_max_reduce_jit(x, reduced_placeholder), x_placeholder), x_placeholder_2), x_placeholder_3)
    return arr_div_jit(e_x, arr_broadcast_jit(arr_sum_reduce_jit(e_x, reduced_placeholder_1), x_placeholder_4), x_placeholder_5)

softmax_input = np.random.random(softmax_input_shape)
numpy_result = softmax(softmax_input)
mlir_result = softmax_mlir(softmax_input)

assert np.allclose(numpy_result, mlir_result)
print("Function executed succesfully. Result: ", mlir_result)

print("\n" + "="*60 + "\n")