
from mlir.ir import *
from mlir.dialects import arith, memref, scf, func, linalg, math, affine
import numpy as np
from mlir.passmanager import PassManager
import mlir.execution_engine as execution_engine
import mlir.runtime as runtime
import ctypes
from ctypes.util import find_library
from utils.llama3.llama3 import ModelArgs, Tokenizer, llama_init
from copy import deepcopy

import math as pymath

_DEBUG = False
_VERIFY = False
_FUZZ = False

class Backend:
    fn_counter = {}

    def run_passes(self, module):
        if _DEBUG:
            module.dump()
        pass_man = PassManager(context=module.context)

        if _DEBUG:
            module.context.enable_multithreading(False)
        if _DEBUG:
            # notebook may hang if ir_printing is enabled and and MLIR failed.
            pass_man.enable_ir_printing()

        pass_man.add("convert-linalg-to-loops")
        pass_man.add("expand-strided-metadata")
        pass_man.add("lower-affine")
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

    def jit_compile(self, module, func, input_shapes, output_shapes, dtype=None, shared_libs=None):
        in_types, out_types = [], []

        with InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            for i in input_shapes:
                if i is not None:
                    in_types.append(MemRefType.get(i, element_type))
                else:
                    in_types.append(element_type)

            for j in output_shapes:
                if j is not None:
                    out_types.append(MemRefType.get(j, element_type))
                else:
                    out_types.append(element_type)

        if shared_libs == None:
            shared_libs = []
        is_ufunc=True

        if len(output_shapes) == 0:
            is_ufunc = False
            out_types = (element_type,)

        fn_jitted = self.jit_compile_extra(module, in_types, out_types, func, is_ufunc=is_ufunc, shared_libs=shared_libs)

        def jit_func_wrap(*args):
            res = [np.zeros(x) for x in output_shapes]
            fn_jitted(*args, *res)
            if len(res) == 1:
                return res[0]
            else:
                return tuple(res)

        return jit_func_wrap

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
        reshape_tuple = list(reshape_tuple)
        for idx, i in enumerate(reshape_tuple):
            if i == -1:
                reshape_tuple[idx] = ShapedType.get_dynamic_size()

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            memref_type_res = MemRefType.get(list(reshape_tuple), element_type)
            shape_memref_type = MemRefType.get([len(reshape_tuple)], element_type=index_type)
            func_type = FunctionType.get([memref_type, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                shape_memref = memref.AllocOp(shape_memref_type, [], [])
                for idx, i in enumerate(reshape_tuple):
                    memref.store(arith.constant(index_type, i), shape_memref, [arith.constant(index_type, idx)])
                out = memref.reshape(memref_type_res, func_op.arguments[0], shape=shape_memref)
                memref.copy(out, func_op.arguments[1])
                func.ReturnOp([])

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

    def gen_array_index(self, module):
        fn_name = self.gen_fn_name("index")

        return fn_name

    def gen_array_stack(self, module, dims, num_inputs, axis, sizes_along_axis):
        fn_name = self.gen_fn_name("stack")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            func_type = FunctionType.get([memref_type] * (num_inputs + 1), [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_args = func_op.arguments[:-1]
                output_memref = func_op.arguments[-1]
                curr_offset = 0

                for input_arg, size_along_axis in zip(input_args, sizes_along_axis):

                    out_shape = [ShapedType.get_dynamic_size()] * dims
                    out_shape[axis] = size_along_axis

                    offsets = [0] * dims
                    offsets[axis] = curr_offset

                    strides = [1] * dims
                    strides[axis] = num_inputs

                    out_strides = [ShapedType.get_dynamic_stride_or_offset()] * dims
                    out_strides[axis] = num_inputs

                    out_layout = StridedLayoutAttr.get(curr_offset, out_strides)
                    memref_type_out = MemRefType.get(out_shape, element_type, layout=out_layout)
                    sizes = [memref.dim(func_op.arguments[0], arith.constant(index_type, i)) for i in range(dims)]
                    sizes.pop(axis)
                    subview = memref.SubViewOp(
                        memref_type_out,
                        output_memref,
                        offsets=[],
                        sizes=sizes,
                        strides=[],
                        static_offsets=DenseI64ArrayAttr.get(offsets),
                        static_sizes=DenseI64ArrayAttr.get(out_shape),
                        static_strides=DenseI64ArrayAttr.get(strides)
                    ).result

                    memref.copy(input_arg, subview)
                    curr_offset += 1

                func.ReturnOp([])

        return fn_name

    @classmethod
    def build_mlir_reassociation(cls, ndim, new_axes):
        new_axes = sorted([ax if ax >= 0 else ndim + len(new_axes) + ax for ax in new_axes])
        final_ndim = ndim + len(new_axes)
        is_new_dim = [False] * final_ndim
        for ax in new_axes:
            is_new_dim[ax] = True

        reassociation = []
        current_group = []
        for output_pos in range(final_ndim):
            current_group.append(output_pos)
            if not is_new_dim[output_pos]:
                reassociation.append(current_group)
                current_group = []

        # Convert to MLIR ArrayAttr
        attr_groups = []
        for group in reassociation:
            group_attrs = [IntegerAttr.get(IntegerType.get_signless(64), idx) for idx in group]
            attr_groups.append(ArrayAttr.get(group_attrs))

        return ArrayAttr.get(attr_groups)

    def gen_array_expand_dims(self, module, dims, axes):
        fn_name = self.gen_fn_name("expand_dims")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            result_shape = [ShapedType.get_dynamic_size()] * (dims + len(axes))
            for i in axes:
                result_shape[i] = 1

            memref_type_res = MemRefType.get(result_shape, element_type)
            func_type = FunctionType.get([memref_type, memref_type_res], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                reassociation = Backend.build_mlir_reassociation(dims, axes)
                output_shape = [memref.dim(func_op.arguments[0], arith.constant(index_type, i)) for i in range(dims)]
                res = memref.expand_shape(memref_type_res,
                    func_op.arguments[0],
                    reassociation=reassociation,
                    output_shape=output_shape, # [2, 3, 4]
                    static_output_shape=DenseI64ArrayAttr.get(result_shape) # [dyn, dyn, 1, dyn, 1]
                )
                memref.copy(res, func_op.arguments[1])
                func.ReturnOp([])

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
                a, b, c = func_op.arguments

                total_dims = dims + 1

                batch_exprs = [AffineExpr.get_dim(i) for i in range(dims - 2)]
                m_expr = AffineExpr.get_dim(dims - 2)  # M dimension
                n_expr = AffineExpr.get_dim(dims - 1)  # N dimension
                k_expr = AffineExpr.get_dim(dims)      # K dimension (reduction)

                map_a = AffineMap.get(total_dims, 0, batch_exprs + [m_expr, k_expr])
                map_b = AffineMap.get(total_dims, 0, batch_exprs + [k_expr, n_expr])
                map_c = AffineMap.get(total_dims, 0, batch_exprs + [m_expr, n_expr])

                indexing_maps = ArrayAttr.get([
                    AffineMapAttr.get(map_a),
                    AffineMapAttr.get(map_b),
                    AffineMapAttr.get(map_c)
                ])

                iterator_types = ArrayAttr.get([
                    StringAttr.get("parallel") for _ in range(dims - 1)  # batch + M + N
                ] + [StringAttr.get("reduction")])  # K

                generic_op = linalg.generic(
                    inputs=[a, b], outputs=[c], result_tensors=[],
                    indexing_maps=indexing_maps,
                    iterator_types=iterator_types
                )

                block = generic_op.regions[0].blocks.append(element_type, element_type, element_type)
                with InsertionPoint(block):
                    a_val, b_val, acc_val = block.arguments
                    mul = arith.mulf(a_val, b_val)
                    add = arith.addf(acc_val, mul)
                    linalg.yield_([add])

                func.ReturnOp([])

        return fn_name

    def gen_array_transpose(self, module, dims, permutation = None, dtype = None):
        fn_name = self.gen_fn_name("transpose")

        if permutation is None:
            permutation=[i for i in range(dims).__reversed__()]

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            func_type = FunctionType.get([memref_type, memref_type], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments
                linalg.transpose(input_memref, outs=[output_memref], permutation=permutation)
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

    def gen_array_take(self, module, dims, axis, slice_idx):
        fn_name = self.gen_fn_name("take")

        out_shape = [ShapedType.get_dynamic_size()] * dims
        out_shape[axis] = 1

        offsets = [0] * dims
        offsets[axis] = slice_idx

        strides = [1] * dims

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)

            out_strides = [ShapedType.get_dynamic_stride_or_offset()] * dims
            out_strides[axis] = 1
            out_layout = StridedLayoutAttr.get(slice_idx, out_strides)
            memref_type_out = MemRefType.get(out_shape, element_type, layout=out_layout)
            func_type = FunctionType.get([memref_type, memref_type_out], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments

                sizes = [memref.dim(func_op.arguments[0], arith.constant(index_type, i)) for i in range(dims)]
                sizes.pop(axis)
                subview = memref.SubViewOp(
                    memref_type_out,
                    input_memref,
                    offsets=[],
                    sizes=sizes,
                    strides=[],
                    static_offsets=DenseI64ArrayAttr.get(offsets),
                    static_sizes=DenseI64ArrayAttr.get(out_shape),
                    static_strides=DenseI64ArrayAttr.get(strides)
                ).result

                memref.copy(subview, output_memref)

                func.ReturnOp([])

        return fn_name

    def gen_array_getitem(self, module, dims, indices):
        fn_name = self.gen_fn_name("getitem")
        assert len(indices) <= dims, "Number of indices should be less than or equal to number of dimensions"
        if len(indices) < dims:
            indices = list(indices) + ([None] * (dims - len(indices)))

        out_shape = [ShapedType.get_dynamic_size()] * dims
        offsets = [0] * dims
        strides = [1] * dims
        out_strides = [ShapedType.get_dynamic_stride_or_offset()] * dims
        modified_axes = []

        for axis, index in enumerate(indices):
            if index is None:
                continue
            elif isinstance(index, int):
                out_shape[index] = 1
                offsets[axis] = index
                modified_axes.append(axis)
            elif isinstance(index, slice):
                index_start = index.start if index.start is not None else 0
                index_step = index.step if index.step is not None else 1
                index_stop = 0
                if index.stop is None:
                    raise ValueError("Slices with undefined stop not supported")
                else:
                    index_stop = index.stop

                out_shape[axis] = (index_stop - index_start) // index_step
                offsets[axis] = index_start
                modified_axes.append(axis)
            else:
                raise TypeError(f"Unknown index type {type(index)}")

        out_strides[-1] = 1

        first_offset = 0
        for offset in offsets:
            if offset != 0:
                first_offset = offset
                break

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            memref_type_out = MemRefType.get(out_shape, element_type, layout=StridedLayoutAttr.get(first_offset, out_strides))
            func_type = FunctionType.get([memref_type, memref_type_out], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, output_memref = func_op.arguments

                sizes = [memref.dim(func_op.arguments[0], arith.constant(index_type, i)) for i in range(dims)]
                for mod in modified_axes:
                    sizes.pop(mod)

                subview = memref.SubViewOp(
                    memref_type_out,
                    input_memref,
                    offsets=[],
                    sizes=sizes,
                    strides=[],
                    static_offsets=DenseI64ArrayAttr.get(offsets),
                    static_sizes=DenseI64ArrayAttr.get(out_shape),
                    static_strides=DenseI64ArrayAttr.get(strides)
                ).result

                memref.copy(subview, output_memref)

                func.ReturnOp([])

        return fn_name

    def gen_array_setitem(self, module, dims, indices):
        fn_name = self.gen_fn_name("setitem")
        assert len(indices) <= dims, "Number of indices should be less than or equal to number of dimensions"

        if len(indices) < dims:
            indices = list(indices) + ([None] * (dims - len(indices)))

        out_shape = [ShapedType.get_dynamic_size()] * dims
        offsets = [0] * dims
        strides = [1] * dims
        out_strides = [ShapedType.get_dynamic_stride_or_offset()] * dims
        modified_axes = []

        for axis, index in enumerate(indices):
            if index is None:
                continue
            elif isinstance(index, int):
                out_shape[index] = 1
                # out_strides[index] = 1
                offsets[axis] = index
                modified_axes.append(axis)
            elif isinstance(index, slice):
                index_start = index.start if index.start is not None else 0
                index_step = index.step if index.step is not None else 1
                index_stop = 0
                if index.stop is None:
                    raise ValueError("Slices with undefined stop not supported")
                else:
                    index_stop = index.stop

                out_shape[axis] = (index_stop - index_start) // index_step
                # out_strides[axis] = index_step
                offsets[axis] = index_start
                modified_axes.append(axis)
            else:
                raise TypeError(f"Unknown index type {type(index)}")
        out_strides[-1] = 1

        first_offset = 0
        for offset in offsets:
            if offset != 0:
                first_offset = offset
                break

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * dims, element_type)
            memref_type_out = MemRefType.get(out_shape, element_type, layout=StridedLayoutAttr.get(first_offset, out_strides))
            func_type = FunctionType.get([memref_type, memref_type], [element_type])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                arr_memref, value_memref = func_op.arguments

                sizes = [memref.dim(func_op.arguments[0], arith.constant(index_type, i)) for i in range(dims)]
                for mod in modified_axes:
                    sizes.pop(mod)

                subview = memref.SubViewOp(
                    memref_type_out,
                    arr_memref,
                    offsets=[],
                    sizes=sizes,
                    strides=[],
                    static_offsets=DenseI64ArrayAttr.get(offsets),
                    static_sizes=DenseI64ArrayAttr.get(out_shape),
                    static_strides=DenseI64ArrayAttr.get(strides)
                ).result

                memref.copy(value_memref, subview)
                func.ReturnOp([arith.constant(element_type, 0.0)])

        return fn_name

context = Context()
context.load_all_available_dialects()

with context:
    module = Module.create(loc=Location.unknown())

backend = Backend()

batch_size, seq_len, n_heads, dims, cache_size = 1, 5, 6, 288, 256
n_local_heads, head_dim = n_heads, dims // n_heads

input_shape = (batch_size, seq_len, dims)
softmax_input_shape = (batch_size, n_local_heads, seq_len, seq_len)

silu_dims = 768
norm_eps = 1e-06
weight_size = 32000

input_ndim = len(input_shape)
softmax_ndim = len(softmax_input_shape)

arr_max_reduce = backend.gen_array_reduce(module, softmax_ndim, (softmax_ndim - 1,), arith.maximumf, None)
arr_sub = backend.gen_array_binary(module, softmax_ndim, softmax_ndim, arith.subf, None)
arr_exp = backend.gen_array_unary(module, softmax_ndim, math.exp, None)
arr_sum_reduce = backend.gen_array_reduce(module, softmax_ndim, (softmax_ndim - 1,), arith.addf, None)
arr_div = backend.gen_array_binary(module, softmax_ndim, softmax_ndim, arith.divf, None)
arr_broadcast = backend.gen_array_broadcast(module, softmax_ndim - 1, softmax_input_shape, broadcast_along=[softmax_ndim - 1])

arr_transpose = backend.gen_array_transpose(module, 2)
arr_expand = backend.gen_array_expand_dims(module, 2, (0,))
arr_reshape_exp_1 = backend.gen_array_reshape(module, 2, (batch_size, dims, dims))
arr_matmul = backend.gen_array_matmul(module, 3, None)
arr_reshape = backend.gen_array_reshape(module, 3, (batch_size, seq_len, n_local_heads, head_dim))

arr_reshape_2 = backend.gen_array_reshape(module, 4, (batch_size, seq_len, n_local_heads, head_dim // 2, 2))
arr_take_0 = backend.gen_array_take(module, 5, 4, 0)
arr_take_1 = backend.gen_array_take(module, 5, 4, 1)
arr_reshape_3 = backend.gen_array_reshape(module, 5, (batch_size, seq_len, n_local_heads, head_dim // 2))
arr_expand_2 = backend.gen_array_expand_dims(module, 2, (0, 2))
arr_broadcast_2 = backend.gen_array_broadcast(module, 2, (batch_size, seq_len, n_local_heads, head_dim // 2), broadcast_along=[0, 2])
arr_add = backend.gen_array_binary(module, 4, 4, arith.addf, None)
arr_sub_1 = backend.gen_array_binary(module, 4, 4, arith.subf, None)
arr_mul = backend.gen_array_binary(module, 4, 4, arith.mulf, None)
arr_stack = backend.gen_array_stack(module, 4, 2, 3, (head_dim // 2, head_dim // 2))

arr_transpose_2 = backend.gen_array_transpose(module, 4, (0, 2, 1, 3))
arr_setitem = backend.gen_array_setitem(module, 4, (None, slice(0, seq_len)))
arr_getitem = backend.gen_array_getitem(module, 4, (None, slice(0, seq_len)))
arr_transpose_3 = backend.gen_array_transpose(module, 4, (0, 1, 3, 2), None)
arr_matmul_1 = backend.gen_array_matmul(module, 4, None)
arr_broadcast_3 = backend.gen_array_broadcast(module, 2, (batch_size, n_local_heads, seq_len, seq_len), broadcast_along=[0, 1])
arr_add_2 = backend.gen_array_binary(module, 4, 4, arith.addf, None)
arr_matmul_2 = backend.gen_array_matmul(module, 4, None)
arr_transpose_4 = backend.gen_array_transpose(module, 4, (0, 2, 1, 3), None)
arr_reshape_4 = backend.gen_array_reshape(module, 4, (batch_size, seq_len, dims))
arr_fill = backend.gen_array_fill_value(module, 4, None)
arr_sqrt = backend.gen_array_unary(module, 4, math.sqrt, None)
arr_div_2 = backend.gen_array_binary(module, 4, 4, arith.divf, None)

# SILU
silu_arr_neg = backend.gen_array_unary(module, 3, arith.negf, None)
silu_arr_exp = backend.gen_array_unary(module, 3, math.exp, None)
silu_arr_fill = backend.gen_array_fill_value(module, 3, None)
silu_arr_add = backend.gen_array_binary(module, 3, 3, arith.addf, None)
silu_arr_mul = backend.gen_array_binary(module, 3, 3, arith.mulf, None)
silu_arr_div = backend.gen_array_binary(module, 3, 3, arith.divf, None)

# Feed forward
ff_arr_transpose = backend.gen_array_transpose(module, 2)
ff_arr_transpose_2 = backend.gen_array_transpose(module, 2)
ff_arr_mul = backend.gen_array_binary(module, 3, 3, arith.mulf, None)
ff_arr_broadcast = backend.gen_array_broadcast(module, 2, (batch_size, dims, silu_dims), broadcast_along=[0])
ff_arr_broadcast_2 = backend.gen_array_broadcast(module, 2, (batch_size, silu_dims, dims), broadcast_along=[0])
ff_arr_matmul = backend.gen_array_matmul(module, 3, None)
ff_arr_matmul_2 = backend.gen_array_matmul(module, 3, None)

# RMSNorm
rms_arr_square = backend.gen_array_binary(module, 3, 3, arith.mulf, None)
rms_arr_sum_reduce = backend.gen_array_reduce(module, 3, (2,), arith.addf, None)
rms_arr_broadcast = backend.gen_array_broadcast(module, 2, (batch_size, seq_len, dims), broadcast_along=[2])
rms_arr_div = backend.gen_array_binary(module, 3, 3, arith.divf, None)
rms_arr_fill = backend.gen_array_fill_value(module, 3, None)
rms_arr_add = backend.gen_array_binary(module, 3, 3, arith.addf, None)
rms_arr_sqrt = backend.gen_array_unary(module, 3, math.sqrt, None)
rms_arr_mul = backend.gen_array_binary(module, 3, 3, arith.mulf, None)
rms_arr_broadcast_2 = backend.gen_array_broadcast(module, 1, (batch_size, seq_len, dims), broadcast_along=[0, 1])

# Transformer
tran_arr_add = backend.gen_array_binary(module, input_ndim, input_ndim, arith.addf, None)

# Llama forward
llm_arr_getitem = backend.gen_array_getitem(module, 2, (slice(0, seq_len),))
llm_arr_fill = backend.gen_array_fill_value(module, 2, None)
llm_arr_triu = backend.gen_array_triu(module, 2, None)
llm_arr_matmul = backend.gen_array_matmul(module, 3, None)
llm_arr_expand_2 = backend.gen_array_broadcast(module, 2, (batch_size, batch_size, dims), broadcast_along=[0])

# Llama generate


print("Generated MLIR:")
print(str(module))
print("\n" + "="*50 + "\n")

if _VERIFY:
    print("Running verification...")
    from utils import MLIRVerifier

    verifier = MLIRVerifier(module)

    verifier.verify(
        [
            "lower-affine",
            "convert-linalg-to-loops",
            "expand-strided-metadata",
            "lower-affine",
            "convert-scf-to-cf",
            "finalize-memref-to-llvm",
            "convert-math-to-libm",
            "convert-func-to-llvm",
            "convert-index-to-llvm",
            "reconcile-unrealized-casts"
        ],
        "outs"
    )

if __name__ == "__main__":


    if _FUZZ:
        from utils import PassFuzzer
        pass_manager = PassManager(context=module.context)

        # TODO: Figure out why this is required, the weird string parse issue?
        pass_manager.add("convert-linalg-to-loops")
        pass_manager.run(module.operation)

        passes = [
                "convert-linalg-to-loops",
                "expand-strided-metadata",
                "convert-scf-to-cf",
                "convert-math-to-libm",
                "reconcile-unrealized-casts",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "convert-index-to-llvm",
            ]
        print("Fuzzing MLIR Passes...")

        fuzzer = PassFuzzer(module, passes)

        sequence = fuzzer.find_effective_pass_sequence()

        print("Effective Pass Sequence:")
        print(sequence)

    backend.run_passes(module)

    print("After lowering to LLVM:")
    print(str(module))
    print("\n" + "="*50 + "\n")

    shared_libs = ("mlir_c_runner_utils",
                "mlir_runner_utils")

    arr_max_reduce = backend.jit_compile(module, arr_max_reduce, (softmax_input_shape,), (softmax_input_shape[:-1],))
    arr_sub = backend.jit_compile(module, arr_sub, (softmax_input_shape, softmax_input_shape), (softmax_input_shape,))
    arr_exp = backend.jit_compile(module, arr_exp, (softmax_input_shape,), (softmax_input_shape,))
    arr_sum_reduce = backend.jit_compile(module, arr_sum_reduce, (softmax_input_shape,), (softmax_input_shape[:-1],))
    arr_div = backend.jit_compile(module, arr_div, (softmax_input_shape, softmax_input_shape), (softmax_input_shape,))
    arr_broadcast = backend.jit_compile(module, arr_broadcast, (softmax_input_shape[:-1],), (softmax_input_shape,))

    arr_transpose = backend.jit_compile(module, arr_transpose, ((dims, dims),), ((dims, dims),))
    arr_expand = backend.jit_compile(module, arr_expand, ((dims, dims),), ((batch_size, dims, dims),))
    arr_reshape_exp_1 = backend.jit_compile(module, arr_reshape_exp_1, ((dims, dims),), ((batch_size, dims, dims),))
    arr_matmul = backend.jit_compile(module, arr_matmul, ((batch_size, seq_len, dims), (batch_size, dims, dims)), ((batch_size, seq_len, dims),))
    arr_reshape = backend.jit_compile(module, arr_reshape, ((batch_size, seq_len, dims),), ((batch_size, seq_len, n_local_heads, head_dim),))

    arr_reshape_2 = backend.jit_compile(module, arr_reshape_2, ((batch_size, seq_len, n_local_heads, head_dim),), ((batch_size, seq_len, n_local_heads, head_dim // 2, 2),))
    arr_take_0 = backend.jit_compile(module, arr_take_0, ((batch_size, seq_len, n_local_heads, head_dim // 2, 2),), ((batch_size, seq_len, n_local_heads, head_dim // 2, 1),), shared_libs=[find_library(x) for x in shared_libs])
    arr_take_1 = backend.jit_compile(module, arr_take_1, ((batch_size, seq_len, n_local_heads, head_dim // 2, 2),), ((batch_size, seq_len, n_local_heads, head_dim // 2, 1),), shared_libs=[find_library(x) for x in shared_libs])
    arr_reshape_3 = backend.jit_compile(module, arr_reshape_3, ((batch_size, seq_len, n_local_heads, head_dim // 2, 1),), ((batch_size, seq_len, n_local_heads, head_dim // 2),))
    arr_expand_2 = backend.jit_compile(module, arr_expand_2, ((dims, dims),), ((batch_size, dims, dims),))
    arr_broadcast_2 = backend.jit_compile(module, arr_broadcast_2, ((seq_len, head_dim // 2),), ((batch_size, seq_len, n_local_heads, head_dim // 2),))
    arr_mul = backend.jit_compile(module, arr_mul, ((batch_size, seq_len, n_local_heads, head_dim // 2), (batch_size, seq_len, n_local_heads, head_dim // 2)), ((batch_size, seq_len, n_local_heads, head_dim // 2),))
    arr_sub_1 = backend.jit_compile(module, arr_sub_1, ((batch_size, seq_len, n_local_heads, head_dim // 2), (batch_size, seq_len, n_local_heads, head_dim // 2)), ((batch_size, seq_len, n_local_heads, head_dim // 2),))
    arr_add = backend.jit_compile(module, arr_add, ((batch_size, seq_len, n_local_heads, head_dim // 2), (batch_size, seq_len, n_local_heads, head_dim // 2)), ((batch_size, seq_len, n_local_heads, head_dim // 2),))
    arr_stack = backend.jit_compile(module, arr_stack, ((batch_size, seq_len, n_local_heads, head_dim // 2), (batch_size, seq_len, n_local_heads, head_dim // 2)), ((batch_size, seq_len, n_local_heads, head_dim),), shared_libs=[find_library(x) for x in shared_libs])

    arr_transpose_2 = backend.jit_compile(module, arr_transpose_2, ((batch_size, seq_len, n_local_heads, head_dim),), ((batch_size, n_local_heads, seq_len, head_dim),))
    arr_setitem = backend.jit_compile(module, arr_setitem, ((batch_size, cache_size, n_heads, head_dim), (batch_size, seq_len, n_heads, head_dim)), (), shared_libs=[find_library(x) for x in shared_libs])
    arr_getitem = backend.jit_compile(module, arr_getitem, ((batch_size, cache_size, n_heads, head_dim),), ((batch_size, seq_len, n_heads, head_dim),), shared_libs=[find_library(x) for x in shared_libs])
    arr_transpose_3 = backend.jit_compile(module, arr_transpose_3, ((batch_size, n_local_heads, seq_len, head_dim),), ((batch_size, n_local_heads, head_dim, seq_len),))
    arr_matmul_1 = backend.jit_compile(module, arr_matmul_1, ((batch_size, n_local_heads, seq_len, head_dim), (batch_size, n_local_heads, head_dim, seq_len)), ((batch_size, n_local_heads, seq_len, seq_len),))
    arr_broadcast_3 = backend.jit_compile(module, arr_broadcast_3, ((seq_len, seq_len),), ((batch_size, n_local_heads, seq_len, seq_len),))
    arr_add_2 = backend.jit_compile(module, arr_add_2, ((batch_size, n_local_heads, seq_len, seq_len), (batch_size, n_local_heads, seq_len, seq_len)), ((batch_size, n_local_heads, seq_len, seq_len),))
    arr_matmul_2 = backend.jit_compile(module, arr_matmul_2, ((batch_size, n_local_heads, seq_len, seq_len), (batch_size, n_local_heads, seq_len, head_dim)), ((batch_size, n_local_heads, seq_len, head_dim),))
    arr_transpose_4 = backend.jit_compile(module, arr_transpose_4, ((batch_size, n_local_heads, seq_len, head_dim),), ((batch_size, seq_len, n_local_heads, head_dim),))
    arr_reshape_4 = backend.jit_compile(module, arr_reshape_4, ((batch_size, seq_len, n_local_heads, head_dim),), ((batch_size, seq_len, dims),))
    arr_fill = backend.jit_compile(module, arr_fill, (None,), ((batch_size, n_local_heads, seq_len, seq_len),), None)
    arr_sqrt = backend.jit_compile(module, arr_sqrt, ((batch_size, n_local_heads, seq_len, seq_len),), ((batch_size, n_local_heads, seq_len, seq_len),), None)
    arr_div_2 = backend.jit_compile(module, arr_div_2, ((batch_size, n_local_heads, seq_len, seq_len), (batch_size, n_local_heads, seq_len, seq_len),), ((batch_size, n_local_heads, seq_len, seq_len),))

    # SILU
    silu_arr_neg = backend.jit_compile(module, silu_arr_neg, ((batch_size, seq_len, silu_dims),), ((batch_size, seq_len, silu_dims),), None) 
    silu_arr_exp = backend.jit_compile(module, silu_arr_exp, ((batch_size, seq_len, silu_dims),), ((batch_size, seq_len, silu_dims),), None)
 
    silu_arr_fill = backend.jit_compile(module, silu_arr_fill, (None,), ((batch_size, seq_len, silu_dims),), None)
    silu_arr_add =  backend.jit_compile(module, silu_arr_add, ((batch_size, seq_len, silu_dims), (batch_size, seq_len, silu_dims)), ((batch_size, seq_len, silu_dims),))
    silu_arr_mul =  backend.jit_compile(module, silu_arr_mul, ((batch_size, seq_len, silu_dims), (batch_size, seq_len, silu_dims)), ((batch_size, seq_len, silu_dims),))
    silu_arr_div =  backend.jit_compile(module, silu_arr_div, ((batch_size, seq_len, silu_dims), (batch_size, seq_len, silu_dims)), ((batch_size, seq_len, silu_dims),))

    # Feed forward
    ff_arr_transpose = backend.jit_compile(module, ff_arr_transpose, ((dims, silu_dims),), ((silu_dims, dims),))
    ff_arr_transpose_2 = backend.jit_compile(module, ff_arr_transpose_2, ((silu_dims, dims),), ((dims, silu_dims),))
    ff_arr_mul = backend.jit_compile(module, ff_arr_mul, ((batch_size, seq_len, silu_dims), (batch_size, seq_len, silu_dims)), ((batch_size, seq_len, silu_dims),))
    ff_arr_broadcast = backend.jit_compile(module, ff_arr_broadcast, ((dims, silu_dims),), ((batch_size, dims, silu_dims),))
    ff_arr_broadcast_2 = backend.jit_compile(module, ff_arr_broadcast_2, ((silu_dims, dims),), ((batch_size, silu_dims, dims),))
    ff_arr_matmul = backend.jit_compile(module, ff_arr_matmul, ((batch_size, seq_len, dims), (batch_size, dims, silu_dims)), ((batch_size, seq_len, silu_dims),))
    ff_arr_matmul_2 = backend.jit_compile(module, ff_arr_matmul_2, ((batch_size, seq_len, silu_dims), (batch_size, silu_dims, dims)), ((batch_size, seq_len, dims),))

    # RMSNorm
    rms_arr_square = backend.jit_compile(module, rms_arr_square, ((batch_size, seq_len, dims),(batch_size, seq_len, dims)), ((batch_size, seq_len, dims),), None)
    rms_arr_sum_reduce = backend.jit_compile(module, rms_arr_sum_reduce, ((batch_size, seq_len, dims),), ((batch_size, seq_len),), None)
    rms_arr_broadcast = backend.jit_compile(module, rms_arr_broadcast, ((batch_size, seq_len),), ((batch_size, seq_len, dims),), None)
    rms_arr_div = backend.jit_compile(module, rms_arr_div, ((batch_size, seq_len, dims), (batch_size, seq_len, dims)), ((batch_size, seq_len, dims),), None)
    rms_arr_fill = backend.jit_compile(module, rms_arr_fill, (None,), ((batch_size, seq_len, dims),), None)
    rms_arr_add = backend.jit_compile(module, rms_arr_add, ((batch_size, seq_len, dims), (batch_size, seq_len, dims)), ((batch_size, seq_len, dims),), None)
    rms_arr_sqrt = backend.jit_compile(module, rms_arr_sqrt, ((batch_size, seq_len, dims),), ((batch_size, seq_len, dims),), None)
    rms_arr_mul = backend.jit_compile(module, rms_arr_mul, ((batch_size, seq_len, dims), (batch_size, seq_len, dims)), ((batch_size, seq_len, dims),))
    rms_arr_broadcast_2 = backend.jit_compile(module, rms_arr_broadcast_2, ((dims,),), ((batch_size, seq_len, dims),), None)

    # Transformer
    tran_arr_add = backend.jit_compile(module, tran_arr_add, ((batch_size, seq_len, dims), (batch_size, seq_len, dims)), ((batch_size, seq_len, dims),))
    
    # Llama forward
    llm_arr_getitem = backend.jit_compile(module, llm_arr_getitem, ((cache_size, head_dim // 2),), ((seq_len, head_dim // 2),), shared_libs=[find_library(x) for x in shared_libs])
    llm_arr_fill = backend.jit_compile(module, llm_arr_fill, (None,), ((seq_len, seq_len),), None)
    llm_arr_triu = backend.jit_compile(module, llm_arr_triu, ((seq_len, seq_len), None), ((seq_len, seq_len),), None)
    llm_arr_matmul = backend.jit_compile(module, llm_arr_matmul, ((batch_size, batch_size, dims), (batch_size, dims, weight_size)), ((batch_size, batch_size, weight_size),))
    llm_arr_expand_2 = backend.jit_compile(module, llm_arr_expand_2, ((dims, weight_size),), ((batch_size, dims, weight_size),))

    # Llama generate

    print("All functions compiled successfully!")
    print("\n" + "=" * 50 + "\n")

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def softmax_mlir(x):
        e_x = arr_exp(arr_sub(x, arr_broadcast(arr_max_reduce(x))))
        return arr_div(e_x, arr_broadcast(arr_sum_reduce(e_x)))

    print("Testing Softmax")

    # Random input data
    softmax_input = np.random.random(softmax_input_shape)
    # NumPy execution
    numpy_result = softmax(softmax_input)
    # SealIR execution
    mlir_result = softmax_mlir(softmax_input)

    # Check Results
    assert np.allclose(numpy_result, mlir_result)
    print("Function executed and verified succesfully.")
    print("\n" + "=" * 50 + "\n")

    def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
        xqri = xq.reshape(*xq.shape[:-1], -1, 2)
        xkri = xk.reshape(*xk.shape[:-1], -1, 2)

        xq_r = xqri[..., 0]
        xq_i = xqri[..., 1]
        xk_r = xkri[..., 0]
        xk_i = xkri[..., 1]
        freqs_cos = np.broadcast_to(np.expand_dims(freqs_cos, axis=(0, 2)), (1, 5, 6, 24))
        freqs_sin = np.broadcast_to(np.expand_dims(freqs_sin, axis=(0, 2)), (1, 5, 6, 24))
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        # Combine real and imaginary parts
        xq_out = np.stack([xq_out_r, xq_out_i], axis=-1).reshape(
            xq_out_r.shape[:-1] + (-1,)
        )
        xk_out = np.stack([xk_out_r, xk_out_i], axis=-1).reshape(
            xk_out_r.shape[:-1] + (-1,)
        )

        return xq_out, xk_out

    def apply_rotary_emb_mlir(xq, xk, freqs_cos, freqs_sin):
        xqri = arr_reshape_2(xq)
        xkri = arr_reshape_2(xk)

        xq_r = arr_reshape_3(arr_take_0(xqri))
        xq_i = arr_reshape_3(arr_take_1(xqri))
        xk_r = arr_reshape_3(arr_take_0(xkri))
        xk_i = arr_reshape_3(arr_take_1(xkri))

        freqs_cos = arr_broadcast_2(freqs_cos)
        freqs_sin = arr_broadcast_2(freqs_sin)

        xq_out_r = arr_sub_1(arr_mul(xq_r, freqs_cos), arr_mul(xq_i, freqs_sin))
        xq_out_i = arr_add(arr_mul(xq_r, freqs_sin), arr_mul(xq_i, freqs_cos))
        xk_out_r = arr_sub_1(arr_mul(xk_r, freqs_cos), arr_mul(xk_i, freqs_sin))
        xk_out_i = arr_add(arr_mul(xk_r, freqs_sin), arr_mul(xk_i, freqs_cos))

        # TODO: Combine real and imaginary parts
        xq_out = arr_stack(xq_out_r, xq_out_i)
        xk_out = arr_stack(xk_out_r, xk_out_i)

        return xq_out, xk_out

    print("Testing Apply Rotatory Embeddings")

    # Generate random data
    xq = np.random.random((batch_size, seq_len, n_local_heads, head_dim))
    xk = np.random.random((batch_size, seq_len, n_local_heads, head_dim))
    freqs_cos = np.random.random((seq_len, head_dim // 2))
    freqs_sin = np.random.random((seq_len, head_dim // 2))

    # NumPy execution
    numpy_result = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    # SealIR execution
    mlir_result = apply_rotary_emb_mlir(xq, xk, freqs_cos, freqs_sin)

    # Check Results
    assert np.allclose(numpy_result, mlir_result)
    print("Function executed and verified succesfully.")
    print("\n" + "=" * 50 + "\n")

    def attention(
        x, # shape = (1, 5, 288)
        start_pos, # 0
        mask, # shape = (5, 5)
        freqs_cos, # shape = (5, 24)
        freqs_sin, # shape = (5, 24)
        attn_weights, # 4 arrays of shape = (288, 288)
        cache_k, # shape = (1, 256, 6, 48)
        cache_v, # shape = (1, 256, 6, 48)
    ):
        q_weight, k_weight, v_weight, o_weight = [w.T for w in attn_weights]

        n_local_heads = n_heads # 6
        head_dim = dims // n_heads # 288/ 6 = 48

        batch_size, seq_len, _ = x.shape

        xq = x @ q_weight
        xk = x @ k_weight
        xv = x @ v_weight

        xq = xq.reshape(batch_size, seq_len, n_local_heads, head_dim)
        xk = xk.reshape(batch_size, seq_len, n_local_heads, head_dim)
        xv = xv.reshape(batch_size, seq_len, n_local_heads, head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
        ks = cache_k[:batch_size, : start_pos + seq_len]
        vs = cache_v[:batch_size, : start_pos + seq_len]

        xq = xq.transpose(0, 2, 1, 3)
        xk = ks.transpose(0, 2, 1, 3)
        xv = vs.transpose(0, 2, 1, 3)

        attention_scores = (xq @ xk.transpose(0, 1, 3, 2)) / pymath.sqrt(head_dim)
        if mask is not None:
            attention_scores = attention_scores + mask[None, None, :, :]
        attn = softmax(attention_scores)
        output = attn @ xv
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = output @ o_weight
        return output, cache_k, cache_v

    def attention_mlir(
        x, # shape = (1, 5, 288)
        start_pos, # 0
        mask, # shape = (5, 5)
        freqs_cos, # shape = (5, 24)
        freqs_sin, # shape = (5, 24)
        attn_weights, # 4 arrays of shape = (288, 288)
        cache_k, # shape = (1, 256, 6, 48)
        cache_v, # shape = (1, 256, 6, 48)
    ):
        q_weight, k_weight, v_weight, o_weight = [arr_transpose(w) for w in attn_weights]

        xq = arr_matmul(x, arr_reshape_exp_1(q_weight))
        xk = arr_matmul(x, arr_reshape_exp_1(k_weight))
        xv = arr_matmul(x, arr_reshape_exp_1(v_weight))

        xq = arr_reshape(xq)
        xk = arr_reshape(xk)
        xv = arr_reshape(xv)

        xq, xk = apply_rotary_emb_mlir(xq, xk, freqs_cos, freqs_sin)

        arr_setitem(cache_k, xk)
        arr_setitem(cache_v, xv)
        ks = arr_getitem(cache_k)
        vs = arr_getitem(cache_v)

        xq = arr_transpose_2(xq)
        xk = arr_transpose_2(ks)
        xv = arr_transpose_2(vs)

        attention_scores = arr_div_2(arr_matmul_1(xq, arr_transpose_3(xk)), arr_sqrt(arr_fill(head_dim)))

        if mask is not None:
            attention_scores = arr_add_2(attention_scores, arr_broadcast_3(mask))

        attn = softmax_mlir(attention_scores)

        output = arr_matmul_2(attn, xv)
        output = arr_reshape_4(arr_transpose_4(output))

        output = arr_matmul(output, arr_reshape_exp_1(o_weight))

        return output, cache_k, cache_v

    print("Testing Attention layer")

    attention_input = np.random.random(batch_size * seq_len * dims).reshape(batch_size, seq_len, dims)
    attention_weights = [np.random.random(dims*dims).reshape(dims, dims) for _ in range(4)]
    mask = np.random.random(seq_len * seq_len).reshape(seq_len, seq_len)
    freqs_cos = np.random.random(seq_len * head_dim // 2).reshape(seq_len, head_dim // 2)
    freqs_sin = np.random.random(seq_len * head_dim // 2).reshape(seq_len, head_dim // 2)
    cache_k = np.zeros((batch_size, cache_size, n_heads, head_dim))
    cache_v = np.zeros((batch_size, cache_size, n_heads, head_dim))
    cache_k_copy = cache_k.copy()
    cache_v_copy = cache_v.copy()

    numpy_result = attention(x=attention_input,
                            start_pos=0,
                            mask=mask,
                            freqs_cos=freqs_cos,
                            freqs_sin=freqs_sin,
                            attn_weights=attention_weights,
                            cache_k=cache_k,
                            cache_v=cache_v)
    mlir_result = attention_mlir(x=attention_input,
                            start_pos=0,
                            mask=mask,
                            freqs_cos=freqs_cos,
                            freqs_sin=freqs_sin,
                            attn_weights=attention_weights,
                            cache_k=cache_k_copy,
                            cache_v=cache_v_copy)

    for res_np, res_mlir in zip(numpy_result, mlir_result):
        assert np.allclose(res_np, res_mlir)

    print("Function executed succesfully.")

    print("\n" + "=" * 50 + "\n")

    def silu(x):
        result = x * (1 / (1 + np.exp(-x)))
        return result

    def silu_mlir(x):
        result = silu_arr_mul(x, silu_arr_div(silu_arr_fill(1.0), (silu_arr_add(silu_arr_fill(1.0), silu_arr_exp(silu_arr_neg(x))))))
        return result

    print("Testing SILU")

    silu_input = np.random.random(batch_size * seq_len * silu_dims).reshape(batch_size, seq_len, silu_dims)

    numpy_result = silu(x=silu_input)
    mlir_result = silu_mlir(x=silu_input)

    for res_np, res_mlir in zip(numpy_result, mlir_result):
        assert np.allclose(res_np, res_mlir)

    print("Function executed succesfully.")

    print("\n" + "=" * 50 + "\n")

    def feed_forward(x, up_weight, gate_weight, down_weight):
        swish = silu(x @ gate_weight.T)
        x_v = x @ up_weight.T
        x_ff = swish * x_v
        x_out = x_ff @ down_weight.T
        return x_out

    def feed_forward_mlir(x, up_weight, gate_weight, down_weight):
        swish = silu_mlir(ff_arr_matmul(x, ff_arr_broadcast(ff_arr_transpose_2(gate_weight))))
        x_v = ff_arr_matmul(x, ff_arr_broadcast(ff_arr_transpose_2(up_weight)))
        x_ff = ff_arr_mul(swish, x_v)
        x_out = ff_arr_matmul_2(x_ff, ff_arr_broadcast_2(ff_arr_transpose(down_weight)))
        return x_out

    print("Testing feed forward")

    x_input = np.random.random(batch_size * seq_len * dims).reshape(batch_size, seq_len, dims)
    up_weight = np.random.random(silu_dims * dims).reshape(silu_dims, dims)
    gate_weight = np.random.random(silu_dims * dims).reshape(silu_dims, dims)
    down_weight = np.random.random(dims * silu_dims).reshape(dims, silu_dims)

    numpy_result = feed_forward(x=x_input,
                                up_weight=up_weight,
                                gate_weight=gate_weight,
                                down_weight=down_weight)
    mlir_result = feed_forward_mlir(x=x_input,
                                up_weight=up_weight,
                                gate_weight=gate_weight,
                                down_weight=down_weight)

    for res_np, res_mlir in zip(numpy_result, mlir_result):
        assert np.allclose(res_np, res_mlir)

    print("Function executed succesfully.")

    print("\n" + "=" * 50 + "\n")

    def rmsnorm(x, weight, eps):
        z_float = np.mean(x**2, -1, keepdims=True) + eps
        z = x / np.sqrt(z_float)
        result = z * weight
        return result

    def rmsnorm_mlir(x, weight, eps):
        z_float = rms_arr_add(rms_arr_div(rms_arr_broadcast(rms_arr_sum_reduce(rms_arr_square(x, x))), rms_arr_fill(float(dims))), rms_arr_fill(float(eps)))
        z = rms_arr_div(x, rms_arr_sqrt(z_float))
        result = rms_arr_mul(z, rms_arr_broadcast_2(weight))
        return result

    print("Testing RMSNorm")

    x_input = np.random.random(batch_size * seq_len * dims).reshape(batch_size, seq_len, dims)
    weight = np.random.random(dims).reshape(dims)
    eps = 1e-06

    numpy_result = rmsnorm(x=x_input,
                           weight=weight,
                           eps=eps)
    mlir_result = rmsnorm_mlir(x=x_input,
                               weight=weight,
                               eps=eps)

    
    assert np.allclose(numpy_result, mlir_result)

    print("Function executed succesfully.")

    print("\n" + "=" * 50 + "\n")

    def transformer_block(
        x,
        start_pos,
        mask,
        freqs_cos,
        freqs_sin,
        block_weights,
        cache_k,
        cache_v,
    ):
        attn_weights, ff_weights, in_norm_weight, post_norm_weight = block_weights

        norm_x = rmsnorm(x, in_norm_weight, norm_eps)
        h1, cache_k, cache_v = attention(
            norm_x,
            start_pos,
            mask,
            freqs_cos,
            freqs_sin,
            attn_weights,
            cache_k,
            cache_v,
        )
        z = x + h1
        norm_z = rmsnorm(z, post_norm_weight, norm_eps)
        h2 = feed_forward(norm_z, *ff_weights)
        out = z + h2
        return out, cache_k, cache_v

    def transformer_block_mlir(
        x,
        start_pos,
        mask,
        freqs_cos,
        freqs_sin,
        block_weights,
        cache_k,
        cache_v,
    ):
        attn_weights, ff_weights, in_norm_weight, post_norm_weight = block_weights

        norm_x = rmsnorm_mlir(x, in_norm_weight, norm_eps)
        h1, cache_k, cache_v = attention_mlir(
            norm_x,
            start_pos,
            mask,
            freqs_cos,
            freqs_sin,
            attn_weights,
            cache_k,
            cache_v,
        )
        z = tran_arr_add(x, h1)
        norm_z = rmsnorm_mlir(z, post_norm_weight, norm_eps)
        h2 = feed_forward_mlir(norm_z, *ff_weights)
        out = tran_arr_add(z, h2)
        return out, cache_k, cache_v

    print("Testing Transformer Layer")

    x_input = np.random.random(batch_size * seq_len * dims).reshape(batch_size, seq_len, dims)
    start_pos = 0
    mask = np.random.random(seq_len * seq_len).reshape(seq_len, seq_len)
    freqs_cos = np.random.random(seq_len * head_dim // 2).reshape(seq_len, head_dim // 2)
    freqs_sin = np.random.random(seq_len * head_dim // 2).reshape(seq_len, head_dim // 2)
    block_weights = [
        [np.random.random(dims*dims).reshape(dims, dims) for _ in range(4)],
        [
            np.random.random(silu_dims * dims).reshape(silu_dims, dims),
            np.random.random(silu_dims * dims).reshape(silu_dims, dims),
            np.random.random(silu_dims * dims).reshape(dims, silu_dims),
        ],
        np.random.random(dims).reshape(dims),
        np.random.random(dims).reshape(dims),
    ]

    cache_k = np.random.random(batch_size * cache_size * n_heads * head_dim).reshape(batch_size, cache_size, n_heads, head_dim)
    cache_v = np.random.random(batch_size * cache_size * n_heads * head_dim).reshape(batch_size, cache_size, n_heads, head_dim)
    
    numpy_result = transformer_block(x=x_input,
                                     start_pos=start_pos,
                                     mask=mask,
                                     freqs_cos=freqs_cos,
                                     freqs_sin=freqs_sin,
                                     block_weights=block_weights,
                                     cache_k=cache_k,
                                     cache_v=cache_v)

    mlir_result = transformer_block_mlir(x=x_input,
                                     start_pos=start_pos,
                                     mask=mask,
                                     freqs_cos=freqs_cos,
                                     freqs_sin=freqs_sin,
                                     block_weights=block_weights,
                                     cache_k=cache_k,
                                     cache_v=cache_v)

    for res_np, res_mlir in zip(numpy_result, mlir_result):
        assert np.allclose(res_np, res_mlir)

    print("Function executed succesfully.")

    print("\n" + "=" * 50 + "\n")

    def llama_forward(model, input_ids, start_pos):
        args = model["args"]
        dtype = model["dtype"]

        _, seq_len = input_ids.shape
        h = model["tok_embedding"][input_ids]

        freqs_cos = model["freqs_cos"][start_pos : start_pos + seq_len]
        freqs_sin = model["freqs_sin"][start_pos : start_pos + seq_len]

        mask = None
        if seq_len > 1:
            mask = np.full((seq_len, seq_len), float("-inf"), dtype=dtype)
            mask = np.triu(mask, k=1)
            zeros_shape = (seq_len, start_pos)
            mask = np.concatenate([np.zeros(zeros_shape, dtype=dtype), mask], axis=1)

        caches_k = model["caches_k"]
        caches_v = model["caches_v"]

        for i, block in enumerate(model["layer_blocks"]):
            h, caches_k[i], caches_v[i] = transformer_block(
                h,
                start_pos,
                mask,
                freqs_cos,
                freqs_sin,
                block,
                caches_k[i],
                caches_v[i],
            )

        h = rmsnorm(h, model["norm_weight"], args.norm_eps)
        logit = h[:, [-1], :] @ model["lm_head_weight"]
        return logit

    def llama_generate(model, input_ids, max_new_tokens):
        batch_size, prompt_len = input_ids.shape
        current_len = prompt_len
        next_id = None  # Initialize next_id to avoid undefined variable error
        for i in range(max_new_tokens):
            current_pos = prompt_len + i
            if i == 0:
                current_input_ids = input_ids
                pos = 0
            else:
                current_input_ids = next_id
                pos = current_pos - 1
            logits = llama_forward(model, current_input_ids, pos)
            next_id = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int32)
            yield next_id
            current_len += 1
            if current_len >= model["args"].max_seq_len:
                break

    def llama_forward_mlir(model, input_ids, start_pos):
        args = model["args"]
        dtype = model["dtype"]
        tok_embedding = model["tok_embedding"]
        freqs_cos = model["freqs_cos"]
        freqs_sin = model["freqs_sin"]
        caches_k = model["caches_k"]
        caches_v = model["caches_v"]
        layer_blocks = model["layer_blocks"]
        norm_weight = model["norm_weight"]
        lm_head_weight = model["lm_head_weight"]

        _, seq_len = input_ids.shape
        h = tok_embedding[input_ids]

        freqs_cos = llm_arr_getitem(freqs_cos)
        freqs_sin = llm_arr_getitem(freqs_sin)

        mask = None
        if seq_len > 1:
            mask = llm_arr_fill(float("-inf"))
            mask = llm_arr_triu(mask, 1)
            # TODO: On first iteration start pos is zero so this evaliates to shape (5, 0)
            # which we can't really handle in MLIR as of now

            # zeros_shape = (seq_len, start_pos)
            # mask = np.concatenate([np.zeros(zeros_shape, dtype=dtype), mask], axis=1)

        for i, block in enumerate(layer_blocks):
            # TODO: Right now we index at compile time, this is a runtime index,
            # will need handling seperately, but mid-end should be able to take care of this
            # particular case depending upon how we implement
            h, caches_k[i], caches_v[i] = transformer_block_mlir(
                h,
                start_pos,
                mask,
                freqs_cos,
                freqs_sin,
                block,
                caches_k[i],
                caches_v[i],
            )

        h = rmsnorm_mlir(h, norm_weight, args.norm_eps)
        # TODO: We don't support this kind of indexing yet
        h_input = h[:, [-1], :]

        logit = llm_arr_matmul(h_input, llm_arr_expand_2(lm_head_weight))
        return logit

    def llama_generate_mlir(model, input_ids, max_new_tokens):
        batch_size, prompt_len = input_ids.shape
        current_len = prompt_len
        next_id = None  # Initialize next_id to avoid undefined variable error
        for i in range(max_new_tokens):
            current_pos = prompt_len + i
            if i == 0:
                current_input_ids = input_ids
                pos = 0
            else:
                current_input_ids = next_id
                pos = current_pos - 1
            logits = llama_forward_mlir(model, current_input_ids, pos)
            # TODO: We don't support this kind of indexing yet
            argmax_inp = logits[:, -1, :]
            # TODO: We will need argmax support
            next_id = np.argmax(argmax_inp, axis=-1, keepdims=True).astype(np.int32)
            yield next_id
            current_len += 1
            if current_len >= model["args"].max_seq_len:
                break


    print("Testing Llama Generate")

    args = ModelArgs()
    print(f"Using precision: {args.dtype}")
    tokenizer = Tokenizer("/home/kc611/Desktop/Workspaces/base/llama3.np/tokenizer.model.np")
    model = llama_init("/home/kc611/Desktop/Workspaces/base/llama3.np/stories15M.model.npz", args)

    prompt = "Once upon a time"

    print("Prompt", f"\n{prompt}")
    input_ids = np.array([tokenizer.encode(prompt)])

    model_mlir = deepcopy(model)

    for id_val, id_val_mlir in zip(llama_generate(model, input_ids, args.max_new_tokens), llama_generate_mlir(model_mlir, input_ids, args.max_new_tokens)):
        output_id = id_val[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print("Numpy Output Token: ", output_id, tokenizer.decode(output_id))

        output_id_mlir = id_val_mlir[0].tolist()
        if output_id_mlir[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print("MLIR Output Token: ", output_id_mlir, tokenizer.decode(output_id_mlir))
        break

    print("Function executed succesfully.")

    print("\n" + "=" * 50 + "\n")
