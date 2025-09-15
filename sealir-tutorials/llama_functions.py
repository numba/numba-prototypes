
from mlir.ir import *
from mlir.dialects import arith, memref, scf, func, linalg, math, affine
import numpy as np
from mlir.passmanager import PassManager
import mlir.execution_engine as execution_engine
import mlir.runtime as runtime
import ctypes
from ctypes.util import find_library

import math as pymath

_DEBUG = True
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

    def gen_array_broadcast(self, module, in_shape, out_shape, axis):
        fn_name = self.gen_fn_name("broadcast")

        if axis == -1:
            axis = len(out_shape) - 1
        
        if out_shape[axis] % in_shape[axis]:
            raise ValueError("Array cannot be broadcasted to given shape along this axis")

        with module.context, InsertionPoint(module.body), Location.unknown():
            ndim = len(out_shape)
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get([ShapedType.get_dynamic_size()] * ndim, element_type)
            memref_type_out = MemRefType.get([ShapedType.get_dynamic_size()] * ndim, element_type)
            func_type = FunctionType.get([memref_type], [memref_type_out])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_arg, = func_op.arguments
                output_memref = memref.alloc(memref_type_out, [arith.constant(index_type, s) for s in out_shape], [])
                curr_offset = 0

                loop_times = out_shape[axis] // in_shape[axis]

                for i in range(loop_times):
                    offsets = [0] * (ndim)
                    offsets[axis] = i

                    strides = [1] * (ndim)
                    # strides[axis] = loop_times

                    out_strides = [ShapedType.get_dynamic_stride_or_offset()] * (ndim)
                    out_strides[-1] = 1

                    out_off = ShapedType.get_dynamic_stride_or_offset() if i !=0 else 0                    
                    out_layout = StridedLayoutAttr.get(out_off, out_strides)
                    memref_type_out_inner = MemRefType.get(in_shape, element_type, layout=out_layout)
                    subview = memref.SubViewOp(
                        memref_type_out_inner,
                        output_memref,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        static_offsets=DenseI64ArrayAttr.get(offsets),
                        static_sizes=DenseI64ArrayAttr.get(in_shape),
                        static_strides=DenseI64ArrayAttr.get(strides)
                    ).result

                    memref.copy(input_arg, subview)

                func.ReturnOp([output_memref])

        return fn_name

    def gen_array_index(self, module):
        fn_name = self.gen_fn_name("index")

        return fn_name

    def gen_array_stack(self, module, num_inputs, inp_shape, out_shape, axis):
        fn_name = self.gen_fn_name("stack")

        if axis == -1:
            axis = len(out_shape) - 1

        with module.context, InsertionPoint(module.body), Location.unknown():
            ndim = len(out_shape) - 1
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type = MemRefType.get(inp_shape, element_type)
            memref_type_out = MemRefType.get(out_shape, element_type)
            func_type = FunctionType.get([memref_type] * num_inputs, [memref_type_out])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_args = func_op.arguments
                output_memref = memref.alloc(memref_type_out, [], [])
                curr_offset = 0
                input_shape = list(inp_shape)
                input_shape.insert(axis, 1)

                strides = [1] * (ndim + 1)
                strides[axis] = num_inputs

                out_strides = [1] * (ndim + 1)
                for i in range(len(out_shape) - 2, -1, -1):
                    out_strides[i] = out_strides[i+1] * out_shape[i+1]
                out_strides[axis] *= num_inputs

                for input_arg in input_args:

                    offsets = [0] * (ndim + 1)
                    offsets[axis] = curr_offset
                    out_layout = StridedLayoutAttr.get(curr_offset, out_strides)
                    memref_type_out_inner = MemRefType.get(input_shape, element_type, layout=out_layout)
                    subview = memref.SubViewOp(
                        memref_type_out_inner,
                        output_memref,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        static_offsets=DenseI64ArrayAttr.get(offsets),
                        static_sizes=DenseI64ArrayAttr.get(input_shape),
                        static_strides=DenseI64ArrayAttr.get(strides)
                    ).result

                    re_shape = list(out_shape)
                    re_shape[axis] = 1
                    memref_type_in_inner = MemRefType.get(re_shape, element_type)

                    reassociation = Backend.build_mlir_reassociation(ndim, [axis-1])

                    input_arg_exp = memref.expand_shape(
                        memref_type_in_inner,
                        input_arg,
                        reassociation=reassociation,
                        output_shape=[],
                        static_output_shape=DenseI64ArrayAttr.get(re_shape)
                    )

                    memref.copy(input_arg_exp, subview)
                    curr_offset += 1

                func.ReturnOp([output_memref])

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

    def gen_array_expand_dims_shaped(self, module, in_shape, axes):
        fn_name = self.gen_fn_name("expand_dims")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get(in_shape, element_type)
            result_shape = list(in_shape)
            for i in sorted(axes):
                result_shape.insert(i, 1)
            dims = len(in_shape)

            memref_type_res = MemRefType.get(result_shape, element_type)
            func_type = FunctionType.get([memref_type], [memref_type_res])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                reassociation = Backend.build_mlir_reassociation(dims, axes)
                res = memref.expand_shape(memref_type_res,
                    func_op.arguments[0],
                    reassociation=reassociation,
                    output_shape=[], # [2, 3, 4]
                    static_output_shape=DenseI64ArrayAttr.get(result_shape) # [dyn, dyn, 1, dyn, 1]
                )
                func.ReturnOp([res])

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

    def gen_array_matmul_shaped(self, module, lhs_shape, rhs_shape, out_shape):
        fn_name = self.gen_fn_name("matmul")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type_lhs = MemRefType.get(lhs_shape, element_type)
            memref_type_rhs = MemRefType.get(rhs_shape, element_type)
            memref_type_out = MemRefType.get(out_shape, element_type)
            func_type = FunctionType.get([memref_type_lhs, memref_type_rhs], [memref_type_out])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            dims = len(out_shape)

            with InsertionPoint(func_op.add_entry_block()):
                a, b, = func_op.arguments

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

                out = memref.alloc(memref_type_out, [], [])
                zero = arith.ConstantOp(element_type, 0.0)
                linalg.fill(zero, outs=[out])

                generic_op = linalg.generic(
                    inputs=[a, b], outputs=[out], result_tensors=[],
                    indexing_maps=indexing_maps,
                    iterator_types=iterator_types
                )

                block = generic_op.regions[0].blocks.append(element_type, element_type, element_type)
                with InsertionPoint(block):
                    a_val, b_val, acc_val = block.arguments
                    mul = arith.mulf(a_val, b_val)
                    add = arith.addf(acc_val, mul)
                    linalg.yield_([add])

                func.ReturnOp([out])

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

    def gen_array_transpose_shaped(self, module, in_shape, permutation = None, dtype = None):
        fn_name = self.gen_fn_name("transpose")

        if permutation is None:
            permutation=[i for i in range(len(in_shape)).__reversed__()]

        out_shape = [in_shape[i] for i in permutation]

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type_in = MemRefType.get(in_shape, element_type)
            memref_type_out = MemRefType.get(out_shape, element_type)
            func_type = FunctionType.get([memref_type_in], [memref_type_out])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, = func_op.arguments
                output_memref = memref.alloc(memref_type_out, [], [])
                linalg.transpose(input_memref, outs=[output_memref], permutation=permutation)
                func.ReturnOp([output_memref])

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


    def gen_array_getitem_shaped(self, module, in_shape, out_shape, indices):
        fn_name = self.gen_fn_name("getitem")

        dims = len(in_shape)

        assert len(indices) <= dims, "Number of indices should be less than or equal to number of dimensions"
        if len(indices) < dims:
            indices = list(indices) + ([None] * (dims - len(indices)))

        offsets = [0] * dims
        strides = [1] * dims
        out_strides = [1] * dims

        for i in range(dims-1,0,-1):
            out_strides[i-1] = out_strides[i] * in_shape[i]

        modified_axes = []

        for axis, index in enumerate(indices):
            if index is None:
                continue
            elif isinstance(index, int):
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

                offsets[axis] = index_start
                modified_axes.append(axis)
            else:
                raise TypeError(f"Unknown index type {type(index)}")

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            index_type = IndexType.get()
            memref_type_in = MemRefType.get(in_shape, element_type)
            memref_type_out = MemRefType.get(out_shape, element_type, layout=StridedLayoutAttr.get(0, out_strides))
            func_type = FunctionType.get([memref_type_in], [memref_type_out])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                input_memref, = func_op.arguments
                subview = memref.SubViewOp(
                    memref_type_out,
                    input_memref,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    static_offsets=DenseI64ArrayAttr.get(offsets),
                    static_sizes=DenseI64ArrayAttr.get(out_shape),
                    static_strides=DenseI64ArrayAttr.get(strides)
                ).result

                func.ReturnOp([subview])

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

    def gen_array_setitem_shaped(self, module, in_shape, indices):
        fn_name = self.gen_fn_name("setitem")
        
        dims = len(in_shape)
        assert len(indices) <= dims, "Number of indices should be less than or equal to number of dimensions"

        if len(indices) < dims:
            indices = list(indices) + ([None] * (dims - len(indices)))

        val_shape = in_shape.copy()
        offsets = [0] * dims
        strides = [1] * dims

        modified_axes = []

        for axis, index in enumerate(indices):
            if index is None:
                continue
            elif isinstance(index, int):
                val_shape[index] = 1
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

                val_shape[axis] = (index_stop - index_start) // index_step
                offsets[axis] = index_start
                modified_axes.append(axis)
            else:
                raise TypeError(f"Unknown index type {type(index)}")

        out_strides = [1] * dims

        for i in range(dims-1,0,-1):
            out_strides[i-1] = out_strides[i] * in_shape[i]

        with module.context, InsertionPoint(module.body), Location.unknown():
            element_type = F64Type.get()
            memref_type = MemRefType.get(in_shape, element_type)
            memref_type_out = MemRefType.get(val_shape, element_type)
            memref_type_subview = MemRefType.get(val_shape, element_type, layout=StridedLayoutAttr.get(0, out_strides))
            func_type = FunctionType.get([memref_type, memref_type_out], [])

            func_op = func.FuncOp(fn_name, func_type)
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            with InsertionPoint(func_op.add_entry_block()):
                arr_memref, value_memref = func_op.arguments

                subview = memref.SubViewOp(
                    memref_type_subview,
                    arr_memref,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    static_offsets=DenseI64ArrayAttr.get(offsets),
                    static_sizes=DenseI64ArrayAttr.get(val_shape),
                    static_strides=DenseI64ArrayAttr.get(strides)
                ).result

                memref.copy(value_memref, subview)
                func.ReturnOp([])

        return fn_name

def main():

    context = Context()
    context.load_all_available_dialects()

    with context:
        module = Module.create(loc=Location.unknown())

    backend = Backend()

    batch_size, seq_len, n_heads, dims, cache_size = 1, 5, 6, 288, 256
    n_local_heads, head_dim = n_heads, dims // n_heads

    input_shape = (batch_size, seq_len, dims)
    softmax_input_shape = (batch_size, n_local_heads, seq_len, seq_len)

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

    print("Function compiled successfully!")

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def softmax_mlir(x):
        e_x = arr_exp(arr_sub(x, arr_broadcast(arr_max_reduce(x))))
        return arr_div(e_x, arr_broadcast(arr_sum_reduce(e_x)))

    # print("Testing Softmax")

    # Random input data
    softmax_input = np.random.random(softmax_input_shape)
    # NumPy execution
    numpy_result = softmax(softmax_input)
    # SealIR execution
    mlir_result = softmax_mlir(softmax_input)

    # Check Results
    assert np.allclose(numpy_result, mlir_result)
    print("Function executed and verified succesfully.")

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
