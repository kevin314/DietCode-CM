; ModuleID = '__compute_module_out_elemental_kernel_module'
source_filename = "__compute_module_out_elemental_kernel_module"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@size_global_ptr = external global i64

; Function Attrs: uwtable
define ptr @out_kernel(ptr %0) #0 {
  %accum_address = alloca float, align 4
  %out.invar_address.reduction = alloca i64, align 8
  %out.invar_address.outer_reduction = alloca i64, align 8
  %out.invar_address.rhs.1 = alloca i64, align 8
  %out.invar_address.lhs.0 = alloca i64, align 8
  %out.bdot.invar_address = alloca i64, align 8
  %num_workgroups_gep = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %num_workgroups = load ptr, ptr %num_workgroups_gep, align 8
  %num_workgroups_x_gep = getelementptr inbounds %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 0
  %num_workgroups_y_gep = getelementptr inbounds %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 1
  %num_workgroups_z_gep = getelementptr inbounds %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 2
  %num_workgroups_x = load i64, ptr %num_workgroups_x_gep, align 4
  %num_workgroups_y = load i64, ptr %num_workgroups_y_gep, align 4
  %num_workgroups_z = load i64, ptr %num_workgroups_z_gep, align 4
  %workgroup_id_gep = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x_gep = getelementptr inbounds %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 0
  %workgroup_id_y_gep = getelementptr inbounds %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 1
  %workgroup_id_z_gep = getelementptr inbounds %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 2
  %workgroup_id_x = load i64, ptr %workgroup_id_x_gep, align 4
  %workgroup_id_y = load i64, ptr %workgroup_id_y_gep, align 4
  %workgroup_id_z = load i64, ptr %workgroup_id_z_gep, align 4
  %args_gep = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep1 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %args_gep3 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  store i64 0, ptr %out.bdot.invar_address, align 4
  br label %out.bdot.loop_header

out.bdot.loop_header:                             ; preds = %out.loop_exit.lhs.0, %1
  %out.bdot.indvar = load i64, ptr %out.bdot.invar_address, align 4
  %2 = icmp uge i64 %out.bdot.indvar, 8
  br i1 %2, label %out.bdot.loop_exit, label %out.bdot.loop_body

out.bdot.loop_body:                               ; preds = %out.bdot.loop_header
  %invar.inc = add nuw nsw i64 %out.bdot.indvar, 1
  store i64 %invar.inc, ptr %out.bdot.invar_address, align 4
  %3 = getelementptr inbounds [8 x [64 x [128 x float]]], ptr %arg0, i64 0, i64 %out.bdot.indvar, i64 0, i64 0
  %4 = getelementptr inbounds [8 x [128 x [32 x float]]], ptr %arg1, i64 0, i64 %out.bdot.indvar, i64 0, i64 0
  %5 = getelementptr inbounds [8 x [64 x [32 x float]]], ptr %arg2, i64 0, i64 %out.bdot.indvar, i64 0, i64 0
  %size_global = load i64, ptr @size_global_ptr, align 4
  %x_plus_c_minus_1 = add i64 %size_global, 7
  %num_tiles = udiv i64 %x_plus_c_minus_1, 8
  store i64 0, ptr %out.invar_address.lhs.0, align 4
  br label %out.loop_header.lhs.0

out.loop_header.lhs.0:                            ; preds = %out.loop_exit.rhs.1, %out.bdot.loop_body
  %out.indvar.lhs.0 = load i64, ptr %out.invar_address.lhs.0, align 4
  %6 = icmp uge i64 %out.indvar.lhs.0, 64
  br i1 %6, label %out.loop_exit.lhs.0, label %out.loop_body.lhs.0

out.loop_body.lhs.0:                              ; preds = %out.loop_header.lhs.0
  store i64 0, ptr %out.invar_address.rhs.1, align 4
  br label %out.loop_header.rhs.1

out.loop_header.rhs.1:                            ; preds = %out.loop_exit.outer_reduction, %out.loop_body.lhs.0
  %out.indvar.rhs.1 = load i64, ptr %out.invar_address.rhs.1, align 4
  %7 = icmp uge i64 %out.indvar.rhs.1, 32
  br i1 %7, label %out.loop_exit.rhs.1, label %out.loop_body.rhs.1

out.loop_body.rhs.1:                              ; preds = %out.loop_header.rhs.1
  store i64 0, ptr %out.invar_address.outer_reduction, align 4
  store float 0.000000e+00, ptr %accum_address, align 4
  br label %out.loop_header.outer_reduction

out.loop_header.outer_reduction:                  ; preds = %out.loop_exit.reduction, %out.loop_body.rhs.1
  %out.indvar.outer_reduction = load i64, ptr %out.invar_address.outer_reduction, align 4
  %8 = icmp uge i64 %out.indvar.outer_reduction, %num_tiles
  br i1 %8, label %out.loop_exit.outer_reduction, label %out.loop_body.outer_reduction

out.loop_body.outer_reduction:                    ; preds = %out.loop_header.outer_reduction
  store i64 0, ptr %out.invar_address.reduction, align 4
  br label %out.loop_header.reduction

out.loop_header.reduction:                        ; preds = %out.loop_body.reduction, %out.loop_body.outer_reduction
  %out.indvar.reduction = load i64, ptr %out.invar_address.reduction, align 4
  %outer_pos = mul i64 %out.indvar.outer_reduction, 8
  %reduction_idx = add i64 %outer_pos, %out.indvar.reduction
  %9 = icmp uge i64 %out.indvar.reduction, 8
  br i1 %9, label %out.loop_exit.reduction, label %out.loop_body.reduction

out.loop_body.reduction:                          ; preds = %out.loop_header.reduction
  %10 = getelementptr inbounds [64 x [128 x float]], ptr %3, i64 0, i64 %out.indvar.lhs.0, i64 %reduction_idx
  %11 = load float, ptr %10, align 4
  %12 = getelementptr inbounds [128 x [32 x float]], ptr %4, i64 0, i64 %reduction_idx, i64 %out.indvar.rhs.1
  %13 = load float, ptr %12, align 4
  %14 = load float, ptr %accum_address, align 4
  %15 = fmul float %11, %13
  %16 = fadd float %14, %15
  store float %16, ptr %accum_address, align 4
  %invar.inc8 = add nuw nsw i64 %out.indvar.reduction, 1
  store i64 %invar.inc8, ptr %out.invar_address.reduction, align 4
  br label %out.loop_header.reduction

out.loop_exit.reduction:                          ; preds = %out.loop_header.reduction
  %invar.inc7 = add nuw nsw i64 %out.indvar.outer_reduction, 1
  store i64 %invar.inc7, ptr %out.invar_address.outer_reduction, align 4
  br label %out.loop_header.outer_reduction, !llvm.loop !6

out.loop_exit.outer_reduction:                    ; preds = %out.loop_header.outer_reduction
  %17 = load float, ptr %accum_address, align 4
  %18 = getelementptr inbounds [64 x [32 x float]], ptr %5, i64 0, i64 %out.indvar.lhs.0, i64 %out.indvar.rhs.1
  store float %17, ptr %18, align 4
  %invar.inc6 = add nuw nsw i64 %out.indvar.rhs.1, 1
  store i64 %invar.inc6, ptr %out.invar_address.rhs.1, align 4
  br label %out.loop_header.rhs.1

out.loop_exit.rhs.1:                              ; preds = %out.loop_header.rhs.1
  %invar.inc5 = add nuw nsw i64 %out.indvar.lhs.0, 1
  store i64 %invar.inc5, ptr %out.invar_address.lhs.0, align 4
  br label %out.loop_header.lhs.0

out.loop_exit.lhs.0:                              ; preds = %out.loop_header.lhs.0
  br label %out.bdot.loop_header, !llvm.loop !8

out.bdot.loop_exit:                               ; preds = %out.bdot.loop_header
  br label %return

return:                                           ; preds = %out.bdot.loop_exit
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!xla_cpu_memory_region_name = !{!0}

!0 = !{!"xla_cpu_emitter__dot_kernel_emitter__hlo_opcode__dot"}
!1 = !{}
!2 = !{i64 262144}
!3 = !{i64 64}
!4 = !{i64 131072}
!5 = !{i64 65536}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.vectorize.enable", i1 false}
!8 = distinct !{!8, !9, !7}
!9 = !{!"llvm.loop.unroll.disable"}