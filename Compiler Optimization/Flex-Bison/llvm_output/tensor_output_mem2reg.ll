; ModuleID = 'llvm_output/tensor_output_instcombine.ll'
source_filename = "tensor_output.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx14.0.0"

; Function Attrs: nounwind
declare ptr @tensor_alloc(i32) #0

; Function Attrs: nounwind
declare void @tensor_free(ptr) #0

; Function Attrs: nounwind
declare void @tensor_matmul(ptr, ptr, ptr, i32, i32, i32) #0

; Function Attrs: nounwind
declare void @tensor_add(ptr, ptr, ptr, i32) #0

; Function Attrs: nounwind
declare void @tensor_mul(ptr, ptr, ptr, i32) #0

; Function Attrs: nounwind
declare void @tensor_transpose(ptr, ptr, i32, i32) #0

; Function Attrs: nounwind
declare float @tensor_reduce_sum(ptr, i32) #0

; Function Attrs: nounwind
declare void @tensor_print(ptr, i32) #0

; Function Attrs: nounwind
declare i32 @printf(ptr, ...) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

define i32 @main() {
entry:
  %0 = call ptr @tensor_alloc(i32 243237764)
  %1 = call ptr @tensor_alloc(i32 49)
  %2 = call ptr @tensor_alloc(i32 4964036)
  %3 = call ptr @tensor_alloc(i32 5000)
  %4 = call ptr @tensor_alloc(i32 10000)
  %5 = call ptr @tensor_alloc(i32 20000)
  %6 = call ptr @tensor_alloc(i32 20000)
  call void @tensor_matmul(ptr %6, ptr %3, ptr %4, i32 100, i32 50, i32 200)
  call void @tensor_free(ptr %0)
  call void @tensor_free(ptr %1)
  call void @tensor_free(ptr %2)
  call void @tensor_free(ptr %3)
  call void @tensor_free(ptr %4)
  call void @tensor_free(ptr %6)
  ret i32 0
}

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
