; ModuleID = 'tensor_output.ll'
source_filename = "tensor_output.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx14.0.0"

; Function Attrs: nounwind
declare ptr @tensor_alloc(i32) local_unnamed_addr #0

; Function Attrs: nounwind
declare void @tensor_free(ptr) local_unnamed_addr #0

; Function Attrs: nounwind
declare void @tensor_matmul(ptr, ptr, ptr, i32, i32, i32) local_unnamed_addr #0

; Function Attrs: nounwind
declare void @tensor_transpose(ptr, ptr, i32, i32) local_unnamed_addr #0

; Function Attrs: nounwind
declare float @tensor_reduce_sum(ptr, i32) local_unnamed_addr #0

; Function Attrs: nounwind
define noundef i32 @main() local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @tensor_alloc(i32 5000)
  %1 = tail call ptr @tensor_alloc(i32 10000)
  %2 = tail call ptr @tensor_alloc(i32 20000)
  %3 = tail call ptr @tensor_alloc(i32 5000)
  %4 = tail call ptr @tensor_alloc(i32 5000)
  %5 = tail call ptr @tensor_alloc(i32 50)
  %6 = tail call ptr @tensor_alloc(i32 5000)
  %7 = tail call ptr @tensor_alloc(i32 10000)
  %8 = tail call ptr @tensor_alloc(i32 20000)
  %9 = tail call ptr @tensor_alloc(i32 5000)
  %10 = tail call ptr @tensor_alloc(i32 10000)
  %11 = tail call ptr @tensor_alloc(i32 20000)
  %12 = tail call ptr @tensor_alloc(i32 5000)
  %13 = tail call ptr @tensor_alloc(i32 5000)
  %14 = tail call ptr @tensor_alloc(i32 10000)
  %15 = tail call ptr @tensor_alloc(i32 20000)
  tail call void @tensor_matmul(ptr %15, ptr %6, ptr %7, i32 100, i32 50, i32 200)
  %16 = tail call ptr @tensor_alloc(i32 20000)
  tail call void @tensor_matmul(ptr %16, ptr %9, ptr %10, i32 100, i32 50, i32 200)
  %17 = tail call ptr @tensor_alloc(i32 5000)
  tail call void @tensor_transpose(ptr %17, ptr %12, i32 100, i32 50)
  %18 = tail call float @tensor_reduce_sum(ptr %14, i32 10000)
  tail call void @tensor_free(ptr %0)
  tail call void @tensor_free(ptr %1)
  tail call void @tensor_free(ptr %2)
  tail call void @tensor_free(ptr %3)
  tail call void @tensor_free(ptr %4)
  tail call void @tensor_free(ptr %5)
  tail call void @tensor_free(ptr %6)
  tail call void @tensor_free(ptr %7)
  tail call void @tensor_free(ptr %15)
  tail call void @tensor_free(ptr %9)
  tail call void @tensor_free(ptr %10)
  tail call void @tensor_free(ptr %16)
  tail call void @tensor_free(ptr %12)
  tail call void @tensor_free(ptr %17)
  tail call void @tensor_free(ptr %14)
  ret i32 0
}

attributes #0 = { nounwind }
