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
define noundef i32 @main() local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @tensor_alloc(i32 243237764)
  %1 = tail call ptr @tensor_alloc(i32 49)
  %2 = tail call ptr @tensor_alloc(i32 4964036)
  %3 = tail call ptr @tensor_alloc(i32 5000)
  %4 = tail call ptr @tensor_alloc(i32 10000)
  %5 = tail call ptr @tensor_alloc(i32 20000)
  %6 = tail call ptr @tensor_alloc(i32 20000)
  tail call void @tensor_matmul(ptr %6, ptr %3, ptr %4, i32 100, i32 50, i32 200)
  tail call void @tensor_free(ptr %0)
  tail call void @tensor_free(ptr %1)
  tail call void @tensor_free(ptr %2)
  tail call void @tensor_free(ptr %3)
  tail call void @tensor_free(ptr %4)
  tail call void @tensor_free(ptr %6)
  ret i32 0
}

attributes #0 = { nounwind }
