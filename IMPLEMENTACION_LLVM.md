# Implementación de LLVM en el Compilador de Tensores

## Ubicación de la Implementación

La integración con LLVM se implementa en cuatro archivos principales ubicados en `Compiler Optimization/Flex-Bison/`:

**1. `llvm_lowering.c` (344 líneas)** - Módulo principal de generación de LLVM IR que traduce el TensorGraph (representación interna) a formato textual LLVM IR (.ll). Implementa funciones para emitir el encabezado del módulo, declaraciones de funciones externas, y genera la función `main()` con todas las operaciones de tensores como llamadas a funciones externas (`@tensor_alloc`, `@tensor_matmul`, etc.). Utiliza nombres de registros SSA (%0, %1, %2...) y maneja casos especiales como operaciones de reducción que retornan escalares en lugar de punteros.

**2. `llvm_lowering.h` (60 líneas)** - Define la interfaz pública del módulo, incluyendo la estructura `LLVMLoweringContext` que mantiene el contexto de generación (archivo de salida, grafo, contadores de registros) y las declaraciones de todas las funciones de emisión de IR.

**3. `llvm_analyze.sh` (231 líneas)** - Script de análisis que ejecuta un pipeline de 9 etapas de optimización LLVM: verificación de IR, métricas base, DCE (eliminación de código muerto), CSE (eliminación de subexpresiones comunes), InstCombine (simplificación algebraica), Mem2Reg (promoción memoria-a-registro), optimización completa -O3, generación de CFG, y reporte de resumen. Extrae métricas como conteo de instrucciones, operaciones de memoria, y porcentaje de reducción de código.

**4. `llvm_config.sh` (44 líneas)** - Configura las rutas de las herramientas LLVM (`clang`, `opt`, `llc`, `llvm-dis`, `llvm-as`) apuntando a la instalación local en `~/Documents/GitHub/llvm-project/build/bin` y verifica que todas las herramientas existan antes de ejecutar.

La integración se completa en `finalAssignment.y` (línea ~124) donde se llama `lower_graph_to_llvm_ir()` después de la generación de CUDA, permitiendo que el compilador genere ambas salidas en paralelo: `generated_kernels.cu` (CUDA) y `tensor_output.ll` (LLVM IR para análisis).

## Funcionamiento del Pipeline

El flujo de trabajo completo funciona así: el parser construye el TensorGraph → `llvm_lowering.c` traduce cada nodo a instrucciones LLVM IR usando llamadas a funciones externas → se genera `tensor_output.ll` → `llvm_analyze.sh` ejecuta passes de optimización (`opt`) extrayendo métricas en cada etapa → se producen múltiples archivos `.ll` optimizados y reportes en `llvm_output/` mostrando reducción de instrucciones, operaciones de memoria, y análisis de flujo de control. Este enfoque permite validar las optimizaciones del compilador contra el estándar industrial LLVM sin requerir ejecución en GPU.
