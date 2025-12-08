# Documentación: Compilador Optimizado para Operaciones Tensoriales

## Resumen del Proyecto

Este proyecto extiende un compilador existente escrito en C (usando Flex y Bison) para optimizar operaciones tensoriales multidimensionales. El compilador analiza código tensorial, lo convierte en un grafo de operaciones, aplica optimizaciones para reducir accesos a memoria y costo computacional, y genera código CUDA altamente paralelo. Además, integra geometría computacional para representar espacios de iteración de tensores como regiones geométricas.

---

## Instrucciones para Probar la Funcionalidad

### Requisitos Previos

1. **Herramientas necesarias:**
   - `flex` (analizador léxico)
   - `bison` (generador de parser)
   - `gcc` o `clang` (compilador C)
   - `make` (herramienta de construcción)

2. **Instalación en macOS (incluyendo M1/M2):**
   ```bash
   # Instalar Homebrew si no lo tienes
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Instalar las herramientas necesarias
   brew install flex bison
   
   # Nota: macOS ya incluye clang (equivalente a gcc) y make
   # Si prefieres gcc específicamente:
   # brew install gcc
   ```

3. **Instalación en sistemas Unix/Linux:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install flex bison gcc make
   ```

### Compilación

1. Navegar al directorio del proyecto:
   ```bash
   cd "Compiler Design/Flex-Bison"
   ```

2. Compilar el proyecto:
   ```bash
   make
   ```

   Esto generará:
   - `compiler`: El ejecutable del compilador
   - `finalAssignment.tab.c` y `finalAssignment.tab.h`: Archivos generados por Bison
   - `lex.yy.c`: Archivo generado por Flex

### Ejecución

#### Prueba 1: Programa básico (sin tensores)
```bash
./compiler pruebaT51.txt
```

#### Prueba 2: Operaciones aritméticas básicas
```bash
./compiler pruebaT52.txt
```

#### Prueba 3: Operaciones tensoriales
```bash
# Opción 1: Usando archivo (recomendado)
./compiler test_tensor.txt

# Opción 2: Usando entrada directa
printf "program tensor_test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nend\n" | ./compiler
```

**IMPORTANTE - Formato de dimensiones:**
- Las dimensiones de los tensores deben escribirse **sin espacios después de las comas**
- ✅ Correcto: `tensor A[100,50];`
- ❌ Incorrecto: `tensor A[100, 50];` (causa errores de parsing)

Este comando debería:
1. Parsear las declaraciones de tensores
2. Construir el grafo de operaciones
3. Mostrar los espacios de iteración calculados
4. Aplicar optimizaciones
5. Generar el archivo `generated_kernels.cu` con código CUDA

### Verificación de Salida

Después de ejecutar `test_tensor.txt`, deberías ver:
- El árbol de ejecución por niveles
- La tabla de símbolos
- El grafo de operaciones tensoriales
- El reporte de optimización
- El archivo `generated_kernels.cu` con kernels CUDA generados

### Limpieza

Para limpiar archivos generados:
```bash
make clean
```

---

## Especificaciones: ¿Qué se Implementó?

### 1. Extensión del Analizador Léxico (Lexer)

**Archivo:** `finalAssignment.l`

**Nuevos tokens agregados:**
- `T_TENSOR`: Palabra clave para declarar tensores
- `T_MATMUL`: Operador `@matmul` para multiplicación de matrices
- `T_TRANSPOSE`: Operador `@transpose` para transposición
- `T_REDUCE`: Operador `@reduce` para reducción
- `T_RESHAPE`: Operador `@reshape` para cambio de forma

### 2. Extensión del Parser

**Archivo:** `finalAssignment.y`

**Nuevas reglas de gramática:**
- `tensor_decl`: Declaración de tensores con dimensiones
- `tensor_dims`: Lista de dimensiones de un tensor
- `tensor_op`: Operaciones tensoriales (matmul, add, mul, transpose, reduce)

**Integración:**
- Construcción automática del grafo de operaciones durante el parsing
- Conversión de tensores a espacios de iteración
- Generación de código CUDA al finalizar el parsing

### 3. Módulo de Representación de Grafo

**Archivos:** `tensor_graph.h`, `tensor_graph.c`

**Estructuras principales:**
- `TensorDim`: Información de dimensiones de un tensor
- `GraphNode`: Nodo del grafo representando una operación
- `TensorGraph`: Grafo completo de operaciones tensoriales

**Funcionalidades:**
- Creación y gestión del grafo de operaciones
- Cálculo de accesos a memoria por operación
- Cálculo de costo computacional por operación
- Visualización del grafo

### 4. Módulo de Optimización

**Archivos:** `optimizer.h`, `optimizer.c`

**Estrategias implementadas:**
- `OPT_FUSE_OPS`: Fusión de operaciones consecutivas
- `OPT_COMMON_SUBEXPR`: Eliminación de subexpresiones comunes
- `OPT_MEMORY_LAYOUT`: Optimización de layout de memoria

**Funcionalidades:**
- Análisis del grafo para identificar oportunidades de optimización
- Aplicación de múltiples estrategias en secuencia
- Cálculo de métricas de reducción (memoria y computación)
- Generación de reporte de optimización

### 5. Módulo de Generación de Código CUDA

**Archivos:** `cuda_gen.h`, `cuda_gen.c`

**Kernels generados:**
- `matmul_kernel`: Multiplicación de matrices paralela
- `add_kernel` / `mul_kernel`: Operaciones elemento a elemento
- `reduce_kernel`: Reducción paralela usando shared memory
- `transpose_kernel`: Transposición optimizada

**Características:**
- Generación automática de código CUDA desde el grafo
- Configuración de grid y block sizes
- Gestión de memoria de dispositivo
- Función main con lanzamiento de kernels

### 6. Módulo de Geometría Computacional

**Archivos:** `geometry.h`, `geometry.c`

**Estructuras:**
- `Point`: Punto en espacio n-dimensional
- `IterationSpace`: Hiper-rectángulo representando espacio de iteración

**Operaciones:**
- Conversión de tensores a espacios de iteración
- Intersección de espacios
- Unión de espacios
- Cálculo de volumen
- Verificación de pertenencia de puntos

---

## ¿Cómo se Implementó?

### Arquitectura General

El proyecto sigue una arquitectura modular:

```
┌─────────────────┐
│   Lexer (Flex)  │ → Tokens
└────────┬────────┘
         │
┌────────▼────────┐
│ Parser (Bison)  │ → AST + Tensor Graph
└────────┬────────┘
         │
    ┌────┴────┬──────────────┬──────────────┐
    │         │              │              │
┌───▼───┐ ┌───▼────┐ ┌───────▼────┐ ┌───────▼────┐
│ Graph │ │Optimizer│ │ CUDA Gen  │ │ Geometry  │
└───────┘ └─────────┘ └───────────┘ └────────────┘
```

### Flujo de Ejecución

1. **Análisis Léxico (Flex):**
   - El lexer tokeniza el código fuente
   - Reconoce palabras clave, operadores y literales
   - Genera tokens para el parser

2. **Análisis Sintáctico (Bison):**
   - El parser construye el AST tradicional
   - Simultáneamente construye el grafo de operaciones tensoriales
   - Para cada tensor declarado, crea un espacio de iteración

3. **Construcción del Grafo:**
   - Cada operación tensorial crea un nodo en el grafo
   - Las dependencias se representan como aristas
   - Se almacena información de formas (shapes) y tipos de operación

4. **Optimización:**
   - Se aplican múltiples estrategias al grafo
   - Se calculan métricas de mejora
   - Se genera un grafo optimizado

5. **Generación de Código CUDA:**
   - Se recorre el grafo optimizado
   - Se genera un kernel CUDA para cada tipo de operación
   - Se crea la función main con lanzamiento de kernels

### Detalles de Implementación

#### Construcción del Grafo

El grafo se construye incrementalmente durante el parsing:

```c
// Ejemplo: A @matmul B
GraphNode *in1 = add_graph_node(graph, OP_IDENTITY, "A", shape_A);
GraphNode *in2 = add_graph_node(graph, OP_IDENTITY, "B", shape_B);
GraphNode *op = add_graph_node(graph, OP_MATMUL, NULL, output_shape);
add_edge(in1, op);  // A → matmul
add_edge(in2, op);  // B → matmul
```

#### Optimización

Las optimizaciones se aplican como transformaciones del grafo:

- **Fusión:** Combina operaciones consecutivas para reducir accesos a memoria
- **Eliminación de subexpresiones:** Identifica y reutiliza cálculos duplicados
- **Layout de memoria:** Reordena operaciones para mejorar localidad

#### Generación CUDA

Para cada operación, se genera un kernel específico:

```cuda
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // ... cálculo paralelo
}
```

#### Geometría Computacional

Los tensores se convierten a espacios de iteración:

```c
// Tensor [100, 50] → IterationSpace [[0, 99], [0, 49]]
IterationSpace *space = tensor_to_iteration_space(tensor_dim);
```

Esto permite:
- Análisis de dependencias
- Optimización de loops
- Verificación de bounds

---

## ¿Por Qué se Implementó de Esta Manera?

### 1. Representación en Grafo

**Razón:** Los grafos son la estructura natural para representar dependencias entre operaciones tensoriales. Permiten:
- Identificar operaciones independientes (paralelizables)
- Detectar oportunidades de optimización
- Visualizar el flujo de datos

### 2. Optimizaciones Múltiples

**Razón:** Diferentes optimizaciones abordan diferentes aspectos:
- **Fusión:** Reduce overhead de lanzamiento de kernels
- **Eliminación de subexpresiones:** Evita cálculos redundantes
- **Layout de memoria:** Mejora cache locality

### 3. Generación de CUDA

**Razón:** CUDA permite paralelismo masivo en GPUs:
- Miles de threads ejecutándose simultáneamente
- Memoria compartida para reducciones eficientes
- Coalesced memory access patterns

### 4. Geometría Computacional

**Razón:** Los espacios de iteración como hiper-rectángulos permiten:
- Análisis matemático formal de dependencias
- Optimización de loops anidados
- Verificación de bounds en tiempo de compilación

### 5. Arquitectura Modular

**Razón:** Separación de responsabilidades:
- Facilita mantenimiento
- Permite reutilización
- Hace testing más sencillo

---

## Análisis de Complejidad

### Complejidad Temporal

| Operación | Complejidad | Justificación |
|-----------|-------------|---------------|
| Construcción del grafo | O(n) | n = número de operaciones, cada una se procesa una vez |
| Optimización (fusión) | O(n²) | Comparación de cada nodo con todos los anteriores |
| Optimización (CSE) | O(n²) | Búsqueda de duplicados requiere comparar todos los pares |
| Generación CUDA | O(n) | Recorrido lineal del grafo |
| Conversión a espacio de iteración | O(d) | d = número de dimensiones del tensor |

### Complejidad Espacial

| Estructura | Complejidad | Justificación |
|------------|-------------|---------------|
| Grafo de operaciones | O(n + e) | n nodos, e aristas |
| Espacios de iteración | O(d) | d dimensiones por tensor |
| Código CUDA generado | O(n × k) | n operaciones, k líneas promedio por kernel |

**Nota:** En la práctica, el número de operaciones (n) es típicamente pequeño comparado con el tamaño de los tensores, por lo que el overhead del compilador es mínimo.

---

## Limitaciones Actuales

### 1. **Análisis Estático Simplificado**
- Las formas (shapes) de los tensores se infieren de manera básica
- No hay verificación completa de compatibilidad de dimensiones
- No se detectan errores de shape en tiempo de compilación

### 2. **Optimizaciones Básicas**
- Las optimizaciones implementadas son versiones simplificadas
- No se implementan optimizaciones avanzadas como:
  - Loop tiling
  - Vectorización automática
  - Memory pooling
  - Kernel fusion avanzada

### 3. **Generación de CUDA**
- Los kernels generados son genéricos y no están completamente optimizados
- No se adaptan automáticamente al hardware específico
- No se optimiza el uso de shared memory más allá de reducciones
- No se generan múltiples variantes de kernels para diferentes tamaños

### 4. **Geometría Computacional**
- Solo se implementan hiper-rectángulos (no formas más complejas)
- No se analizan dependencias de datos entre iteraciones
- No se optimiza el orden de loops basado en geometría

### 5. **Gestión de Memoria**
- No se optimiza la asignación de memoria de dispositivo
- No se reutiliza memoria entre operaciones
- No se implementa memory pooling

### 6. **Soporte de Tipos**
- Solo se soportan tensores de punto flotante (float)
- No hay soporte para tipos mixtos o enteros

### 7. **Manejo de Errores**
- El manejo de errores es básico
- No hay mensajes de error detallados para operaciones tensoriales

### 8. **Testing**
- No hay suite de tests automatizados
- No se valida la corrección del código CUDA generado

---

## Cómo Mejorar el Proyecto

### Mejoras de Corto Plazo (1-2 semanas)

1. **Análisis de Shapes Mejorado**
   - Implementar propagación de shapes a través del grafo
   - Verificación de compatibilidad en tiempo de compilación
   - Mensajes de error más informativos

2. **Optimizaciones Adicionales**
   - Implementar loop fusion más agresivo
   - Agregar constant folding para operaciones con constantes
   - Implementar dead code elimination

3. **Mejora de Kernels CUDA**
   - Optimizar acceso a memoria (coalesced access)
   - Usar shared memory más efectivamente
   - Implementar múltiples variantes de kernels según tamaño

### Mejoras de Mediano Plazo (1-2 meses)

1. **Análisis de Dependencias**
   - Implementar análisis de dependencias de datos (dataflow analysis)
   - Detectar operaciones independientes para paralelización
   - Optimizar el orden de ejecución basado en dependencias

2. **Optimizaciones Avanzadas**
   - **Loop Tiling:** Dividir loops grandes en tiles para mejor cache locality
   - **Vectorización:** Usar instrucciones SIMD cuando sea posible
   - **Memory Pooling:** Reutilizar memoria de dispositivo entre operaciones
   - **Kernel Fusion:** Fusionar múltiples operaciones en un solo kernel

3. **Soporte de Tipos**
   - Agregar soporte para diferentes tipos de datos (int, double, etc.)
   - Implementar type inference
   - Soporte para tipos mixtos en operaciones

4. **Geometría Computacional Avanzada**
   - Implementar análisis de dependencias usando poliedros
   - Optimización de loops usando técnicas de polyhedral compilation
   - Detección de patrones de acceso a memoria

### Mejoras de Largo Plazo (3-6 meses)

1. **Backend Múltiple**
   - Soporte para OpenCL además de CUDA
   - Generación de código para CPU (OpenMP)
   - Soporte para otros aceleradores (TPU, etc.)

2. **Optimización Adaptativa**
   - Perfilado automático del código generado
   - Auto-tuning de parámetros (block size, etc.)
   - Selección automática de la mejor variante de kernel

3. **Compilación JIT (Just-In-Time)**
   - Compilación en tiempo de ejecución
   - Optimización basada en datos reales
   - Especialización de kernels según shapes reales

4. **Integración con Frameworks Existentes**
   - Interfaz con PyTorch/TensorFlow
   - Soporte para operaciones de deep learning
   - Integración con cuDNN/cuBLAS

5. **Análisis y Visualización**
   - Herramientas de visualización del grafo
   - Profiling integrado
   - Análisis de performance predictivo

6. **Testing y Validación**
   - Suite completa de tests unitarios
   - Tests de regresión
   - Validación automática de código CUDA generado
   - Benchmarks comparativos

### Mejoras Arquitectónicas

1. **IR (Intermediate Representation)**
   - Diseñar una IR más rica y expresiva
   - Separar análisis de optimización de generación de código
   - Facilitar múltiples backends

2. **Pase de Optimización Modular**
   - Sistema de pases de optimización pluggable
   - Orden configurable de optimizaciones
   - Métricas de calidad por pase

3. **Análisis Interprocedural**
   - Soporte para funciones/procedimientos
   - Inlining automático
   - Análisis de efectos secundarios

### Mejoras de Usabilidad

1. **Lenguaje de Entrada Mejorado**
   - Sintaxis más expresiva para operaciones tensoriales
   - Soporte para slicing y indexing
   - Operaciones de broadcasting

2. **Documentación y Ejemplos**
   - Tutoriales paso a paso
   - Ejemplos de casos de uso comunes
   - Guía de mejores prácticas

3. **Herramientas de Desarrollo**
   - Debugger para el grafo de operaciones
   - Visualizador de espacios de iteración
   - Profiler integrado

---

## Conclusión

Este proyecto proporciona una base sólida para un compilador optimizado de operaciones tensoriales. Las mejoras sugeridas permitirían escalar el proyecto desde un prototipo funcional hasta un sistema de producción capaz de competir con frameworks establecidos como TensorFlow o PyTorch en términos de optimización de código.

La arquitectura modular facilita la implementación incremental de estas mejoras, permitiendo que el proyecto evolucione de manera controlada y mantenible.

---

## Referencias y Recursos

- **Flex & Bison:** Documentación oficial de GNU
- **CUDA Programming Guide:** NVIDIA CUDA Toolkit Documentation
- **Polyhedral Compilation:** Técnicas avanzadas de optimización de loops
- **Tensor Compilers:** TVM, XLA, MLIR (inspiración para mejoras futuras)

---

**Fecha de creación:** 2024  
**Versión:** 1.0  
**Autor:** Equipo de Desarrollo

