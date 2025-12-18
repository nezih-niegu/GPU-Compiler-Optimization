# ✅ Solución Final - Cómo Ejecutar Correctamente

## El Problema

En macOS con zsh, `echo` con múltiples líneas dentro de comillas dobles no funciona correctamente para pasar datos al compilador.

## ✅ Solución que SÍ Funciona

### Usa `printf` en lugar de `echo`:

```bash
printf "program tensor_test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nD := A + A;\nE := @transpose(A);\nF := @reduce(A, 0);\nend\n" | ./compiler
```

**Nota importante:** Usa `\n` para los saltos de línea, NO uses saltos de línea reales dentro de las comillas.

## 📝 Ejemplos Completos

### Ejemplo 1: Programa simple con un tensor
```bash
printf "program test\ntensor A[100,50];\nbegin\nend\n" | ./compiler
```

### Ejemplo 2: Programa completo con operaciones
```bash
printf "program tensor_test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nD := A + A;\nE := @transpose(A);\nF := @reduce(A, 0);\nend\n" | ./compiler
```

### Ejemplo 3: Programas básicos (sin tensores) - Funcionan con archivos
```bash
./compiler pruebaT51.txt
./compiler pruebaT52.txt
```

## 🔍 Qué Deberías Ver

Si todo funciona correctamente, verás:

1. ✅ Información de tensores: `Tensor A: Iteration Space [[0, 49], [0, 99]] Volume: 5000`
2. ✅ Árbol de ejecución por niveles
3. ✅ Tabla de símbolos
4. ✅ Grafo de operaciones tensoriales
5. ✅ Reporte de optimización (Memory reduction, Compute reduction)
6. ✅ Mensaje: "CUDA code generated in: generated_kernels.cu"
7. ✅ Archivo `generated_kernels.cu` creado

## ⚠️ Recordatorios Importantes

1. **NO pongas espacios después de las comas en dimensiones:**
   - ✅ Correcto: `tensor A[100,50];`
   - ❌ Incorrecto: `tensor A[100, 50];`

2. **Usa `printf` con `\n`, NO `echo` con saltos de línea reales**

3. **Todas las líneas deben terminar con punto y coma donde corresponda**

## 🚀 Comando Rápido para Copiar y Pegar

```bash
cd "Compiler Design/Flex-Bison"
printf "program tensor_test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nD := A + A;\nE := @transpose(A);\nF := @reduce(A, 0);\nend\n" | ./compiler
```

