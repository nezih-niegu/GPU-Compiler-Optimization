# Cómo Ejecutar el Compilador - Guía Rápida

##  Forma que SÍ Funciona (Recomendada)

### Opción 1: Usando echo directamente en la terminal
```bash
echo "program tensor_test
tensor A[100,50];
tensor B[50,200];
begin
C := A @matmul B;
D := A + A;
E := @transpose(A);
F := @reduce(A, 0);
end" | ./compiler
```

### Opción 2: Usando printf (también funciona)
```bash
printf "program test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nend\n" | ./compiler
```

### Opción 3: Programas básicos (sin tensores) - Funciona con archivos
```bash
./compiler pruebaT51.txt
./compiler pruebaT52.txt
```

## Formato Importante

**Las dimensiones de tensores NO deben tener espacios después de las comas:**

- **Correcto:** `tensor A[100,50];`
- **Incorrecto:** `tensor A[100, 50];`

##  Ejemplo Completo

```bash
# Compilar el proyecto
make clean
make

# Ejecutar con operaciones tensoriales
printf "program test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nend\n" | ./compiler
```

##  Qué Deberías Ver

Si todo funciona correctamente, verás:

1. Información de tensores y espacios de iteración
2. Árbol de ejecución por niveles
3. Tabla de símbolos
4. Grafo de operaciones tensoriales
5. Reporte de optimización (reducción de memoria y computación)
6. Mensaje: "CUDA code generated in: generated_kernels.cu"
7. Archivo `generated_kernels.cu` creado

##  Problemas Conocidos

- **Archivos con operaciones tensoriales:** Actualmente hay un problema al leer archivos directamente con `./compiler archivo.txt` cuando contienen operaciones tensoriales. Usa `printf` o el script `ejecutar_tensor.sh` en su lugar.

- **Espacios en dimensiones:** No pongas espacios después de las comas en las dimensiones de tensores.

##  Tips

- Para programas simples sin tensores, los archivos funcionan perfectamente
- Para operaciones tensoriales, usa `printf` con `\n` explícitos
- Puedes modificar `ejecutar_tensor.sh` para tus propios programas

