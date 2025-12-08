# Instrucciones de Instalación y Uso - macOS (M1/M2)

## Instalación de Herramientas

### 1. Instalar Homebrew (si no lo tienes)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Instalar Flex y Bison

```bash
brew install flex bison
```

**Nota:** macOS ya incluye `clang` (compilador C) y `make`, así que no necesitas instalarlos por separado.

### 3. Verificar la instalación

```bash
flex --version
bison --version
gcc --version  # o clang --version
make --version
```

## Compilación del Proyecto

```bash
cd "Compiler Design/Flex-Bison"
make clean
make
```

Si encuentras errores, asegúrate de que todas las herramientas estén instaladas correctamente.

## Ejecución

### Ejemplo básico (funciona con archivos):

```bash
./compiler pruebaT51.txt
./compiler pruebaT52.txt
```

### Ejemplo con operaciones tensoriales (usar entrada directa):

**⚠️ NOTA:** Hay un problema conocido al leer archivos con operaciones tensoriales. Usa entrada directa:

```bash
# Opción 1: Usando printf (recomendado)
printf "program tensor_test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nD := A + A;\nE := @transpose(A);\nF := @reduce(A, 0);\nend\n" | ./compiler

# Opción 2: Usando echo con -e
echo -e "program test\ntensor A[100,50];\nbegin\nend" | ./compiler

# Opción 3: Usar el script incluido
./ejecutar_tensor.sh

# Opción 4: Usar heredoc (funciona bien)
./compiler << 'EOF'
program tensor_test
tensor A[100,50];
tensor B[50,200];
begin
C := A @matmul B;
end
EOF
```

## Formato Importante

**⚠️ IMPORTANTE:** Las dimensiones de los tensores deben escribirse **sin espacios después de las comas**:

- ✅ **Correcto:** `tensor A[100,50];`
- ❌ **Incorrecto:** `tensor A[100, 50];`

## Solución de Problemas

### Error: "command not found: flex" o "command not found: bison"
```bash
brew install flex bison
```

### Error: "invalid directive: %code"
Ya está corregido en el código. Si persiste, asegúrate de tener Bison 3.0 o superior:
```bash
bison --version
```

### Error de compilación
```bash
make clean
make
```

### Error al ejecutar: "syntax error"
- Verifica que las dimensiones de tensores no tengan espacios después de las comas
- Asegúrate de que todas las líneas terminen con punto y coma donde sea necesario
- Verifica que el archivo tenga el formato correcto

## Verificación de Funcionamiento

Después de ejecutar `./compiler test_tensor.txt`, deberías ver:

1. Información de tensores y espacios de iteración
2. Árbol de ejecución
3. Tabla de símbolos
4. Grafo de operaciones tensoriales
5. Reporte de optimización
6. Archivo `generated_kernels.cu` generado

Si todo esto aparece, el compilador está funcionando correctamente.

