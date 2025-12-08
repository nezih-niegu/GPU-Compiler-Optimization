# Compilador Optimizado para Operaciones Tensoriales

## Compilación Rápida

```bash
make
```

## Ejecución

```bash
./compiler test_tensor.txt
```

## Archivos Generados

- `compiler`: Ejecutable del compilador
- `generated_kernels.cu`: Código CUDA generado automáticamente

## Documentación Completa

### Documentos Principales:
- **`DOCUMENTACION.md`** - Documentación completa del proyecto
- **`ANALISIS_ALGORITMOS.md`** - Análisis detallado de algoritmos y complejidad
- **`LIMITACIONES.md`** - Lista completa de limitaciones identificadas
- **`RESUMEN_ANALISIS.md`** - Resumen ejecutivo con métricas y conclusiones

### Gráficas de Optimización:
- **`optimizacion_compilador.png`** - Gráficas visuales de las mejoras (alta resolución)
- **`optimizacion_compilador.pdf`** - Versión PDF de las gráficas

### Para generar las gráficas:
```bash
python3 generar_graficas.py
```

### Resultados de Optimización:
- ✅ 61% reducción de memoria
- ✅ 99% reducción de computación  
- ✅ 50% reducción de operaciones

## Estructura del Proyecto

- `finalAssignment.l`: Analizador léxico (Flex)
- `finalAssignment.y`: Parser y generador de código (Bison)
- `tensor_graph.c/h`: Representación de grafo de operaciones
- `optimizer.c/h`: Módulo de optimización
- `cuda_gen.c/h`: Generación de código CUDA
- `geometry.c/h`: Geometría computacional para espacios de iteración

