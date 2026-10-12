# Resultados

Esta carpeta contiene los resultados de la ejecución en Colab con GPU Tesla T4. Se genera en la sección 11 del notebook o con `python src/comparar_modelos.py`. En Colab se descarga `resultados_actividad4.zip`: descomprímelo aquí.

| Archivo | Contenido |
|---|---|
| `metricas_modelos.csv` | Tabla comparativa: métricas en test, latencia, recursos, umbral τ y cumplimiento de umbrales |
| `ajuste_minimo_dev.csv` | macro-F1 de cada configuración de plantilla y etiquetas en dev |
| `metricas_por_clase.csv` | precision, recall y F1 por clase y modelo |
| `predicciones_test.csv` | Predicción, confianza y probabilidad por clase de cada claim de test y cada modelo |
| `metadatos_hub.csv` | Licencia, idiomas, descargas, Spaces y commit de cada modelo (model cards) |
| `resumen.json` | Modelo recomendado, veredicto, umbrales, configuraciones y entorno |
| `entorno.json` | Versiones de librerías, GPU, CUDA y commits de los modelos |
| `figuras/` | Seis figuras: distribución, ajuste mínimo, precisión vs. latencia, recall por clase, matrices de confusión y cobertura vs. exactitud |
