# Ficha del dataset: `claims_etiquetado`

## Origen

Dataset propio, construido para el caso de uso ClaimCheck IA (organismo certificador ficticio *CertiBajío, S.C.*). **No proviene de etiquetas reales de clientes.** Los claims se redactaron a partir de la estructura y la redacción típica de los marcos que regulan las declaraciones en etiquetas de alimentos:

- Reglamento (CE) 1924/2006: declaraciones nutricionales y de propiedades saludables.
- Reglamento (UE) 432/2012: lista de declaraciones de función autorizadas.
- Reglamento de Ejecución (UE) 828/2014: ausencia de gluten.
- Reglamento (UE) 1169/2011: información alimentaria facilitada al consumidor.
- FDA 21 CFR 101.13, 101.14, 101.54–101.62, 101.72–101.83 y 101.91.
- NOM-051-SCFI/SSA1-2010: etiquetado de alimentos y bebidas preenvasados (México).

| Fuente del registro | Claims | Descripción |
|---|---|---|
| `redaccion_manual` | 133 | Escritos a mano, con casos frontera entre clases |
| `plantilla` | 107 | Combinación de plantillas con listas de nutrimentos, funciones, sustancias y descriptores, muestreadas con semilla 42 |

**Procedimiento reproducible:** `python src/generar_dataset.py` escribe exactamente el mismo `claims_etiquetado.csv` (mismo orden, IDs y partición). `python src/exportar_excel.py` genera la versión `.xlsx`.

## Composición

| Clase | Total | dev | test | Casos frontera |
|---|---|---|---|---|
| NUTRIMENTAL | 70 | 21 | 49 | 10 |
| SALUD | 60 | 18 | 42 | 18 |
| AUSENCIA | 55 | 16 | 39 | 8 |
| MARKETING | 55 | 16 | 39 | 7 |
| **Total** | **240** | **71** | **169** | **43** |

## Estructura (un registro por claim)

| Campo | Tipo | Descripción |
|---|---|---|
| `id_claim` | texto | Identificador único (`CLM-0001` … `CLM-0240`) |
| `texto_claim` | texto | Claim tal como aparecería en el empaque; es la entrada del modelo |
| `categoria` | categórico | **Etiqueta objetivo**: `NUTRIMENTAL`, `SALUD`, `AUSENCIA` o `MARKETING` |
| `subtipo` | categórico | Subtipo fino para el análisis de errores (p. ej., `comparativa`, `reduccion_riesgo`, `aditivo`) |
| `referencia_normativa` | texto | Marco que inspira la redacción; es contexto, no un dictamen de cumplimiento |
| `dificultad` | categórico | `estandar` o `ambiguo` (caso frontera entre dos clases) |
| `origen_registro` | categórico | `redaccion_manual` o `plantilla` |
| `n_palabras` | entero | Número de palabras del claim |
| `particion` | categórico | `dev` (30 %, para elegir ajuste y umbral) o `test` (70 %, para las métricas finales) |

## Guía de etiquetado (se aplica en este orden)

1. **SALUD**: relaciona un alimento, nutrimento o componente con la salud, una función del cuerpo o la reducción de riesgo de enfermedad. Tiene prioridad aunque también mencione el contenido. *Fuente de calcio, que ayuda a mantener huesos fuertes* → SALUD.
2. **NUTRIMENTAL**: presencia, cantidad, ausencia o comparación de un nutrimento (energía, grasa, grasas saturadas o trans, azúcares, sodio o sal, fibra, proteína, vitaminas, minerales, colesterol). *Sin azúcares añadidos* → NUTRIMENTAL.
3. **AUSENCIA**: ausencia de algo que no es nutrimento: alérgeno o sustancia de intolerancia, aditivo, ingrediente o práctica de producción. *Sin edulcorantes artificiales* → AUSENCIA.
4. **MARKETING**: calidad, origen, proceso, sabor, textura, conveniencia, sostenibilidad o "natural", sin referencia a nutrimentos ni a la salud. *Ligero y crujiente* → MARKETING.

## Limitaciones

- Lo redactó el autor: es más "limpio" que las etiquetas reales (sin errores de OCR, abreviaturas ni claims combinados en una sola frase), así que las métricas son un techo optimista.
- Solo en español; el alcance del proyecto incluye etiquetas en inglés.
- Una sola etiqueta por claim; en la práctica, algunos claims combinan contenido y salud.
