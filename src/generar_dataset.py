"""
Genera el dataset `claims_etiquetado.csv` de forma 100 % reproducible.

Origen de los datos
-------------------
Dataset propio, construido para el caso de uso "ClaimCheck IA" (organismo certificador
ficticio CertiBajío, S.C.). Los claims NO provienen de etiquetas reales de clientes
(confidencialidad): se redactaron a partir de la estructura y la redacción típica que
establecen los marcos regulatorios de declaraciones en etiquetado de alimentos:

  * Reglamento (CE) 1924/2006 (declaraciones nutricionales y de propiedades saludables)
  * Reglamento (UE) 432/2012 (lista de declaraciones de función autorizadas)
  * Reglamento de Ejecución (UE) 828/2014 (ausencia de gluten)
  * Reglamento (UE) 1169/2011 (información alimentaria facilitada al consumidor)
  * FDA 21 CFR 101.13, 101.14, 101.54-101.62, 101.72-101.83 y 101.91
  * NOM-051-SCFI/SSA1-2010 (etiquetado de alimentos y bebidas preenvasados, México)

Hay dos fuentes de registros:
  1. `redaccion_manual`: claims escritos a mano, incluidos casos ambiguos o frontera.
  2. `plantilla`: claims generados combinando plantillas con listas de nutrimentos,
     funciones, sustancias y descriptores, muestreados con semilla fija.

Uso
---
    python src/generar_dataset.py                 # escribe data/claims_etiquetado.csv
    python src/generar_dataset.py --salida otro.csv --semilla 42

La misma semilla produce siempre el mismo archivo (mismo orden, mismos IDs, misma partición).
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

SEMILLA = 42
OBJETIVO_POR_CLASE = {"NUTRIMENTAL": 70, "SALUD": 60, "AUSENCIA": 55, "MARKETING": 55}
FRACCION_DEV = 0.30  # partición para elegir plantilla y umbral; el resto es test

REFERENCIAS = {
    "contenido_bajo_o_sin": "Reg. (CE) 1924/2006, Anexo; FDA 21 CFR 101.13 y 101.60-101.62; NOM-051-SCFI/SSA1-2010",
    "contenido_alto_o_fuente": "Reg. (CE) 1924/2006, Anexo; FDA 21 CFR 101.54; NOM-051-SCFI/SSA1-2010",
    "comparativa": "Reg. (CE) 1924/2006, Art. 9 y Anexo; FDA 21 CFR 101.13(j) y 101.56; NOM-051-SCFI/SSA1-2010",
    "cantidad_declarada": "FDA 21 CFR 101.13(i); NOM-051-SCFI/SSA1-2010",
    "funcion_nutrimento": "Reg. (CE) 1924/2006, Art. 13; Reg. (UE) 432/2012",
    "reduccion_riesgo": "Reg. (CE) 1924/2006, Art. 14; FDA 21 CFR 101.14 y 101.72-101.83",
    "beneficio_generico": "Reg. (CE) 1924/2006, Art. 10(3)",
    "gluten": "Reg. de Ejecución (UE) 828/2014; FDA 21 CFR 101.91",
    "alergeno_intolerancia": "Reg. (UE) 1169/2011, Art. 36 (información voluntaria)",
    "aditivo": "Reg. (UE) 1169/2011, Art. 7; NOM-051-SCFI/SSA1-2010",
    "ingrediente_o_produccion": "Reg. (UE) 1169/2011, Art. 7; NOM-051-SCFI/SSA1-2010",
    "marketing": "Reg. (UE) 1169/2011, Art. 7 (prácticas informativas leales); NOM-051-SCFI/SSA1-2010",
}

# ---------------------------------------------------------------------------
# 1. Claims redactados manualmente: (texto, subtipo, dificultad)
#    dificultad = "ambiguo" marca casos frontera entre clases.
# ---------------------------------------------------------------------------
MANUALES = {
    "NUTRIMENTAL": [
        ("Bajo en grasa", "contenido_bajo_o_sin", "estandar"),
        ("Muy bajo en sal", "contenido_bajo_o_sin", "estandar"),
        ("Sin azúcares añadidos", "contenido_bajo_o_sin", "ambiguo"),
        ("Sin grasa", "contenido_bajo_o_sin", "ambiguo"),
        ("0 % grasa", "contenido_bajo_o_sin", "estandar"),
        ("Cero azúcar", "contenido_bajo_o_sin", "ambiguo"),
        ("Libre de grasas trans", "contenido_bajo_o_sin", "ambiguo"),
        ("Sin sal añadida", "contenido_bajo_o_sin", "ambiguo"),
        ("Bajo en colesterol", "contenido_bajo_o_sin", "estandar"),
        ("Sin calorías", "contenido_bajo_o_sin", "estandar"),
        ("Menos de 5 kcal por lata", "contenido_bajo_o_sin", "estandar"),
        ("Bajo valor energético", "contenido_bajo_o_sin", "estandar"),
        ("Bajo en azúcares y fuente de fibra", "contenido_bajo_o_sin", "estandar"),
        ("Sin azúcar añadido; contiene azúcares naturalmente presentes", "contenido_bajo_o_sin", "ambiguo"),
        ("Alto en proteína: 20 g por porción", "contenido_alto_o_fuente", "estandar"),
        ("Excelente fuente de fibra", "contenido_alto_o_fuente", "estandar"),
        ("Fortificado con hierro y zinc", "contenido_alto_o_fuente", "estandar"),
        ("Adicionado con vitaminas A y D", "contenido_alto_o_fuente", "estandar"),
        ("Rico en grasas monoinsaturadas", "contenido_alto_o_fuente", "estandar"),
        ("Fuente natural de fibra", "contenido_alto_o_fuente", "ambiguo"),
        ("Integral, alto en fibra", "contenido_alto_o_fuente", "ambiguo"),
        ("Contiene 3 g de fibra por porción", "cantidad_declarada", "estandar"),
        ("Con 25 g de proteína por porción", "cantidad_declarada", "estandar"),
        ("Aporta el 30 % del valor diario recomendado de calcio", "cantidad_declarada", "estandar"),
        ("Una porción cubre el 50 % de tu requerimiento diario de vitamina C", "cantidad_declarada", "estandar"),
        ("Con 300 mg de omega-3 (EPA y DHA) por porción", "cantidad_declarada", "estandar"),
        ("Leche descremada: 0.5 % de grasa", "cantidad_declarada", "estandar"),
        ("Light: 30 % menos calorías que la versión regular", "comparativa", "estandar"),
        ("25 % menos sodio que nuestras papas originales", "comparativa", "estandar"),
        ("Reducido en azúcares: 30 % menos que el producto de referencia", "comparativa", "estandar"),
        ("Con la mitad de grasa que la mayonesa tradicional", "comparativa", "estandar"),
        ("Ahora con menos azúcar", "comparativa", "ambiguo"),
        ("Más proteína que un huevo", "comparativa", "estandar"),
        ("El doble de calcio que la leche entera", "comparativa", "estandar"),
        ("Light", "comparativa", "ambiguo"),
    ],
    "SALUD": [
        ("El calcio es necesario para el mantenimiento de los huesos en condiciones normales", "funcion_nutrimento", "estandar"),
        ("La vitamina C contribuye al funcionamiento normal del sistema inmunitario", "funcion_nutrimento", "estandar"),
        ("El hierro contribuye a disminuir el cansancio y la fatiga", "funcion_nutrimento", "estandar"),
        ("La fibra de salvado de trigo contribuye a acelerar el tránsito intestinal", "funcion_nutrimento", "estandar"),
        ("Los betaglucanos de avena contribuyen a mantener niveles normales de colesterol sanguíneo", "funcion_nutrimento", "estandar"),
        ("El DHA contribuye al funcionamiento normal del cerebro", "funcion_nutrimento", "estandar"),
        ("La vitamina D contribuye al mantenimiento de los huesos y dientes en condiciones normales", "funcion_nutrimento", "estandar"),
        ("El magnesio contribuye al funcionamiento normal de los músculos", "funcion_nutrimento", "estandar"),
        ("Las proteínas contribuyen a conservar la masa muscular", "funcion_nutrimento", "estandar"),
        ("El potasio contribuye al mantenimiento de la tensión arterial normal", "funcion_nutrimento", "estandar"),
        ("Los ácidos grasos EPA y DHA contribuyen al funcionamiento normal del corazón", "funcion_nutrimento", "estandar"),
        ("El ácido fólico contribuye al crecimiento de los tejidos maternos durante el embarazo", "funcion_nutrimento", "estandar"),
        ("Con vitamina A, que contribuye al mantenimiento de la visión normal", "funcion_nutrimento", "ambiguo"),
        ("Fuente de calcio, que ayuda a mantener huesos fuertes", "funcion_nutrimento", "ambiguo"),
        ("Rico en fibra que favorece el tránsito intestinal", "funcion_nutrimento", "ambiguo"),
        ("Gracias a su contenido de hierro, ayuda a reducir el cansancio", "funcion_nutrimento", "ambiguo"),
        ("Una dieta baja en grasas saturadas y colesterol puede reducir el riesgo de enfermedad cardiaca", "reduccion_riesgo", "estandar"),
        ("Una ingesta adecuada de calcio y vitamina D a lo largo de la vida puede reducir el riesgo de osteoporosis", "reduccion_riesgo", "estandar"),
        ("Los esteroles vegetales reducen el colesterol sanguíneo; el colesterol elevado es un factor de riesgo de cardiopatía coronaria", "reduccion_riesgo", "estandar"),
        ("Las dietas bajas en sodio pueden reducir el riesgo de hipertensión arterial", "reduccion_riesgo", "estandar"),
        ("Masticar chicle sin azúcar ayuda a neutralizar los ácidos de la placa dental, un factor de riesgo de caries", "reduccion_riesgo", "estandar"),
        ("Una dieta rica en frutas, verduras y granos enteros puede reducir el riesgo de algunos tipos de cáncer", "reduccion_riesgo", "estandar"),
        ("El consumo adecuado de ácido fólico antes y durante el embarazo puede reducir el riesgo de defectos del tubo neural", "reduccion_riesgo", "estandar"),
        ("La fibra soluble de la avena, como parte de una dieta baja en grasas saturadas, puede reducir el riesgo de enfermedad coronaria", "reduccion_riesgo", "estandar"),
        ("Ayuda a prevenir enfermedades del corazón", "reduccion_riesgo", "estandar"),
        ("Reduce el riesgo de diabetes tipo 2", "reduccion_riesgo", "estandar"),
        ("Ayuda a fortalecer tus defensas", "beneficio_generico", "estandar"),
        ("Favorece una digestión saludable", "beneficio_generico", "estandar"),
        ("Para un corazón sano", "beneficio_generico", "estandar"),
        ("Cuida tu salud intestinal con probióticos", "beneficio_generico", "estandar"),
        ("Te ayuda a mantener tu peso", "beneficio_generico", "estandar"),
        ("Bueno para tus huesos", "beneficio_generico", "estandar"),
        ("Mejora tu concentración", "beneficio_generico", "estandar"),
        ("Apoya tu energía durante todo el día", "beneficio_generico", "ambiguo"),
        ("Contribuye a tu bienestar", "beneficio_generico", "ambiguo"),
        ("Natural y saludable", "beneficio_generico", "ambiguo"),
        ("Hidratación para tu rendimiento deportivo", "beneficio_generico", "ambiguo"),
    ],
    "AUSENCIA": [
        ("Sin gluten", "gluten", "estandar"),
        ("Libre de gluten: menos de 20 mg/kg", "gluten", "estandar"),
        ("Apto para celíacos", "gluten", "ambiguo"),
        ("Sin lactosa", "alergeno_intolerancia", "estandar"),
        ("Deslactosada", "alergeno_intolerancia", "ambiguo"),
        ("Elaborado en una planta libre de cacahuate", "alergeno_intolerancia", "estandar"),
        ("No contiene huevo ni lácteos", "alergeno_intolerancia", "estandar"),
        ("Libre de frutos secos", "alergeno_intolerancia", "estandar"),
        ("Sin conservadores", "aditivo", "estandar"),
        ("Sin colorantes artificiales", "aditivo", "estandar"),
        ("Sin saborizantes artificiales", "aditivo", "estandar"),
        ("Sin edulcorantes artificiales", "aditivo", "ambiguo"),
        ("Sin glutamato monosódico añadido", "aditivo", "estandar"),
        ("Sin nitritos ni nitratos añadidos", "aditivo", "estandar"),
        ("Sin aditivos", "aditivo", "estandar"),
        ("Sin conservadores ni colorantes", "aditivo", "estandar"),
        ("0 % conservadores, 100 % sabor", "aditivo", "ambiguo"),
        ("No GMO", "ingrediente_o_produccion", "estandar"),
        ("Libre de transgénicos", "ingrediente_o_produccion", "estandar"),
        ("Sin aceite de palma", "ingrediente_o_produccion", "estandar"),
        ("Sin jarabe de maíz de alta fructosa", "ingrediente_o_produccion", "ambiguo"),
        ("Sin cafeína", "ingrediente_o_produccion", "estandar"),
        ("Descafeinado", "ingrediente_o_produccion", "ambiguo"),
        ("Sin hormonas añadidas", "ingrediente_o_produccion", "estandar"),
        ("Pollo criado sin antibióticos", "ingrediente_o_produccion", "ambiguo"),
        ("Sin ingredientes de origen animal", "ingrediente_o_produccion", "ambiguo"),
    ],
    "MARKETING": [
        ("Receta tradicional de la abuela", "marketing", "estandar"),
        ("Elaboración artesanal en pequeños lotes", "marketing", "estandar"),
        ("Hecho a mano", "marketing", "estandar"),
        ("Horneado todos los días", "marketing", "estandar"),
        ("Tostado lentamente para un sabor más profundo", "marketing", "estandar"),
        ("Madurado 12 meses", "marketing", "estandar"),
        ("Desde 1952", "marketing", "estandar"),
        ("Hecho en México", "marketing", "estandar"),
        ("Café de altura de Chiapas", "marketing", "estandar"),
        ("Mango Ataulfo del Soconusco", "marketing", "estandar"),
        ("Con vainilla de Papantla, Veracruz", "marketing", "estandar"),
        ("Elaborado con leche de vacas de pastoreo de Los Altos de Jalisco", "marketing", "estandar"),
        ("Del campo a tu mesa", "marketing", "estandar"),
        ("Sabor intenso", "marketing", "estandar"),
        ("Textura extra cremosa", "marketing", "estandar"),
        ("Irresistiblemente crujiente", "marketing", "estandar"),
        ("El auténtico sabor de siempre", "marketing", "estandar"),
        ("Nuevo sabor mejorado", "marketing", "estandar"),
        ("Ligero y crujiente", "marketing", "ambiguo"),
        ("Ingredientes seleccionados", "marketing", "estandar"),
        ("Calidad premium", "marketing", "estandar"),
        ("Edición limitada", "marketing", "estandar"),
        ("El favorito de las familias mexicanas", "marketing", "estandar"),
        ("Calidad de exportación", "marketing", "estandar"),
        ("Empaque 100 % reciclable", "marketing", "estandar"),
        ("Cultivado de forma sustentable", "marketing", "estandar"),
        ("Comercio justo", "marketing", "estandar"),
        ("Orgánico certificado", "marketing", "ambiguo"),
        ("100 % natural", "marketing", "ambiguo"),
        ("Hecho con ingredientes naturales", "marketing", "ambiguo"),
        ("Puro jugo de naranja", "marketing", "ambiguo"),
        ("Hecho con fruta real", "marketing", "ambiguo"),
        ("Energía natural para tu día", "marketing", "ambiguo"),
        ("Ideal para compartir en familia", "marketing", "estandar"),
        ("Listo en 3 minutos", "marketing", "estandar"),
    ],
}

# ---------------------------------------------------------------------------
# 2. Plantillas: cada función devuelve la lista completa de candidatos
#    (texto, subtipo, dificultad); después se muestrean con semilla fija.
# ---------------------------------------------------------------------------
# (nombre, género, plural) para concordar adjetivos: "Galletas bajas en grasa", "Leche adicionada..."
PRODUCTOS = [("Yogur", "m", False), ("Cereal", "m", False), ("Galletas", "f", True),
             ("Bebida de avena", "f", False), ("Pan de caja", "m", False), ("Leche", "f", False),
             ("Barra de granola", "f", False), ("Queso panela", "m", False),
             ("Salsa de tomate", "f", False), ("Atún en agua", "m", False)]


FRASES_CONTENIDO = {
    "bajo en grasa": ("bajo", "en grasa", "contenido_bajo_o_sin"),
    "bajo en sodio": ("bajo", "en sodio", "contenido_bajo_o_sin"),
    "alto en proteína": ("alto", "en proteína", "contenido_alto_o_fuente"),
    "adicionado con vitamina D": ("adicionado", "con vitamina D", "contenido_alto_o_fuente"),
}
PLAUSIBLES = {
    "Yogur": {"contenido": ["bajo en grasa", "alto en proteína", "adicionado con vitamina D"], "comparativa": ["azúcar", "grasa"]},
    "Cereal": {"contenido": ["fuente de fibra", "adicionado con vitamina D"], "comparativa": ["azúcar", "sodio"]},
    "Galletas": {"contenido": ["fuente de fibra", "bajo en grasa"], "comparativa": ["azúcar", "grasas saturadas"]},
    "Bebida de avena": {"contenido": ["fuente de fibra", "adicionado con vitamina D"], "comparativa": ["azúcar", "calorías"]},
    "Pan de caja": {"contenido": ["fuente de fibra", "bajo en grasa"], "comparativa": ["sodio", "azúcar"]},
    "Leche": {"contenido": ["bajo en grasa", "adicionado con vitamina D"], "comparativa": ["grasa", "calorías"]},
    "Barra de granola": {"contenido": ["fuente de fibra", "alto en proteína"], "comparativa": ["azúcar"]},
    "Queso panela": {"contenido": ["bajo en grasa", "alto en proteína", "bajo en sodio"], "comparativa": ["grasa", "sodio"]},
    "Salsa de tomate": {"contenido": ["bajo en sodio"], "comparativa": ["sodio", "azúcar"]},
    "Atún en agua": {"contenido": ["alto en proteína", "bajo en sodio"], "comparativa": ["sodio"]},
}


def concuerda(adjetivo: str, genero: str, plural: bool) -> str:
    """Concordancia de género y número para adjetivos terminados en -o o en consonante."""
    if adjetivo.endswith("o"):
        forma = adjetivo[:-1] + ("a" if genero == "f" else "o")
        return forma + ("s" if plural else "")
    return adjetivo + ("es" if plural else "")


def candidatos_nutrimental():
    out = []
    negativos = ["grasa", "grasas saturadas", "azúcares", "sodio", "calorías", "colesterol"]
    for plantilla in ["Bajo en {n}", "Reducido en {n}", "Contenido reducido de {n}", "Bajo contenido de {n}"]:
        for n in negativos:
            out.append((plantilla.format(n=n), "contenido_bajo_o_sin", "estandar"))
    positivos = ["fibra", "proteína", "calcio", "hierro", "vitamina D", "vitamina C", "zinc",
                 "ácidos grasos omega-3", "potasio", "magnesio", "vitamina B12", "ácido fólico", "vitamina E"]
    for plantilla in ["Fuente de {n}", "Alto en {n}", "Rico en {n}", "Alto contenido de {n}",
                      "Buena fuente de {n}", "Adicionado con {n}"]:
        for n in positivos:
            out.append((plantilla.format(n=n), "contenido_alto_o_fuente", "estandar"))
    for p in [25, 30, 50]:
        for n in ["grasa", "azúcar", "sodio", "calorías", "grasas saturadas"]:
            out.append((f"{p} % menos {n} que la versión original", "comparativa", "estandar"))
    # Solo combinaciones producto-claim plausibles (p. ej., no "atún fuente de fibra")
    for prod, g, pl in PRODUCTOS:
        for clave in PLAUSIBLES[prod]["contenido"]:
            if clave == "fuente de fibra":
                out.append((f"{prod} fuente de fibra", "contenido_alto_o_fuente", "estandar"))
            else:
                adj, resto, sub = FRASES_CONTENIDO[clave]
                out.append((f"{prod} {concuerda(adj, g, pl)} {resto}", sub, "estandar"))
        for n in PLAUSIBLES[prod]["comparativa"]:
            for p in [25, 30]:
                out.append((f"{prod} con {p} % menos {n}", "comparativa", "estandar"))
    return out


def candidatos_salud():
    pares = [
        ("El calcio", "al funcionamiento normal de los músculos"),
        ("La vitamina D", "a la absorción normal del calcio"),
        ("La vitamina D", "al funcionamiento normal del sistema inmunitario"),
        ("La vitamina C", "a mejorar la absorción del hierro"),
        ("La vitamina C", "a la formación normal de colágeno"),
        ("El hierro", "a la formación normal de glóbulos rojos y hemoglobina"),
        ("El hierro", "a la función cognitiva normal"),
        ("El zinc", "al funcionamiento normal del sistema inmunitario"),
        ("El magnesio", "a disminuir el cansancio y la fatiga"),
        ("La vitamina B12", "al funcionamiento normal del sistema nervioso"),
        ("La vitamina B6", "a regular la actividad hormonal"),
        ("Las proteínas", "al mantenimiento de los huesos en condiciones normales"),
        ("El ácido fólico", "a la síntesis normal de aminoácidos"),
        ("La vitamina A", "al mantenimiento de la piel en condiciones normales"),
        ("La vitamina E", "a la protección de las células frente al daño oxidativo"),
        ("El selenio", "al funcionamiento normal de la tiroides"),
        ("El yodo", "a la función cognitiva normal"),
        ("La tiamina", "al funcionamiento normal del corazón"),
        ("La fibra de centeno", "al funcionamiento intestinal normal"),
    ]
    out = []
    for sujeto, funcion in pares:
        verbo = "contribuyen" if sujeto.startswith(("Los ", "Las ")) else "contribuye"
        out.append((f"{sujeto} {verbo} {funcion}", "funcion_nutrimento", "estandar"))
        nutr = sujeto.split(" ", 1)[1]
        out.append((f"Con {nutr}, que {verbo} {funcion}", "funcion_nutrimento", "ambiguo"))
    return out


def candidatos_ausencia():
    sustancias = {
        "gluten": "gluten", "lactosa": "alergeno_intolerancia", "huevo": "alergeno_intolerancia",
        "cacahuate": "alergeno_intolerancia", "soya": "alergeno_intolerancia",
        "conservadores": "aditivo", "colorantes artificiales": "aditivo",
        "saborizantes artificiales": "aditivo", "glutamato monosódico": "aditivo",
        "nitritos añadidos": "aditivo", "aceite de palma": "ingrediente_o_produccion",
        "transgénicos": "ingrediente_o_produccion", "cafeína": "ingrediente_o_produccion",
    }
    out = []
    for plantilla in ["Sin {s}", "Libre de {s}", "No contiene {s}", "Elaborado sin {s}", "Formulado sin {s}"]:
        for s, sub in sustancias.items():
            out.append((plantilla.format(s=s), sub, "estandar"))
    return out


def candidatos_marketing():
    regiones = ["Oaxaca", "Michoacán", "Veracruz", "Sonora", "Yucatán", "Jalisco", "Chihuahua", "Puebla"]
    out = []
    for r in regiones:
        out.append((f"Hecho en {r}", "marketing", "estandar"))
        out.append((f"Receta tradicional de {r}", "marketing", "estandar"))
    for prod, g, pl in PRODUCTOS:
        for d in [concuerda("artesanal", g, pl), "con el sabor de siempre", "de calidad superior",
                  "de receta original", concuerda("elaborado", g, pl) + " en pequeños lotes"]:
            out.append((f"{prod} {d}", "marketing", "estandar"))
    for d in ["Sabor auténtico", "Receta original", "Calidad superior", "Seleccionado a mano",
              "Tradición familiar desde hace tres generaciones", "Sabor que te encanta",
              "Nueva presentación", "Empaque resellable"]:
        out.append((d, "marketing", "estandar"))
    return out


PLANTILLAS = {
    "NUTRIMENTAL": candidatos_nutrimental,
    "SALUD": candidatos_salud,
    "AUSENCIA": candidatos_ausencia,
    "MARKETING": candidatos_marketing,
}


def referencia(subtipo: str) -> str:
    return REFERENCIAS.get(subtipo, REFERENCIAS["marketing"])


def construir(semilla: int = SEMILLA):
    rng = random.Random(semilla)
    filas, vistos = [], set()

    for clase, objetivo in OBJETIVO_POR_CLASE.items():
        manuales = MANUALES[clase]
        for texto, subtipo, dif in manuales:
            vistos.add(texto.lower())
            filas.append((texto, clase, subtipo, dif, "redaccion_manual"))
        faltan = objetivo - len(manuales)
        candidatos = [c for c in PLANTILLAS[clase]() if c[0].lower() not in vistos]
        rng.shuffle(candidatos)
        if faltan > len(candidatos):
            raise ValueError(f"No hay suficientes candidatos de plantilla para {clase}")
        for texto, subtipo, dif in candidatos[:faltan]:
            vistos.add(texto.lower())
            filas.append((texto, clase, subtipo, dif, "plantilla"))

    rng.shuffle(filas)

    # Partición estratificada dev/test por clase
    por_clase = defaultdict(list)
    for i, f in enumerate(filas):
        por_clase[f[1]].append(i)
    particion = {}
    for clase, idx in por_clase.items():
        idx = idx[:]
        rng.shuffle(idx)
        n_dev = round(len(idx) * FRACCION_DEV)
        for j, i in enumerate(idx):
            particion[i] = "dev" if j < n_dev else "test"

    registros = []
    for i, (texto, clase, subtipo, dif, origen) in enumerate(filas):
        registros.append({
            "id_claim": f"CLM-{i + 1:04d}",
            "texto_claim": texto,
            "categoria": clase,
            "subtipo": subtipo,
            "referencia_normativa": referencia(subtipo),
            "dificultad": dif,
            "origen_registro": origen,
            "n_palabras": len(texto.split()),
            "particion": particion[i],
        })
    return registros


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parents[1]
    parser.add_argument("--salida", default=str(raiz / "data" / "claims_etiquetado.csv"))
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args()

    registros = construir(args.semilla)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(registros[0].keys()))
        w.writeheader()
        w.writerows(registros)

    print(f"Escrito {salida} con {len(registros)} claims")
    print("Por clase:      ", dict(Counter(r["categoria"] for r in registros)))
    print("Por partición:  ", dict(Counter(r["particion"] for r in registros)))
    print("Por dificultad: ", dict(Counter(r["dificultad"] for r in registros)))
    print("Por origen:     ", dict(Counter(r["origen_registro"] for r in registros)))


if __name__ == "__main__":
    main()
