"""
Demo para un Space de Hugging Face (SDK Gradio): clasifica claims de etiquetado con el
modelo recomendado y muestra a qué checklist se enrutaría cada uno.

Para publicarlo: crea un Space con SDK Gradio, sube este archivo como app.py y un
requirements.txt con: transformers, torch, sentencepiece, protobuf, gradio.
Variables opcionales del Space: MODEL_ID y TAU (umbral de confianza elegido en dev).
"""
import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
TAU = float(os.environ.get("TAU", "0.60"))
PLANTILLA = "Esta declaración de una etiqueta de alimentos habla de {}."
ETIQUETAS = {
    "NUTRIMENTAL": "la cantidad de un nutrimento como grasa, azúcar, sodio, fibra, proteína, vitaminas o calorías",
    "SALUD": "un beneficio para la salud o una función del cuerpo",
    "AUSENCIA": "la ausencia de un alérgeno, aditivo o ingrediente como gluten, lactosa, conservadores o colorantes",
    "MARKETING": "la calidad, el origen, la elaboración o el sabor del producto",
}
CHECKLIST = {
    "NUTRIMENTAL": "Umbrales vs. tabla nutrimental",
    "SALUD": "Declaraciones autorizadas y evidencia",
    "AUSENCIA": "Evidencia analítica / alérgenos",
    "MARKETING": "Veracidad (no engañoso)",
}
VERBAL_A_CLASE = {v: k for k, v in ETIQUETAS.items()}
clf = pipeline("zero-shot-classification", model=MODEL_ID)


def clasificar(texto: str):
    claims = [c.strip() for c in texto.splitlines() if c.strip()]
    if not claims:
        return []
    salidas = clf(claims, candidate_labels=list(ETIQUETAS.values()), hypothesis_template=PLANTILLA)
    if isinstance(salidas, dict):
        salidas = [salidas]
    filas = []
    for claim, s in zip(claims, salidas):
        clase, conf = VERBAL_A_CLASE[s["labels"][0]], float(s["scores"][0])
        ruta = CHECKLIST[clase] if conf >= TAU else "Revisión humana (confianza < τ)"
        filas.append([claim, clase, round(conf, 3), ruta])
    return filas


demo = gr.Interface(
    fn=clasificar,
    inputs=gr.Textbox(lines=8, label="Claims de la etiqueta (uno por línea)"),
    outputs=gr.Dataframe(headers=["Claim", "Clase propuesta", "Confianza", "Enrutamiento"]),
    title="ClaimCheck IA · Enrutamiento de claims",
    description=f"Modelo: {MODEL_ID} · umbral τ = {TAU}. La IA propone, el revisor decide.",
    examples=[["Sin azúcares añadidos\nFuente de fibra\nSin gluten\nEl calcio es necesario para el mantenimiento de los huesos\nReceta tradicional"]],
)

if __name__ == "__main__":
    demo.launch()
