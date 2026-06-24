"""
Utilidades para predicciones en Jupyter Notebooks
=================================================

Funciones helper diseñadas especialmente para usar en notebooks.
Importa este archivo en tus notebooks para acceder a funciones de predicción listos para usar.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from prediction_pipeline import PredictionPipeline
from model import InVolcModel
from modelo_v3 import BinarySegmentationUNet
import helper_functions as hf


# Variable global para el pipeline (inicializar una sola vez)
_pipeline = None
_pipelines = {}


def _make_pipeline_key(model_path: str, model_class, model_kwargs: Optional[Dict], transform_config: Optional[Dict], device: Optional[torch.device]) -> Tuple[str, str, str, str, str]:
    """Genera una clave estable para cachear pipelines."""
    model_class_repr = model_class.__name__ if model_class is not None else "auto"
    model_repr = repr(model_kwargs or {})
    transform_repr = repr(transform_config or {})
    device_repr = str(device) if device is not None else "auto"
    return (str(model_path), model_class_repr, model_repr, transform_repr, device_repr)


def init_prediction_pipeline(model_path: str = "invol_v4_model.pth",
                            device: Optional[torch.device] = None,
                            model_class=None,
                            model_kwargs: Optional[Dict] = None,
                            transform_config: Optional[Dict] = None) -> PredictionPipeline:
    """
    Inicializa el pipeline de predicciones.
    Se recomienda ejecutar esto una sola vez al principio del notebook.

    Args:
        model_path: Ruta al archivo del modelo
        device: Device para inference (auto-detecta GPU si está disponible)
        model_class: Clase del modelo a cargar (por defecto InVolcModel; usar BinarySegmentationUNet para model_v3)
        model_kwargs: Parámetros para instanciar el modelo
        transform_config: Configuración de transforms (resize/normalize)

    Returns:
        Pipeline inicializado

    Example:
        >>> pipeline = init_prediction_pipeline()
        >>> pred = pipeline.predict("imagen.png")
    """
    global _pipeline, _pipelines

    if device is None:
        device = hf.set_device()

    if model_class is None:
        model_class = InVolcModel

    model_kwargs = model_kwargs or {}
    transform_config = transform_config or {}

    if not transform_config:
        if model_class is BinarySegmentationUNet or model_kwargs.get("in_channels", 1) == 1:
            transform_config = {
                "resize": True,
                "normalize": True,
                "mean": [0.485],
                "std": [0.229],
            }
        else:
            transform_config = {
                "resize": True,
                "normalize": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }

    key = _make_pipeline_key(model_path, model_class, model_kwargs, transform_config, device)

    if key not in _pipelines:
        print("Inicializando pipeline...")
        _pipelines[key] = PredictionPipeline(
            model_path=model_path,
            model_class=model_class,
            device=device,
            target_size=(224, 224),
            model_kwargs=model_kwargs,
            transform_config=transform_config,
        )
        print("✓ Pipeline listo!")

    _pipeline = _pipelines[key]
    return _pipeline


def predict_image(image_path: Union[str, Path],
                  pipeline: Optional[PredictionPipeline] = None) -> Dict:
    """
    Realiza predicción rápida en una imagen.

    Args:
        image_path: Ruta a la imagen
        pipeline: Pipeline de predicciones (si None, usa el global)

    Returns:
        Diccionario con resultados

    Example:
        >>> result = predict_image("test.png")
        >>> print(f"Volcán: {result['clf_pred']}, Confianza: {result['clf_prob']:.3f}")
    """
    if pipeline is None:
        pipeline = _pipeline or init_prediction_pipeline()

    return pipeline.predict(str(image_path))


def predict_batch(image_dir: Union[str, Path],
                 pattern: str = "*.png",
                  pipeline: Optional[PredictionPipeline] = None,
                 save_masks: bool = False,
                 save_csv: bool = True) -> pd.DataFrame:
    """
    Predicción en batch de un directorio de imágenes.

    Args:
        image_dir: Directorio con imágenes
        pattern: Patrón glob para filtrar archivos (default: *.png)
        pipeline: Pipeline de predicciones
        save_masks: Si True, guarda máscaras de segmentación
        save_csv: Si True, guarda resultados en CSV

    Returns:
        DataFrame con resultados

    Example:
        >>> results = predict_batch("../data/InSAR_img", save_csv=True)
        >>> print(f"Volcanes encontrados: {results['clf_pred'].sum()}")
    """
    if pipeline is None:
        pipeline = _pipeline or init_prediction_pipeline()

    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob(pattern))

    print(f"Encontradas {len(image_paths)} imágenes")

    output_csv = f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv" if save_csv else None

    results = pipeline.batch_predict(
        image_paths=[str(p) for p in image_paths],
        save_results=save_masks,
        output_csv=output_csv
    )

    return results


def show_prediction(image_path: Union[str, Path],
                   pipeline: Optional[PredictionPipeline] = None,
                   figsize: Tuple[int, int] = (15, 5)):
    """
    Muestra una imagen con su predicción.

    Args:
        image_path: Ruta a la imagen
        pipeline: Pipeline de predicciones
        figsize: Tamaño de la figura

    Example:
        >>> show_prediction("test.png")
    """
    if pipeline is None:
        pipeline = _pipeline or init_prediction_pipeline()

    # Realizar predicción
    result = pipeline.predict(str(image_path))

    # Obtener imagen original
    from PIL import Image
    img_mode = "L" if getattr(pipeline, "in_channels", 3) == 1 else "RGB"
    img_pil = Image.open(image_path).convert(img_mode).resize(pipeline.target_size)
    img_np = np.array(img_pil) / 255.0

    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Imagen original
    axes[0].imshow(img_np)
    axes[0].set_title("Imagen Original", fontweight='bold')
    axes[0].axis('off')

    # Clasificación
    if result.get('clf_prob') is not None and result.get('clf_pred') is not None:
        class_label = f"VOLCÁN\n({result['clf_prob']:.3f})" if result['clf_pred'] == 1 else f"No volcán\n({result['clf_prob']:.3f})"
        color = "red" if result['clf_pred'] == 1 else "green"
        axes[1].imshow(img_np)
        axes[1].set_title(f"Clasificación: {class_label}", fontweight='bold', color=color)
    else:
        axes[1].imshow(img_np)
        axes[1].set_title("Segmentación", fontweight='bold')
    axes[1].axis('off')

    # Máscara
    seg_mask = np.asarray(result['seg_mask'])
    seg_pred = np.asarray(result.get('seg_pred', (seg_mask > 0.5).astype(np.uint8)))
    axes[2].imshow(img_np, alpha=0.5)
    axes[2].imshow(seg_mask, cmap='hot', alpha=0.6, vmin=0.0, vmax=1.0)
    axes[2].contour(seg_pred, levels=[0.5], colors='cyan', linewidths=1)
    axes[2].set_title(
        f"Segmentación\nmax={seg_mask.max():.2e} | mean={seg_mask.mean():.2e}",
        fontweight='bold'
    )
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return result


def show_batch_predictions(image_dir: Union[str, Path],
                          num_images: int = 9,
                          pattern: str = "*.png",
                           pipeline: Optional[PredictionPipeline] = None,
                          figsize: Tuple[int, int] = (12, 12)):
    """
    Muestra grid de predicciones.

    Args:
        image_dir: Directorio con imágenes
        num_images: Número de imágenes a mostrar
        pattern: Patrón glob para filtrar
        pipeline: Pipeline de predicciones
        figsize: Tamaño de la figura

    Example:
        >>> show_batch_predictions("../data/InSAR_img", num_images=9)
    """
    if pipeline is None:
        pipeline = _pipeline or init_prediction_pipeline()

    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob(pattern))[:num_images]

    if not image_paths:
        print(f"No se encontraron imágenes en {image_dir}")
        return

    pipeline.visualize_batch([str(p) for p in image_paths], max_samples=len(image_paths))


def results_summary(results_df: pd.DataFrame) -> None:
    """
    Imprime un resumen de resultados de predicción.

    Args:
        results_df: DataFrame con resultados

    Example:
        >>> results = predict_batch("../data/InSAR_img")
        >>> results_summary(results)
    """
    if results_df.empty:
        print("DataFrame vacío")
        return

    print("\n" + "="*50)
    print("RESUMEN DE RESULTADOS")
    print("="*50)

    total = len(results_df)
    volcanos = (results_df['clf_pred'] == 1).sum()
    no_volcanos = (results_df['clf_pred'] == 0).sum()

    print(f"\n📊 Estadísticas generales:")
    print(f"   Total de imágenes procesadas: {total}")
    print(f"   Volcanes detectados: {volcanos} ({100*volcanos/total:.1f}%)")
    print(f"   No volcanes: {no_volcanos} ({100*no_volcanos/total:.1f}%)")

    if 'clf_prob' in results_df.columns:
        probs = results_df['clf_prob'].dropna()
        print(f"\n📈 Confianza:")
        print(f"   Promedio: {probs.mean():.4f}")
        print(f"   Mínima: {probs.min():.4f}")
        print(f"   Máxima: {probs.max():.4f}")
        print(f"   Desviación estándar: {probs.std():.4f}")

    # Análisis por rango de confianza
    if 'clf_prob' in results_df.columns:
        print(f"\n📉 Distribución por confianza:")
        high_conf = (results_df['clf_prob'] > 0.7).sum()
        mid_conf = ((results_df['clf_prob'] > 0.4) & (results_df['clf_prob'] <= 0.7)).sum()
        low_conf = (results_df['clf_prob'] <= 0.4).sum()

        print(f"   Alta confianza (>0.7): {high_conf}")
        print(f"   Media confianza (0.4-0.7): {mid_conf}")
        print(f"   Baja confianza (<0.4): {low_conf}")

    print("\n" + "="*50 + "\n")


def get_high_confidence_detections(results_df: pd.DataFrame,
                                   threshold: float = 0.7) -> pd.DataFrame:
    """
    Filtra detecciones de volcanes con alta confianza.

    Args:
        results_df: DataFrame con resultados
        threshold: Umbral de confianza mínima

    Returns:
        DataFrame filtrado

    Example:
        >>> results = predict_batch("../data/InSAR_img")
        >>> high_conf = get_high_confidence_detections(results)
        >>> print(f"Volcanes de alta confianza: {len(high_conf)}")
    """
    return results_df[
        (results_df['clf_pred'] == 1) &
        (results_df['clf_prob'] > threshold)
    ].sort_values('clf_prob', ascending=False)


def compare_predictions(image_path: Union[str, Path],
                       models: Dict[str, str],
                       figsize: Tuple[int, int] = (15, 5)):
    """
    Compara predicciones de múltiples modelos en la misma imagen.

    Args:
        image_path: Ruta a la imagen
        models: Dict con {nombre: ruta_modelo}
        figsize: Tamaño de la figura

    Example:
        >>> models = {
        ...     "Model v1": "invol_v1_model.pth",
        ...     "Model v2": "invol_v2_model.pth"
        ... }
        >>> compare_predictions("test.png", models)
    """
    from PIL import Image

    # Cargar imagen
    img_pil = Image.open(image_path).convert("RGB").resize((256, 256))
    img_np = np.array(img_pil) / 255.0

    # Realizar predicciones con cada modelo
    predictions = {}
    for name, model_path in models.items():
        pipeline = PredictionPipeline(
            model_path=model_path,
            model_class=InVolcModel,
            device=torch.device("cpu")  # Para comparación rápida
        )
        predictions[name] = pipeline.predict(str(image_path))

    # Visualizar
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models + 1, figsize=figsize)

    # Imagen original
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontweight='bold')
    axes[0].axis('off')

    # Predicciones de cada modelo
    for idx, (name, pred) in enumerate(predictions.items(), 1):
        color = "red" if pred['clf_pred'] == 1 else "green"
        label = f"{name}\n{'VOLCÁN' if pred['clf_pred'] == 1 else 'No volcán'}\n({pred['clf_prob']:.3f})"

        axes[idx].imshow(img_np, alpha=0.5)
        axes[idx].imshow(pred['seg_mask'], cmap='hot', alpha=0.6)
        axes[idx].set_title(label, fontweight='bold', color=color)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    return predictions


# ============================================================================
# UTILIDADES PARA ANÁLISIS Y VISUALIZACIÓN
# ============================================================================

def plot_confidence_distribution(results_df: pd.DataFrame, bins: int = 30):
    """Histograma de distribución de confianza."""
    plt.figure(figsize=(10, 5))

    volcanos_conf = results_df[results_df['clf_pred'] == 1]['clf_prob']
    no_volcanos_conf = results_df[results_df['clf_pred'] == 0]['clf_prob']

    plt.hist(volcanos_conf, bins=bins, alpha=0.6, label="Volcanes", color='red')
    plt.hist(no_volcanos_conf, bins=bins, alpha=0.6, label="No volcanes", color='green')
    plt.xlabel("Confianza")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Confianza de Predicciones")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def export_detections_to_csv(results_df: pd.DataFrame,
                            output_path: str = "detecciones.csv"):
    """Exporta solo los volcanes detectados a CSV."""
    volcanos = results_df[results_df['clf_pred'] == 1].sort_values('clf_prob', ascending=False)
    volcanos.to_csv(output_path, index=False)
    print(f"✓ {len(volcanos)} volcanes exportados a {output_path}")
    return volcanos


if __name__ == "__main__":
    print("Este archivo debe importarse en un notebook o script")
    print("Ejemplo:")
    print("  from notebook_utils import init_prediction_pipeline, predict_image")
    print("  pipeline = init_prediction_pipeline()")
    print("  result = predict_image('imagen.png')")
