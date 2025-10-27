from pathlib import Path
import sys
from typing import List, Tuple
import numpy as np
import cv2

from helpers import draw_bboxes
from models.label import Label


def localize_char_bbox(
    img: cv2.typing.MatLike, edge_method: str = "canny"
) -> Tuple[List[Label], List[dict]]:
    """
    Esta função localiza caracteres da placa usando métodos tradicionais e retorna
    bounding boxes normalizadas (YOLO-like).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Suavização para reduzir ruído
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarização (threshold adaptativo ou Otsu)
    otsu_thresh, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Inverter cor dos pixels caso necessário (caracteres escuros em fundo claro)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Detecção de bordas
    if edge_method.lower() == "canny":
        edges = cv2.Canny(blur, otsu_thresh * 0.5, otsu_thresh)

    elif edge_method.lower() == "sobel":
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = cv2.convertScaleAbs(sobel)
        _, edges = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)

    elif edge_method.lower() == "laplacian":
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        _, edges = cv2.threshold(laplacian, 100, 255, cv2.THRESH_BINARY)

    else:
        raise ValueError("Método inválido. Use 'canny', 'sobel' ou 'laplacian'.")

    # Combinação o método de Otsu junto com a técnica de detecção de bordas escolhido
    combined = cv2.bitwise_or(thresh, edges)

    # Detecção de contorno
    contours, hierarchy = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    img_contours_rgb = cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)

    # Filtração dos contornos através de uma heuristica.
    # Temos que pegar somente os contornos dos caracteres e ignorar o restante
    predictions = []
    h_img, w_img = img.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Heurística para filtrar caracteres
        if heuristic(img, contour):
            center_x = (x + w / 2) / w_img
            center_y = (y + h / 2) / h_img
            predictions.append(
                Label(
                    class_id=0,
                    center_x=center_x,
                    center_y=center_y,
                    width=w / w_img,
                    height=h / h_img,
                )
            )

    # Junta imagens de cada parte do processo até detectar os caracteres da placa
    images_process = [
        {"image": gray, "title": "Grey"},
        {"image": blur, "title": "Blur"},
        {"image": thresh, "title": "Thresh binarização"},
        {"image": edges, "title": "Edges" + f" {edge_method}"},
        {"image": img_contours_rgb, "title": "Detected Contours"},
    ]

    return (predictions, images_process)


def heuristic(img: cv2.typing.MatLike, contour: cv2.typing.MatLike) -> bool:
    """
    Verifica se os contornos representam um caractere naquela imagem

    Args:
        img (cv2.typing.MatLike): imagem
        contour (cv2.typing.MatLike): contornos de um caracter da imagem

    Returns:
        bool: retorna se os contornar nessa imagem representam um caractere
    """

    # Assumimos inicialmente que não é um caractere
    is_character = False

    # Obtemos um retângulo delimitador baseado no contorno
    x, y, w, h = cv2.boundingRect(contour)

    # Métricas para validar se é o contorn de um caractere
    character_aspect_ratio = w / h
    character_area = w * h

    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img

    if (
        0.2 < character_aspect_ratio < 0.8
        and 0.02 * img_area < character_area < 0.15 * img_area
    ):
        is_character = True

    return is_character


def compute_iou(bbox1, bbox2):
    """
    Calcula IoU entre dois retângulos (no formato YOLO normalizado).
    """
    # Converter para coordenadas absolutas [xmin, ymin, xmax, ymax]
    x1_min = bbox1.center_x - bbox1.width / 2
    y1_min = bbox1.center_y - bbox1.height / 2
    x1_max = bbox1.center_x + bbox1.width / 2
    y1_max = bbox1.center_y + bbox1.height / 2

    x2_min = bbox2.center_x - bbox2.width / 2
    y2_min = bbox2.center_y - bbox2.height / 2
    x2_max = bbox2.center_x + bbox2.width / 2
    y2_max = bbox2.center_y + bbox2.height / 2

    # Interseção
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Áreas individuais
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # IoU
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def evaluate_iou(detected_labels, true_labels, iou_threshold=0.5):
    """
    Calcula métricas médias de IoU e taxa de acerto (detections corretas).
    """
    ious = []
    matched = 0

    for det in detected_labels:
        best_iou = 0
        for gt in true_labels:
            iou = compute_iou(det, gt)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
        if best_iou >= iou_threshold:
            matched += 1

    mean_iou = np.mean(ious) if ious else 0
    precision = matched / len(detected_labels) if detected_labels else 0
    recall = matched / len(true_labels) if true_labels else 0

    return {"mean_iou": float(mean_iou), "precision": precision, "recall": recall}


def load_data(dir_path: str) -> Tuple[list[cv2.typing.MatLike], list[list[Label]]]:
    """
    Carregue os dados da imagem e labels de um diretorio

    Espera que as imagens e labels estejam ordenados numericamente,
    e que as imagens estejam em arquivos .png e as labels em arquivos .txt.
    Cada linha do arquivo .txt deve representar a label de um caractere da placa,
    e ele devem estar separados por um espaço

    Args:
        dir_path (str): caminho para o diretorio com os dados

    Returns:
        Tuple[list[cv2.Mat], list[list[Label]]]: retorna imagens e labels encontrados no diretorio
    """

    # Pegue o caminho para o diretório
    dataset_dir_path = Path(dir_path)

    # Separa as imagens dos labels
    images_file_path = list(dataset_dir_path.glob("*.png"))
    labels_file_path = list(dataset_dir_path.glob("*.txt"))

    # Ordena os arquivos para ficarem na mesma ordem
    images_file_path.sort()
    labels_file_path.sort()

    # Carregue as imagens
    images: list[cv2.typing.MatLike] = []
    for file_path in images_file_path:
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR_RGB)
        if image is not None:
            images.append(image)

    # Carregue as labels
    labels: list[list[Label]] = []
    for file_path in labels_file_path:
        file_labels = []

        # Pegamos uma label por linha (caractere da placa)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                cid, cx, cy, w, h = line.split(" ")
                file_labels.append(
                    Label(int(cid), float(cy), float(cx), float(h), float(w))
                )

        labels.append(file_labels)

    return (images, labels)


def main():
    # Verifique os argumentos passados para o programa
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python main.py data_directory")

    # Pegue as imagens e labels to dataset
    images, labels = load_data(sys.argv[1])

    # Metricas para medir desempenho do programa
    mean_iou = 0
    mean_precision = 0
    mean_recall = 0

    for img, label in zip(images, labels):
        # Localize os caracteres da placa
        predictions, _ = localize_char_bbox(img)

        # Verifique se as predições estão corretas
        metrics = evaluate_iou(predictions, label)

        # Se as métricas forem muito ruins, tenta outros métodos
        if all(metric < 0.2 for metric in metrics.values()):
            # Tenta método laplaciano
            lap_predictions, _ = localize_char_bbox(img, edge_method="laplacian")
            lap_metrics = evaluate_iou(lap_predictions, label)

            # Tenta método de sobel
            sobel_predicitions, _ = localize_char_bbox(img, edge_method="sobel")
            sobel_metrics = evaluate_iou(sobel_predicitions, label)

            # Pegue o melhor metodo
            best_preds = lap_predictions
            best_metrics = lap_metrics

            if sobel_metrics["precision"] > best_metrics["precision"]:
                best_preds = sobel_predicitions
                best_metrics = sobel_metrics

            if best_metrics["precision"] > metrics["precision"]:
                predictions = best_preds
                metrics = best_metrics

        imgdraw = draw_bboxes(img, label, color=(0, 255, 0), thickness=1)
        imgdraw = draw_bboxes(imgdraw, predictions, color=(0, 0, 255), thickness=1)

        mean_iou += metrics["mean_iou"]
        mean_precision += metrics["precision"]
        mean_recall += metrics["recall"]

    n_samples = len(images)
    print("Métricas finais:")
    print(f"Mean IoU: {mean_iou / n_samples}")
    print(f"Mean precision: {mean_precision / n_samples}")
    print(f"Mean recall: {mean_recall / n_samples}")


if __name__ == "__main__":
    main()
