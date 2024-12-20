"""
Experimental segmentation module based on SegFormer.
Only intended for testing purposes.
Only supported in inference, may be part of the model in the future.
It uses too much GPU resources, not viable for training or real time inference yet.
Nvidia pls launch faster GPUs :)

Requires the transformers library from huggingface to be installed (huggingface.co/transformers)
"""

# Import necessary libraries.
import torch  # Import PyTorch library for tensor operations (English: Importieren Sie die PyTorch-Bibliothek für Tensor-Operationen)
from torch.nn import functional  # Import functional from torch.nn for functional operations (English: Importieren Sie funktionale Operationen von torch.nn)
from torchvision import transforms  # Importing the torchvision library for image transformations (English: Importieren der torchvision-Bibliothek für Bildtransformationen)
import numpy as np  # Import numpy for array manipulation (English: Importieren von numpy für Array-Manipulation)
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation  # Import SegFormer-specific classes for feature extraction and semantic segmentation (English: Importieren von SegFormer-spezifischen Klassen für Merkmalsextraktion und semantische Segmentierung)
from typing import List, Dict  # Import List and Dict from typing module (English: Importieren von List und Dict aus dem Modul "typing")

def cityscapes_palette():
    """
    Returns the cityscapes palette.

    :return: List[List[int]] - The cityscapes palette.
    """
    return [
        [128, 64, 128],  # Color for road (English: Farbe für Straße)
        [244, 35, 232],  # Color for sidewalk (English: Farbe für Gehweg)
        [70, 70, 70],    # Color for building (English: Farbe für Gebäude)
        [102, 102, 156], # Color for wall (English: Farbe für Wand)
        [190, 153, 153], # Color for fence (English: Farbe für Zaun)
        [153, 153, 153], # Color for vegetation (English: Farbe für Vegetation)
        [250, 170, 30],  # Color for tree (English: Farbe für Baum)
        [220, 220, 0],   # Color for sign (English: Farbe für Schild)
        [107, 142, 35],  # Color for terrain (English: Farbe für Gelände)
        [152, 251, 152], # Color for car (English: Farbe für Auto)
        [70, 130, 180],  # Color for bus (English: Farbe für Bus)
        [220, 20, 60],   # Color for person (English: Farbe für Person)
        [255, 0, 0],     # Color for bicycle (English: Farbe für Fahrrad)
        [0, 0, 142],     # Color for motorcycle (English: Farbe für Motorrad)
        [0, 0, 70],      # Color for airplane (English: Farbe für Flugzeug)
        [0, 60, 100],    # Color for train (English: Farbe für Zug)
        [0, 80, 100],    # Color for truck (English: Farbe für Lkw)
        [0, 0, 230],     # Color for boat (English: Farbe für Boot)
        [119, 11, 32],   # Color for traffic light (English: Farbe für Ampel)
    ]

class SequenceResize(object):
    """Prepares the images for the model (English: Bereitet die Bilder für das Modell vor)"""

    def __init__(self, size=(1024, 1024)):
        """
        INIT (English: Initialisierung)

        :param Tuple[int, int] size:  - The size of the output images (English: Die Größe der Ausgabebilder).
        """
        self.size = size

    def __call__(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies the transformation to the images (English: Wendet die Transformation auf die Bilder an).

        :param List[np.ndarray] images: - The images to transform (English: Die Bilder, die transformiert werden sollen).
        :return: List[np.ndarray] - The transformed images (English: Die transformierten Bilder).
        """
        return functional.interpolate(
            images,
            size=self.size,
            mode="bilinear",
            align_corners=False,
        )

class ToTensor(object):
    """Convert np.ndarray images to Tensors. (English: Konvertiert np.ndarray Bilder in Tensoren.)"""

    def __call__(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Applies the transformation to the sequence of images (English: Wendet die Transformation auf die Bildsequenz an).

        :param List[np.ndarray] images: - The images to transform (English: Die Bilder, die transformiert werden sollen).
        :return: List[torch.Tensor] - The transformed images (English: Die transformierten Bilder).
        """
        image1, image2, image3, image4, image5 = (
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
        )

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = image1.transpose((2, 0, 1)).astype(float)
        image2 = image2.transpose((2, 0, 1)).astype(float)
        image3 = image3.transpose((2, 0, 1)).astype(float)
        image4 = image4.transpose((2, 0, 1)).astype(float)
        image5 = image5.transpose((2, 0, 1)).astype(float)

        return [
            torch.from_numpy(image1),
            torch.from_numpy(image2),
            torch.from_numpy(image3),
            torch.from_numpy(image4),
            torch.from_numpy(image5),
        ]

class MergeImages(object):
    """Merges the images into one torch.Tensor (English: Mischt die Bilder in einen einzigen torch.Tensor)."""

    def __call__(self, images: List[torch.tensor]) -> torch.tensor:
        """
        Applies the transformation to the sequence of images (English: Wendet die Transformation auf die Bildsequenz an).

        :param List[torch.tensor] images: - The images to transform (English: Die Bilder, die transformiert werden sollen).
        :return: torch.Tensor - The transformed image (English: Das transformierte Bild).
        """
        image1, image2, image3, image4, image5 = (
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
        )

        return torch.stack([image1, image2, image3, image4, image5])

class ImageSegmentation:
    """
    Class for performing image segmentation. (English: Klasse zur Durchführung der Bildsegmentierung.)
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    ):
        """
        INIT (English: Initialisierung)

        :param torch.device device: - The device to use (English: Das zu verwendende Gerät).
        :param str model_name: - The name of the model to use (English: Der Name des zu verwendenden Modells).
        """
        print(f"Loading feature extractor for {model_name}")  # Loading feature extractor (English: Lädt den Merkmalsextraktor)
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)  # Initialize feature extractor (English: Initialisiert den Merkmalsextraktor)
        print(f"Loading segmentation model for {model_name}")  # Loading segmentation model (English: Lädt das Segmentierungsmodell)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)  # Load segmentation model (English: Lädt das Segmentierungsmodell)
        self.device = device  # Set the device for computations (English: Setzt das Gerät für Berechnungen)
        self.model = self.model.to(device=self.device)  # Transfer model to device (English: Überträgt das Modell auf das Gerät)

        self.image_transforms = transforms.Compose(
            [ToTensor(), MergeImages(), SequenceResize()]
        )  # Defines the transformation pipeline (English: Definiert die Transformations-Pipeline)

    def add_segmentation(self, images: np.ndarray) -> np.ndarray:
        """
        Adds the segmentation to the images. The segmentation is added as a mask over the original images.
        (English: Fügt die Segmentierung zu den Bildern hinzu. Die Segmentierung wird als Maske über die Originalbilder gelegt.)

        :param np.ndarray images: - The images to add the segmentation to (English: Die Bilder, zu denen die Segmentierung hinzugefügt werden soll).
        :return: np.ndarray - The images with the segmentation added (English: Die Bilder mit hinzugefügter Segmentierung).
        """

        original_image_size = images[0].shape  # Get the size of the original image (English: Erhalte die Größe des Originalbildes)
        inputs = torch.vstack(
            [
                self.feature_extractor(images=image, return_tensors="pt")[
                    "pixel_values"
                ]
                for image in images
            ]
        ).to(device=self.device)  # Apply feature extraction (English: Wende Merkmalsextraktion an)

        outputs = self.model(inputs).logits.detach().cpu()  # Perform segmentation on the inputs (English: Führe die Segmentierung auf den Eingaben durch)

        logits = functional.interpolate(
            outputs,
            size=(original_image_size[0], original_image_size[1]),
            mode="bilinear",
            align_corners=False,
        )  # Resize the output to match the original image size (English: Skaliere die Ausgabe, um der Originalbildgröße zu entsprechen)

        segmented_images = logits.argmax(dim=1)  # Get the segmentation labels (English: Erhalte die Segmentierungskennzeichen)

        for image_no, seg in enumerate(segmented_images):  # Loop over each image in the sequence (English: Schleife durch jedes Bild in der Sequenz)
            color_seg = np.zeros(
                (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
            )  # Create an empty color segmentation map (English: Erstelle eine leere Farb-Segmentierungskarte)
            palette = np.array(cityscapes_palette())  # Get the palette for segmentation (English: Hole die Farbpalette für die Segmentierung)
            for label, color in enumerate(palette):  # Assign colors to the segmented labels (English: Ordne den segmentierten Labels Farben zu)
                color_seg[seg == label, :] = color

            images[image_no] = images[image_no] * 0.5 + color_seg * 0.5  # Blend the segmentation with the original image (English: Mische die Segmentierung mit dem Originalbild)

        return images  # Return the images with segmentation (English: Gib die Bilder mit Segmentierung zurück)
