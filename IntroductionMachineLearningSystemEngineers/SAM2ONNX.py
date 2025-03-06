# Based on Ibai Gorordo's work in https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything
# Models by https://huggingface.co/mabote-itumeleng/ONNX-SAM2-Segment-Anything/tree/main


import numpy as np
import onnxruntime

import skimage as ski

import time

from typing import Any

class SAM2Image:
    def __init__(self, encoder_path: str, decoder_path: str) -> None:
        # Initialize models
        self.encoder = SAM2ImageEncoder(encoder_path)
        self.orig_im_size = self.encoder.input_shape[2:]
        self.decoder_path = decoder_path
        self.decoders = {}

        self.point_coords = {}
        self.box_coords = {}
        self.point_labels = {}
        self.masks = {}

    def set_image(self, image: np.ndarray) -> None:
        self.image_embeddings = self.encoder(image)
        self.orig_im_size = (image.shape[0], image.shape[1])
        self.reset_points()

    def add_point(self, point_coords: tuple[int, int], is_positive: bool, label_id: int) -> dict[int, np.ndarray]:

        if label_id not in self.decoders:
            self.decoders[label_id] = SAM2ImageDecoder(self.decoder_path, self.encoder.input_shape[2:], self.orig_im_size)

        if label_id not in self.point_coords:
            self.point_coords[label_id] = np.array([point_coords])
            self.point_labels[label_id] = np.array([1 if is_positive else 0])
        else:
            self.point_coords[label_id] = np.append(self.point_coords[label_id], np.array([point_coords]), axis=0)
            self.point_labels[label_id] = np.append(self.point_labels[label_id], 1 if is_positive else 0)

        return self.decode_mask(label_id)

    def set_box(self, box_coords: tuple[tuple[int, int], tuple[int, int]], label_id: int) -> dict[int, np.ndarray]:

        if label_id not in self.decoders:
            self.decoders[label_id] = SAM2ImageDecoder(self.decoder_path, self.encoder.input_shape[2:], self.orig_im_size)

        point_coords = np.array([box_coords[0], box_coords[1]]) # Convert from 1x4 to 2x2

        self.box_coords[label_id] = point_coords

        return self.decode_mask(label_id)

    def decode_mask(self, label_id: int) -> dict[int, np.ndarray]:
        concat_coords, concat_labels = self.merge_points_and_boxes(label_id)

        decoder = self.decoders[label_id]
        high_res_feats_0, high_res_feats_1, image_embed = self.image_embeddings
        if concat_coords.size == 0:
            mask = np.zeros((self.orig_im_size[0], self.orig_im_size[1]), dtype=np.uint8)
        else:
            mask, _ = decoder(image_embed, high_res_feats_0, high_res_feats_1, concat_coords, concat_labels)
        self.masks[label_id] = mask

        return self.masks

    def merge_points_and_boxes(self, label_id: int) -> tuple[np.ndarray, np.ndarray]:
        concat_coords = []
        concat_labels = []
        has_points = label_id in self.point_coords
        has_boxes = label_id in self.box_coords

        if not has_points and not has_boxes:
            return np.array([]), np.array([])

        if has_points:
            concat_coords.append(self.point_coords[label_id])
            concat_labels.append(self.point_labels[label_id])
        if has_boxes:
            concat_coords.append(self.box_coords[label_id])
            concat_labels.append(np.array([2, 3]))
        concat_coords = np.concatenate(concat_coords, axis=0)
        concat_labels = np.concatenate(concat_labels, axis=0)

        return concat_coords, concat_labels

    def remove_point(self,  point_coords: tuple[int, int], label_id: int) -> dict[int, np.ndarray]:
        point_id = np.where((self.point_coords[label_id][:, 0] == point_coords[0]) & (self.point_coords[label_id][:, 1] == point_coords[1]))[0][0]
        self.point_coords[label_id] = np.delete(self.point_coords[label_id], point_id, axis=0)
        self.point_labels[label_id] = np.delete(self.point_labels[label_id], point_id, axis=0)

        return self.decode_mask(label_id)

    def remove_box(self, label_id: int) -> dict[int, np.ndarray]:
        del self.box_coords[label_id]
        return self.decode_mask(label_id)

    def get_masks(self) -> dict[int, np.ndarray]:
        return self.masks

    def reset_points(self) -> None:
        self.point_coords = {}
        self.box_coords = {}
        self.point_labels = {}
        self.masks = {}
        self.decoders = {}


class SAM2ImageEncoder:
    def __init__(self, path: str) -> None:
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.encode_image(image)

    def encode_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        outputs = self.infer(input_tensor)

        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        # Resize input image
        input_img = ski.transform.resize(image, (self.input_width, self.input_height), preserve_range = True)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return outputs[0], outputs[1], outputs[2]

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


class SAM2ImageDecoder:
    def __init__(self, path: str,
                 encoder_input_size: tuple[int, int],
                 orig_im_size: tuple[int, int] = None,
                 mask_threshold: float = 0.0) -> None:
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        self.orig_im_size = orig_im_size if orig_im_size is not None else encoder_input_size
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image_embed: np.ndarray,
                 high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                 point_coords: np.ndarray, point_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        return self.predict(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels)

    def predict(self, image_embed: np.ndarray,
                 high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                 point_coords: np.ndarray, point_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        inputs = self.prepare_inputs(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels)

        outputs = self.infer(inputs)

        return self.process_output(outputs)


    def prepare_inputs(self, image_embed: np.ndarray,
                       high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                       point_coords: np.ndarray, point_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        input_point_coords, input_point_labels = self.prepare_points(point_coords, point_labels)

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros((num_labels, 1, self.encoder_input_size[0] // self.scale_factor, self.encoder_input_size[1] // self.scale_factor), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        original_size = np.array([self.orig_im_size[0], self.orig_im_size[1]], dtype=np.int32)

        return image_embed, high_res_feats_0, high_res_feats_1, input_point_coords, input_point_labels, mask_input, has_mask_input, original_size


    def prepare_points(self, point_coords: np.ndarray, point_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        input_point_coords = point_coords[np.newaxis, ...]
        input_point_labels = point_labels[np.newaxis, ...]

        input_point_coords[..., 0] = input_point_coords[..., 0] / self.orig_im_size[1] * self.encoder_input_size[1]  # Normalize x
        input_point_coords[..., 1] = input_point_coords[..., 1] / self.orig_im_size[0] * self.encoder_input_size[0]  # Normalize y

        return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)

    def infer(self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> list[np.ndarray]:
        start = time.perf_counter()

        outputs = self.session.run(self.output_names,
                                   {self.input_names[i]: inputs[i] for i in range(len(self.input_names))})

        print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:

        scores = outputs[1].squeeze()
        masks = outputs[0]
        masks = masks > self.mask_threshold
        masks = masks.astype(np.uint8).squeeze()

        return masks, scores

    def set_image_size(self, orig_im_size: tuple[int, int]) -> None:
        self.orig_im_size = orig_im_size

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
