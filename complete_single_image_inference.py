#!/usr/bin/env python3
"""
Complete IterativeSG Single Image Inference Script

This script runs the IterativeSG model on a single image to generate scene graphs.
It handles model loading, image preprocessing, inference, and visualization.

Requirements:
- VG-SGG-with-attri.h5
- VG-SGG-dicts-with-attri.json  
- image_data.json
- Trained model checkpoint

Usage:
    python complete_single_image_inference.py --image path/to/image.jpg --config configs/iterative_model.yaml --weights path/to/model.pth
"""

import argparse
import os
import sys
import json
import h5py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances, Boxes
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model

# Project imports - import directly to avoid relative import issues
try:
    from configs.defaults import add_dataset_config, add_scenegraph_config
except ImportError:
    sys.path.append(os.path.join(project_root, 'configs'))
    from defaults import add_dataset_config, add_scenegraph_config

try:
    from modeling import Detr
except ImportError:
    sys.path.append(os.path.join(project_root, 'modeling'))
    from meta_arch.detr import Detr

try:
    from data.dataset_mapper import DetrDatasetMapper
except ImportError:
    sys.path.append(os.path.join(project_root, 'data'))
    from dataset_mapper import DetrDatasetMapper

try:
    from data.tools import register_datasets
except ImportError:
    sys.path.append(os.path.join(project_root, 'data'))
    from tools import register_datasets

class IterativeSGInference:
    """Complete inference pipeline for IterativeSG model"""
    
    def __init__(self, 
                 config_file: str,
                 model_weights: str,
                 vg_dict_file: str,
                 vg_h5_file: str, 
                 image_data_file: str,
                 device: str = "cuda"):
        """
        Initialize the inference pipeline
        
        Args:
            config_file: Path to model config YAML file
            model_weights: Path to trained model weights
            vg_dict_file: Path to VG-SGG-dicts-with-attri.json
            vg_h5_file: Path to VG-SGG-with-attri.h5
            image_data_file: Path to image_data.json
            device: Device to run inference on
        """
        self.device = device
        self.config_file = config_file
        self.model_weights = model_weights
        
        # Load Visual Genome data files
        self.vg_dict_file = vg_dict_file
        self.vg_h5_file = vg_h5_file
        self.image_data_file = image_data_file
        
        # Setup logging
        setup_logger()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and load weights
        self._setup_config()
        self._load_vg_data()
        self._build_model()
        self._load_weights()
        
    def _setup_config(self):
        """Setup detectron2 configuration"""
        self.cfg = get_cfg()
        add_dataset_config(self.cfg)
        add_scenegraph_config(self.cfg)
        
        # Load config file
        self.cfg.merge_from_file(self.config_file)
        
        # Set paths for Visual Genome data
        self.cfg.DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY = self.vg_dict_file
        self.cfg.DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 = self.vg_h5_file
        self.cfg.DATASETS.VISUAL_GENOME.IMAGE_DATA = self.image_data_file
        self.cfg.DATASETS.VISUAL_GENOME.IMAGES = ""  # Not needed for single image inference
        
        # Set device
        self.cfg.MODEL.DEVICE = self.device
        
        # Set to eval mode - disable some training-specific features
        self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE = "sgdet"
        
        # Disable frequency bias for inference since we don't have statistics
        if hasattr(self.cfg.MODEL.DETR, 'USE_FREQ_BIAS'):
            self.cfg.MODEL.DETR.USE_FREQ_BIAS = False
            
        self.cfg.freeze()
        
        # Register datasets
        try:
            register_datasets(self.cfg)
            
            # Add dummy statistics to avoid the error
            metadata = MetadataCatalog.get("VG_train")
            if not hasattr(metadata, 'statistics'):
                # We need to load VG data first to get the correct number of predicates
                # This is a temporary load just to get the size
                temp_vg_dict = None
                try:
                    with open(self.vg_dict_file, 'r') as f:
                        temp_vg_dict = json.load(f)
                    num_predicates = len(temp_vg_dict['idx_to_predicate'])
                except:
                    num_predicates = 51  # fallback
                
                # Create dummy statistics structure with correct size
                dummy_stats = {
                    'fg_rel_count': torch.zeros(num_predicates),  # Use actual number of predicates
                    'bg_rel_count': torch.zeros(1),
                    'pred_dist': torch.ones(num_predicates) / num_predicates  # Uniform distribution
                }
                metadata.set(statistics=dummy_stats)
                self.logger.info(f"Added dummy statistics to metadata with {num_predicates} predicates")
                
        except Exception as e:
            self.logger.warning(f"Dataset registration failed: {e}, continuing...")
        
    def _load_vg_data(self):
        """Load Visual Genome dictionaries and mappings"""
        try:
            # Load VG dictionaries
            with open(self.vg_dict_file, 'r') as f:
                vg_dict = json.load(f)
                
            self.idx_to_label = vg_dict['idx_to_label']
            self.idx_to_predicate = vg_dict['idx_to_predicate'] 
            self.idx_to_attribute = vg_dict.get('idx_to_attribute', {})
            
            # Convert string keys to integers where needed
            self.idx_to_label = {int(k) if k.isdigit() else k: v for k, v in self.idx_to_label.items()}
            self.idx_to_predicate = {int(k) if k.isdigit() else k: v for k, v in self.idx_to_predicate.items()}
            
            print(f"Loaded {len(self.idx_to_label)} object classes")
            print(f"Loaded {len(self.idx_to_predicate)} predicate classes")
            
        except Exception as e:
            self.logger.error(f"Failed to load VG dictionaries: {e}")
            raise
            
    def _build_model(self):
        """Build the IterativeSG model"""
        try:
            # Use detectron2's build_model directly instead of JointTransformerTrainer
            self.model = build_model(self.cfg)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Model built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            raise
            
    def _load_weights(self):
        """Load model weights"""
        try:
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.model_weights)
            self.logger.info(f"Loaded weights from {self.model_weights}")
        except Exception as e:
            self.logger.error(f"Failed to load weights: {e}")
            raise
            
    def preprocess_image(self, image_path: str) -> Dict:
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with preprocessed image data
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # Create data dict in detectron2 format
            data_dict = {
                "file_name": image_path,
                "image_id": 0,
                "height": height,
                "width": width,
                "image": torch.from_numpy(image_rgb.transpose(2, 0, 1)),
            }
            
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess image: {e}")
            raise
            
    def run_inference(self, image_path: str) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Preprocess image
            data_dict = self.preprocess_image(image_path)
            
            # Prepare input
            inputs = [data_dict]
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(inputs)
                
            return outputs[0]
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
            
    def postprocess_results(self, outputs: Dict) -> Dict:
        """
        Postprocess model outputs to extract scene graph
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed scene graph data
        """
        try:
            instances = outputs["instances"]
            
            # Extract object detections
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            
            # Extract relationships if available
            relations = []
            
            # Check for relationships at the top level of outputs first
            if "rel_pair_idxs" in outputs and outputs["rel_pair_idxs"] is not None:
                rel_pairs = outputs["rel_pair_idxs"].cpu().numpy()
                rel_scores = outputs["pred_rel_scores"].cpu().numpy() if "pred_rel_scores" in outputs else None
                
                # Extract relationship labels from scores (argmax over predicate classes)
                # rel_scores shape should be (num_relations, num_predicate_classes)
                if rel_scores is not None and rel_scores.ndim == 2:
                    # Get the predicted relationship class (exclude background class at index -1)
                    rel_scores_no_bg = rel_scores[:, :-1]  # Remove background class
                    rel_labels = np.argmax(rel_scores_no_bg, axis=1)
                    rel_confidence = np.max(rel_scores_no_bg, axis=1)
                    
                    for i, (subj_idx, obj_idx) in enumerate(rel_pairs):
                        rel_score = rel_confidence[i]
                        rel_label = rel_labels[i]
                        
                        # Only include relationships with reasonable confidence
                        if rel_score > 0.1:  # Threshold for relationship confidence
                            # Convert relationship class index to predicate name  
                            predicate_name = self.idx_to_predicate.get(int(rel_label), f"unknown_predicate_{rel_label}")
                            
                            relations.append({
                                'subject_idx': int(subj_idx),
                                'object_idx': int(obj_idx), 
                                'predicate_idx': int(rel_label),
                                'predicate': predicate_name,
                                'score': float(rel_score)
                            })
            
            # Check instances for relationship data
            elif hasattr(instances, '_rel_pair_idxs') and instances._rel_pair_idxs is not None:
                rel_pairs = instances._rel_pair_idxs.cpu().numpy()
                rel_scores = instances._pred_rel_scores.cpu().numpy() if hasattr(instances, '_pred_rel_scores') else None
                rel_labels = instances._pred_rel_labels.cpu().numpy() if hasattr(instances, '_pred_rel_labels') else None
                
                for i, (subj_idx, obj_idx) in enumerate(rel_pairs):
                    if rel_scores is not None and rel_labels is not None:
                        # Get the relationship score (max across all predicate classes)
                        rel_score = rel_scores[i].max() if rel_scores[i].ndim > 0 else rel_scores[i]
                        rel_label = rel_labels[i]
                        
                        # Convert relationship class index to predicate name  
                        predicate_name = self.idx_to_predicate.get(int(rel_label), f"unknown_predicate_{rel_label}")
                        
                        relations.append({
                            'subject_idx': int(subj_idx),
                            'object_idx': int(obj_idx), 
                            'predicate_idx': int(rel_label),
                            'predicate': predicate_name,
                            'score': float(rel_score)
                        })
            
            # Check for old-style relationship attributes (fallback)
            elif len(relations) == 0 and hasattr(instances, 'pred_relations') and instances.pred_relations is not None:
                rel_pairs = instances.pred_relations.cpu().numpy()
                rel_scores = instances.rel_scores.cpu().numpy() if hasattr(instances, 'rel_scores') else None
                rel_classes = instances.rel_classes.cpu().numpy() if hasattr(instances, 'rel_classes') else None
                
                for i, (subj_idx, obj_idx) in enumerate(rel_pairs):
                    if rel_scores is not None and rel_classes is not None:
                        relations.append({
                            'subject_idx': int(subj_idx),
                            'object_idx': int(obj_idx), 
                            'predicate_idx': int(rel_classes[i]),
                            'predicate': self.idx_to_predicate.get(int(rel_classes[i]), f"unknown_{rel_classes[i]}"),
                            'score': float(rel_scores[i])
                        })
            
            # Build objects list
            objects = []
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                # The model outputs class indices 0-149, but VG dictionary uses 1-150
                # So we need to add 1 to convert model output to VG dictionary index
                class_idx = int(cls)
                vg_class_idx = class_idx + 1  # Convert model output to VG dictionary index
                
                # Get class name using the VG dictionary index
                class_name = self.idx_to_label.get(vg_class_idx, f"unknown_class_{class_idx}")
                
                objects.append({
                    'idx': i,
                    'bbox': box.tolist(),
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'score': float(score)
                })
                
            return {
                'objects': objects,
                'relations': relations,
                'image_size': (outputs.get('height', 0), outputs.get('width', 0))
            }
            
        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            raise
            
    def visualize_scene_graph(self, 
                             image_path: str, 
                             scene_graph: Dict, 
                             output_path: str = None,
                             show_relations: bool = True,
                             min_score: float = 0.5) -> None:
        """
        Visualize scene graph on image
        
        Args:
            image_path: Path to original image
            scene_graph: Processed scene graph data
            output_path: Path to save visualization (optional)
            show_relations: Whether to draw relationship lines
            min_score: Minimum score threshold for display
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            # Colors for different object classes
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.idx_to_label)))
            color_map = {i: colors[i % len(colors)] for i in range(len(self.idx_to_label))}
            
            # Draw object bounding boxes
            drawn_objects = {}
            for obj in scene_graph['objects']:
                if obj['score'] >= min_score:
                    bbox = obj['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    color = color_map.get(obj['class_idx'], 'red')
                    
                    # Draw bounding box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    label = f"{obj['class_name']}: {obj['score']:.2f}"
                    ax.text(x1, y1-5, label, fontsize=8, color=color, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                    
                    # Store center for relationship lines
                    drawn_objects[obj['idx']] = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Draw relationships
            if show_relations:
                for rel in scene_graph['relations']:
                    if (rel['score'] >= min_score and 
                        rel['subject_idx'] in drawn_objects and 
                        rel['object_idx'] in drawn_objects):
                        
                        subj_center = drawn_objects[rel['subject_idx']]
                        obj_center = drawn_objects[rel['object_idx']]
                        
                        # Draw line
                        ax.plot([subj_center[0], obj_center[0]], 
                               [subj_center[1], obj_center[1]], 
                               'r-', linewidth=2, alpha=0.6)
                        
                        # Add relationship label
                        mid_x = (subj_center[0] + obj_center[0]) / 2
                        mid_y = (subj_center[1] + obj_center[1]) / 2
                        ax.text(mid_x, mid_y, rel['predicate'], fontsize=8, 
                               color='red', ha='center',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
            
            ax.set_title(f"Scene Graph Detection (min_score={min_score})")
            ax.axis('off')
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                self.logger.info(f"Visualization saved to {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise
            
    def print_scene_graph(self, scene_graph: Dict, min_score: float = 0.5) -> None:
        """
        Print scene graph in text format
        
        Args:
            scene_graph: Processed scene graph data
            min_score: Minimum score threshold for display
        """
        print("\n" + "="*60)
        print("SCENE GRAPH ANALYSIS")
        print("="*60)
        
        # Print objects
        print(f"\nOBJECTS (score >= {min_score}):")
        print("-" * 40)
        for obj in scene_graph['objects']:
            if obj['score'] >= min_score:
                print(f"  {obj['idx']:2d}: {obj['class_name']:15s} (score: {obj['score']:.3f})")
                
        # Print relationships
        if scene_graph['relations']:
            print(f"\nRELATIONSHIPS (score >= {min_score}):")
            print("-" * 40)
            for rel in scene_graph['relations']:
                if rel['score'] >= min_score:
                    subj_obj = next(obj for obj in scene_graph['objects'] 
                                  if obj['idx'] == rel['subject_idx'])
                    obj_obj = next(obj for obj in scene_graph['objects'] 
                                 if obj['idx'] == rel['object_idx'])
                    print(f"  {subj_obj['class_name']} --{rel['predicate']}--> {obj_obj['class_name']} (score: {rel['score']:.3f})")
        else:
            print("\nNo relationships detected.")
            
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="IterativeSG Single Image Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--config", default="configs/iterative_model.yaml", 
                       help="Path to config file")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--vg-dict", required=True, 
                       help="Path to VG-SGG-dicts-with-attri.json")
    parser.add_argument("--vg-h5", required=True,
                       help="Path to VG-SGG-with-attri.h5") 
    parser.add_argument("--image-data", required=True,
                       help="Path to image_data.json")
    parser.add_argument("--output", help="Path to save visualization")
    parser.add_argument("--min-score", type=float, default=0.5,
                       help="Minimum detection score threshold")
    parser.add_argument("--no-relations", action="store_true",
                       help="Don't visualize relationships")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return
    if not os.path.exists(args.vg_dict):
        print(f"Error: VG dict file not found: {args.vg_dict}")
        return
    if not os.path.exists(args.vg_h5):
        print(f"Error: VG H5 file not found: {args.vg_h5}")
        return
    if not os.path.exists(args.image_data):
        print(f"Error: Image data file not found: {args.image_data}")
        return
    
    try:
        # Initialize inference pipeline
        print("Initializing IterativeSG inference pipeline...")
        inference = IterativeSGInference(
            config_file=args.config,
            model_weights=args.weights,
            vg_dict_file=args.vg_dict,
            vg_h5_file=args.vg_h5,
            image_data_file=args.image_data,
            device=args.device
        )
        
        # Run inference
        print(f"Running inference on {args.image}...")
        outputs = inference.run_inference(args.image)
        
        # Process results
        print("Processing results...")
        scene_graph = inference.postprocess_results(outputs)
        
        # Print results
        inference.print_scene_graph(scene_graph, min_score=args.min_score)
        
        # Visualize results
        print("Creating visualization...")
        output_path = args.output or args.image.replace('.', '_scene_graph.')
        inference.visualize_scene_graph(
            args.image, 
            scene_graph, 
            output_path=output_path,
            show_relations=not args.no_relations,
            min_score=args.min_score
        )
        
        print(f"\nInference completed successfully!")
        if args.output:
            print(f"Visualization saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()