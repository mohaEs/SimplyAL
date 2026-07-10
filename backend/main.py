from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, List, Tuple
import json
import time
import types
import asyncio
import concurrent.futures
import glob
import shutil
import tempfile
import traceback
from model_utils import safe_load_model
import zipfile
import tempfile
import shutil
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score as sklearn_f1_score
import csv
from io import StringIO

from export_utils import add_inference_artifacts

class ProjectInit(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    project_name: str
    model_type: Literal["resnet18", "resnet50", "vit", "vision-transformer", "dinov2", "efficientnet", "custom"] = Field(
        description="Type of model to use"
    )
    num_classes: int = Field(gt=0, description="Number of classes to classify")
    val_split: float = Field(
        default=0.2, 
        gt=0.0, 
        lt=1.0, 
        description="Validation split ratio (0-1)"
    )
    initial_labeled_ratio: float = Field(
        default=0.1, 
        gt=0.0, 
        lt=1.0, 
        description="Initial labeled data ratio (0-1)"
    )

    sampling_strategy: str = Field(
        default="least_confidence",
        description="Active learning sampling strategy"
    )
    batch_size: int = Field(
        default=16,
        gt=0,
        description="Training batch size"
    )
    epochs: int = Field(
        default=5,
        gt=0,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Initial learning rate"
    )

class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    sampling_strategy: str
    learning_rate: float = 0.001

class BatchRequest(BaseModel):
    strategy: str
    batch_size: int

class LabelSubmission(BaseModel):
    image_id: int
    label: int

class TrainEpisodeRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001

class SimpleViTClassifier(nn.Module):
    """Simple ViT-based classifier for RETFound and similar models"""
    def __init__(self, num_classes=2, feature_dim=768):
        super().__init__()

        self.feature_extractor = nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):

        features = self.feature_extractor(x)
        if len(features.shape) > 2:
            features = features.mean(dim=1)
        return self.classifier(features)

def create_vit_model(num_classes=2, image_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
    """Create a Vision Transformer model"""
    try:
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        return model
    except ImportError:
        return SimpleViTClassifier(num_classes=num_classes, feature_dim=embed_dim)

def create_dinov2_model(num_classes=2):
    """Create a DINOv2 (ViT-B/14) model"""
    try:
        import timm
        model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=num_classes,
            img_size=224
        )
        return model
    except Exception as e:
        print(f"timm DINOv2 failed ({e}), trying torch.hub...")
        try:
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            backbone.head = nn.Linear(backbone.embed_dim, num_classes)
            return backbone
        except Exception as e2:
            print(f"torch.hub DINOv2 also failed ({e2}), using SimpleViTClassifier fallback")
            return SimpleViTClassifier(num_classes=num_classes, feature_dim=768)

class ImprovedViTClassifier(nn.Module):
    """Improved ViT-based classifier"""
    def __init__(self, num_classes=2, image_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        
        x = self.transformer(x)
        
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        
        return x

class ActiveLearningManager:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labeled_data = {}
        self.unlabeled_data = {}
        self.validation_data = {}
        self.current_batch = []
        self.episode = 0
        self.annotation_tracking = {}
        self.project_name = None
        self.image_paths = {}
        self.output_dir = None
        self.checkpoint_manager = None
        self.lr_scheduler = None
        self.lr_config = {
            'strategy': 'plateau',
            'initial_lr': 0.001,
            'factor': 0.1,
            'patience': 5,
            'min_lr': 1e-6
        }
        
        self.plot_episode_xvalues = []
        self.plot_episode_yvalues = []
        self.plot_epoch_xvalues = []
        self.plot_epoch_yvalues = []
        self.best_val_acc = 0
        self.best_model_state = None
        self.training_config = {
        'sampling_strategy': 'least_confidence',
        'batch_size': 16,
        'epochs': 5,
        'learning_rate': 0.001,
        'scheduler': {
            'strategy': 'plateau',
            'params': {
                'mode': 'max',
                'factor': 0.1,
                'patience': 5,
                'verbose': True,
                'min_lr': 1e-6
            }
        }
    }

        self.config = {
            'val_split': 0.2,
            'initial_labeled_ratio': 0.1,
        }
        
        self.episode_history = []
        self.ground_truth_labels = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def add_machine_annotation(self, image_id: int, predicted_label: int, confidence: float):
        """Track machine-generated annotation"""
        self.annotation_tracking[image_id] = {
            'label': predicted_label,
            'source': 'machine',
            'confidence': confidence,
            'episode': self.episode
        }

    def initialize_project(self, project_name: str, model_name: str, num_classes: int, config: dict = None):
        try:

            if not project_name:
                raise ValueError("Project name is required")
                
            self.project_name = project_name
            self.output_dir = os.path.join("output", project_name, 
                datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.output_dir, exist_ok=True)
                
            if model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif model_name == "dinov2":
                self.model = create_dinov2_model(num_classes=num_classes)
            elif model_name == "vision-transformer" or model_name == "vit":
                self.model = create_vit_model(num_classes=num_classes)
            elif model_name == "custom":

                print(f"Initializing custom model architecture for {num_classes} classes")
                self.model = self._create_custom_model(num_classes)
            else:

                print(f"Model type '{model_name}' not explicitly supported, treating as custom")
                self.model = self._create_custom_model(num_classes, model_type=model_name)
                
            self.model = self.model.to(self.device)

            self.training_config['model_type'] = model_name
            self.training_config['num_classes'] = num_classes
            self.training_config['project_name'] = project_name

            if config:
                self.training_config.update(config)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config['learning_rate']
            )
            
            scheduler_config = self.training_config.get('scheduler', {
                'strategy': 'plateau',
                'params': {
                    'mode': 'max',
                    'factor': 0.1,
                    'patience': 5,
                    'verbose': True,
                    'min_lr': 1e-6
                }
            })
            
            self.lr_scheduler = LRSchedulerManager(
                optimizer=self.optimizer,
                strategy=scheduler_config['strategy'],
                initial_lr=self.training_config['learning_rate'],
                **scheduler_config['params']
            )
            
            self.checkpoint_manager = CheckpointManager(self.output_dir)
                
            return {
                "status": "success",
                "output_dir": self.output_dir,
                "config": self.training_config
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    def _create_custom_model(self, num_classes, model_type="custom"):
        """
        Create a flexible custom model that can be adapted to different architectures
        """
        try:

            if "vit" in model_type.lower() or "transformer" in model_type.lower():
                return self._create_custom_vit(num_classes)
            elif "resnet" in model_type.lower():
                return self._create_custom_resnet(num_classes)
            elif "efficientnet" in model_type.lower():
                return self._create_custom_efficientnet(num_classes)
            else:

                return self._create_flexible_custom_model(num_classes)
                
        except Exception as e:
            print(f"Error creating custom model: {e}")

            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            return model

    def _create_custom_vit(self, num_classes):
        """Create a custom Vision Transformer model"""
        return ImprovedViTClassifier(
            num_classes=num_classes,
            image_size=224,
            patch_size=16,
            embed_dim=768,
            num_heads=12,
            num_layers=12
        )

    def _create_custom_resnet(self, num_classes):
        """Create a custom ResNet-style model"""
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    def _create_custom_efficientnet(self, num_classes):
        """Create a custom EfficientNet-style model"""
        try:

            import timm
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
            return model
        except ImportError:
            print("timm not available, falling back to ResNet")
            return self._create_custom_resnet(num_classes)

    def _create_flexible_custom_model(self, num_classes):
        """
        Create a flexible custom model that can adapt to different state dicts
        """
        class FlexibleCustomModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()

                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256 * 7 * 7, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return FlexibleCustomModel(num_classes)

    def load_custom_model_weights(self, state_dict, num_classes=None):
        """
        Load weights into a custom model with flexible adaptation
        """
        try:

            model_type = self._detect_model_type_from_state_dict(state_dict)
            
            if num_classes is None:
                num_classes = self._detect_num_classes_from_state_dict(state_dict)
            
            if model_type == "vit":
                self.model = self._create_custom_vit(num_classes)
            elif model_type == "resnet":
                self.model = self._create_custom_resnet(num_classes)
            else:
                self.model = self._create_flexible_custom_model(num_classes)
            
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading custom model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading custom model: {unexpected_keys}")
                
            self.model = self.model.to(self.device)
            return True
            
        except Exception as e:
            print(f"Error loading custom model weights: {e}")
            return False

    def _detect_model_type_from_state_dict(self, state_dict):
        """Detect model type from state dict keys"""
        keys = list(state_dict.keys())
        
        if any(key in str(keys) for key in ['cls_token', 'pos_embed', 'patch_embed']):
            return "vit"
        
        if any(key.startswith('layer') for key in keys):
            return "resnet"
        
        if 'features' in str(keys) and 'classifier' in str(keys):
            return "cnn"
        
        return "unknown"

    def _detect_num_classes_from_state_dict(self, state_dict):
        """Detect number of classes from the final layer"""

        final_layer_patterns = ['fc.weight', 'classifier.weight', 'head.weight']
        
        for pattern in final_layer_patterns:
            if pattern in state_dict:
                return state_dict[pattern].shape[0]
        
        for key in state_dict.keys():
            if key.endswith('.weight') and any(pattern.split('.')[0] in key for pattern in final_layer_patterns):
                return state_dict[key].shape[0]
        
        return 2

    def save_state(self, is_best: bool = False):
        """Save complete model and training state"""
        if not self.checkpoint_manager:
            raise ValueError("Checkpoint manager not initialized")
            
        state = {
            'episode': self.episode,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_config': self.training_config,
            'labeled_indices': list(self.labeled_data.keys()),
            'unlabeled_indices': list(self.unlabeled_data.keys()),
            'validation_indices': list(self.validation_data.keys()),
            'metrics': {
                'episode_accuracies': {
                    'x': self.plot_episode_xvalues,
                    'y': self.plot_episode_yvalues
                },
                'epoch_losses': {
                    'x': self.plot_epoch_xvalues,
                    'y': self.plot_epoch_yvalues
                }
            },
            'episode_history': self.episode_history
        }
        
        return self.checkpoint_manager.save_checkpoint(state, is_best)

    def evaluate_model_on_unlabeled(self, num_samples=10):
        """
        Evaluate model performance on a sample of unlabeled data
        Returns predictions with confidence scores for the next batch of images
        """
        try:
            if not self.model or len(self.unlabeled_data) == 0:
                return None
                
            self.model.eval()
            
            sample_size = min(num_samples, len(self.unlabeled_data))
            # sample_ids = list(self.unlabeled_data.keys())[:sample_size]
            
            ## select random sample of unlabeled data for evaluation:
            # all_ids = list(self.unlabeled_data.keys())
            # sample_ids = np.random.choice(
            #     all_ids,
            #     size=sample_size,
            #     replace=False
            # ).tolist()
            
            ## select most uncertain samples of unlabeled data for evaluation:
            sample_batch = self.get_next_batch(
                strategy=self.config.get("sampling_strategy", "least_confidence"),
                batch_size=sample_size
                )

            sample_ids = [x["image_id"] for x in sample_batch]

            predictions = []
            all_confidences = []
            
            with torch.no_grad():
                for img_id in sample_ids:
                    img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
                    outputs = self.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    
                    top_prob, top_class = torch.max(probs, dim=1)
                    confidence = float(top_prob.item())
                    predicted_class = int(top_class.item())
                    
                    all_probs = []
                    for i, prob in enumerate(probs[0]):
                        all_probs.append({
                            'class_index': i,
                            'probability': float(prob.item())
                        })
                    
                    all_probs.sort(key=lambda x: x['probability'], reverse=True)
                    
                    predictions.append({
                        'image_id': img_id,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'all_probabilities': all_probs
                    })
                    
                    all_confidences.append(confidence)
            
            overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            return {
                'predictions': predictions,
                'overall_confidence': overall_confidence,
                'num_evaluated': len(predictions),
                'episode_info': {
                    'episode': self.episode,
                    'validation_accuracy': self.best_val_acc
                }
            }
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None

    def get_evaluation_batch(self, num_samples=10):
        """
        Get the next batch of unlabeled images for evaluation display
        Similar to get_next_batch but focused on evaluation metrics
        """
        try:
            if not self.model or len(self.unlabeled_data) == 0:
                return None
                
            evaluation_data = self.evaluate_model_on_unlabeled(num_samples)
            
            if evaluation_data:

                for pred in evaluation_data['predictions']:

                    pred['uncertainty'] = 1 - pred['confidence']
                    
                    pred['predictions'] = [
                        {
                            'label': f"Class {i}",
                            'confidence': prob['probability']
                        }
                        for i, prob in enumerate(pred['all_probabilities'])
                    ]
            
            return evaluation_data
            
        except Exception as e:
            print(f"Error getting evaluation batch: {str(e)}")
            return None
        
    async def add_initial_data(self, files: List[UploadFile], val_split: float = None):
        """Add initial dataset and split into labeled/unlabeled/validation"""
        if val_split is not None:
            self.config['val_split'] = val_split

        all_data = {}
        for img_file in files:
            content = await img_file.read()
            img = Image.open(io.BytesIO(content)).convert('RGB')
            img_tensor = self.transform(img)
            img_id = len(all_data)
            all_data[img_id] = img_tensor
            
            self.image_paths[img_id] = img_file.filename or f"uploaded_image_{img_id}.jpg"

        total_images = len(all_data)
        val_size = int(total_images * self.config['val_split'])
        initial_labeled_size = int((total_images - val_size) * self.config['initial_labeled_ratio'])

        all_indices = list(all_data.keys())
        np.random.shuffle(all_indices)

        val_indices = all_indices[:val_size]
        initial_labeled_indices = all_indices[val_size:val_size + initial_labeled_size]
        unlabeled_indices = all_indices[val_size + initial_labeled_size:]

        for idx in unlabeled_indices:
            self.unlabeled_data[idx] = all_data[idx]

        for idx in initial_labeled_indices:
            self.unlabeled_data[idx] = all_data[idx]

        for idx in val_indices:

            temp_label = np.random.randint(0, self.model.fc.out_features if hasattr(self.model, 'fc') else 2)
            self.validation_data[idx] = (all_data[idx], temp_label)

        split_info = {
            "total_images": total_images,
            "validation": len(self.validation_data),
            "initial_labeled": 0,
            "unlabeled": len(self.unlabeled_data)
        }

        return split_info

    def rebuild_validation_split(self, val_split: float = None):
        """
        Rebuild validation set from all currently labeled samples.
        Used before every training episode.
        """

        if val_split is not None:
            self.config["val_split"] = val_split

        val_split = self.config.get("val_split", 0.2)

        # Move old validation samples back to labeled_data
        for img_id, (img_tensor, label) in list(self.validation_data.items()):
            if label is not None:
                self.labeled_data[img_id] = (img_tensor, label)

        self.validation_data = {}

        labeled_ids = list(self.labeled_data.keys())
        np.random.shuffle(labeled_ids)

        val_size = int(len(labeled_ids) * val_split)

        if len(labeled_ids) > 1 and val_size == 0:
            val_size = 1

        val_ids = labeled_ids[:val_size]

        for img_id in val_ids:
            img_tensor, label = self.labeled_data.pop(img_id)
            self.validation_data[img_id] = (img_tensor, label)

        print(
            f"Validation rebuilt: "
            f"{len(self.validation_data)} validation / "
            f"{len(self.labeled_data)} training labeled / "
            f"val_split={val_split}"
        )


    def set_custom_model(self, model, model_name="custom"):
        """
        Set a custom model directly (for when users provide their own model)
        """
        try:
            self.model = model.to(self.device)
            self.model_name = model_name
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config.get('learning_rate', 0.001)
            )
            
            print(f"Custom model '{model_name}' set successfully")
            return True
            
        except Exception as e:
            print(f"Error setting custom model: {e}")
            return False

    def adapt_model_for_classes(self, num_classes):
        """
        Adapt the current model for a different number of classes
        """
        try:
            if hasattr(self.model, 'fc'):

                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(self.model, 'classifier'):

                if isinstance(self.model.classifier, nn.Linear):
                    in_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(in_features, num_classes)
                elif isinstance(self.model.classifier, nn.Sequential):

                    for i, layer in enumerate(reversed(self.model.classifier)):
                        if isinstance(layer, nn.Linear):
                            in_features = layer.in_features
                            self.model.classifier[-i-1] = nn.Linear(in_features, num_classes)
                            break
            elif hasattr(self.model, 'head'):

                in_features = self.model.head.in_features
                self.model.head = nn.Linear(in_features, num_classes)
            else:
                print("Warning: Could not adapt model for new number of classes")
                return False
                
            self.model = self.model.to(self.device)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config.get('learning_rate', 0.001)
            )
            
            return True
            
        except Exception as e:
            print(f"Error adapting model for {num_classes} classes: {e}")
            return False

    def get_model_info(self):
        """
        Get information about the current model
        """
        if not self.model:
            return None
            
        try:
            info = {
                'model_class': self.model.__class__.__name__,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(next(self.model.parameters()).device),
                'model_type': 'custom'
            }
            
            if hasattr(self.model, 'fc'):
                info['final_layer'] = 'fc'
                info['num_classes'] = self.model.fc.out_features
            elif hasattr(self.model, 'classifier'):
                info['final_layer'] = 'classifier'
                if isinstance(self.model.classifier, nn.Linear):
                    info['num_classes'] = self.model.classifier.out_features
            elif hasattr(self.model, 'head'):
                info['final_layer'] = 'head'
                info['num_classes'] = self.model.head.out_features
                
            return info
            
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None

    def get_next_batch(self, strategy: str, batch_size: int) -> List[dict]:
        """
        Select next batch of samples using specified strategy with improved error handling
        Args:
            strategy: Sampling strategy to use
            batch_size: Number of samples to select
        Returns:
            List of selected samples with metadata
        """
        if not self.model:
            raise HTTPException(status_code=400, detail="Model not initialized")

        if len(self.unlabeled_data) == 0:
            raise HTTPException(status_code=400, detail="No unlabeled data available")
            
        if batch_size > len(self.unlabeled_data):
            print(f"Warning: Requested batch size {batch_size} is larger than available unlabeled data ({len(self.unlabeled_data)})")
            batch_size = len(self.unlabeled_data)
            
        if batch_size <= 0:
            batch_size = min(16, len(self.unlabeled_data))
            
        try:

            sample_scores = self._get_sample_scores(strategy)
            
            selected_samples = self._select_samples(sample_scores, batch_size, strategy)
            
            self.current_batch = [x["image_id"] for x in selected_samples]
            
            return selected_samples
        except Exception as e:
            print(f"Error in get_next_batch: {str(e)}")
            traceback.print_exc()
            
            if strategy != "random":
                print("Falling back to random sampling strategy")
                try:

                    image_ids = list(self.unlabeled_data.keys())
                    selected_ids = random.sample(image_ids, min(batch_size, len(image_ids)))
                    
                    selected_samples = []
                    for img_id in selected_ids:

                        img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            try:
                                outputs = self.model(img_tensor)
                                probs = torch.softmax(outputs, dim=1)
                                
                                predictions = [
                                    {"label": f"Label {i}", "confidence": float(p)} 
                                    for i, p in enumerate(probs[0])
                                ]
                                
                                selected_samples.append({
                                    "image_id": img_id,
                                    "uncertainty": 0.5,
                                    "predictions": predictions
                                })
                            except Exception as inner_e:

                                print(f"Error making predictions: {str(inner_e)}")
                                selected_samples.append({
                                    "image_id": img_id,
                                    "uncertainty": 0.5,
                                    "predictions": [{"label": "Unknown", "confidence": 0.0}]
                                })
                    
                    self.current_batch = [x["image_id"] for x in selected_samples]
                    return selected_samples
                except Exception as fallback_e:
                    print(f"Fallback strategy also failed: {str(fallback_e)}")
                    traceback.print_exc()
                    raise
            
            else:

                raise
    
    def _get_sample_scores(self, strategy: str) -> List[Tuple[int, float, List[dict]]]:
        """
        Compute scores for all unlabeled samples
        Returns:
            List of tuples (image_id, uncertainty_score, predictions)
        """
        sample_scores = []
        self.model.eval()
        
        with torch.no_grad():
            for img_id, img_tensor in self.unlabeled_data.items():

                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                features = self._get_features(img_tensor) if strategy == "diversity" else None
                
                uncertainty = self._compute_uncertainty(probs, strategy, features)
                
                predictions = [
                    {"label": f"Label {i}", "confidence": float(p)} 
                    for i, p in enumerate(probs[0])
                ]
                
                sample_scores.append((img_id, uncertainty, predictions))
                
        return sample_scores

    def _select_samples(self, sample_scores: List[Tuple[int, float, List[dict]]], 
                   batch_size: int, strategy: str) -> List[dict]:
        """
        Select batch_size samples based on scores and strategy
        """
        if strategy == "diversity":

            selected = self._select_diverse_samples(sample_scores, batch_size)
        else:

            sample_scores.sort(key=lambda x: x[1], reverse=True)
            selected = sample_scores[:batch_size]
        
        return [
            {
                "image_id": img_id,
                "uncertainty": score,
                "predictions": preds
            }
            for img_id, score, preds in selected
        ]
    
    def _save_episode_annotations_csv(self):
        """Save annotations for current episode to CSV"""
        if not self.output_dir:
            return
        
        csv_dir = os.path.join(self.output_dir, 'episode_csvs')
        os.makedirs(csv_dir, exist_ok=True)
        
        episode_annotations = []
        
        for img_id, tracking_info in self.annotation_tracking.items():
            if tracking_info.get('episode') == self.episode:

                image_path = self.image_paths.get(img_id, f"image_{img_id}")
                
                label_idx = tracking_info['label']
                label_name = None
                if hasattr(self, 'current_labels') and self.current_labels:
                    if label_idx < len(self.current_labels):
                        label_name = self.current_labels[label_idx]
                
                annotation = {
                    'image_id': img_id,
                    'image_path': image_path,
                    'label_index': label_idx,
                    'label_name': label_name if label_name else f"Class {label_idx}",
                    'annotation_source': tracking_info['source'],
                    'confidence': tracking_info.get('confidence', 1.0) if tracking_info['source'] == 'machine' else None,
                    'episode': self.episode
                }
                
                episode_annotations.append(annotation)
        
        if episode_annotations:
            csv_path = os.path.join(csv_dir, f'episode_{self.episode:03d}_annotations.csv')
            df = pd.DataFrame(episode_annotations)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(episode_annotations)} annotations to {csv_path}")

    def _save_episode_annotations_csv(self):
        """Save annotations for current episode to CSV with machine predictions"""
        if not self.output_dir:
            return
        
        csv_dir = os.path.join(self.output_dir, 'episode_csvs')
        os.makedirs(csv_dir, exist_ok=True)
        
        episode_annotations = []
        
        for img_id, tracking_info in self.annotation_tracking.items():
            if tracking_info.get('episode') == self.episode:
                image_path = self.image_paths.get(img_id, f"image_{img_id}")
                
                label_idx = tracking_info['label']
                label_name = None
                if hasattr(self, 'current_labels') and self.current_labels:
                    if label_idx < len(self.current_labels):
                        label_name = self.current_labels[label_idx]
                
                machine_pred = tracking_info.get('machine_prediction')
                machine_pred_name = None
                if machine_pred is not None and hasattr(self, 'current_labels') and self.current_labels:
                    if machine_pred < len(self.current_labels):
                        machine_pred_name = self.current_labels[machine_pred]
                
                agreement = (machine_pred == label_idx) if machine_pred is not None else None
                
                annotation = {
                    'image_id': img_id,
                    'image_path': image_path,
                    'human_label_index': label_idx,
                    'human_label_name': label_name if label_name else f"Class {label_idx}",
                    'machine_prediction_index': machine_pred,
                    'machine_prediction_name': machine_pred_name if machine_pred_name else (f"Class {machine_pred}" if machine_pred is not None else "N/A"),
                    'machine_confidence': tracking_info.get('machine_confidence'),
                    'human_machine_agreement': agreement,
                    'annotation_source': tracking_info['source'],
                    'episode': self.episode
                }
                
                episode_annotations.append(annotation)
        
        if episode_annotations:
            csv_path = os.path.join(csv_dir, f'episode_{self.episode:03d}_annotations.csv')
            df = pd.DataFrame(episode_annotations)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(episode_annotations)} annotations to {csv_path}")
            
            if len(episode_annotations) > 0:
                agreements = [a['human_machine_agreement'] for a in episode_annotations if a['human_machine_agreement'] is not None]
                if agreements:
                    accuracy = sum(agreements) / len(agreements) * 100
                    print(f"Episode {self.episode} machine accuracy: {accuracy:.2f}% ({sum(agreements)}/{len(agreements)} correct)")

    def _select_diverse_samples(self, sample_scores: List[Tuple[int, float, List[dict]]], 
                          batch_size: int) -> List[Tuple[int, float, List[dict]]]:
        """
        Select diverse samples using greedy approach
        """
        if batch_size >= len(sample_scores):
            return sample_scores
            
        selected = []
        remaining = sample_scores.copy()
        
        remaining.sort(key=lambda x: x[1], reverse=True)
        selected.append(remaining.pop(0))
        
        while len(selected) < batch_size and remaining:

            remaining_features = []
            for _, _, preds in remaining:
                probs = torch.tensor([[p["confidence"] for p in preds]])
                remaining_features.append(probs)
            remaining_features = torch.cat(remaining_features, dim=0)
            
            selected_features = []
            for _, _, preds in selected:
                probs = torch.tensor([[p["confidence"] for p in preds]])
                selected_features.append(probs)
            selected_features = torch.cat(selected_features, dim=0)
            
            distances = torch.cdist(remaining_features, selected_features)
            min_distances = distances.min(dim=1)[0]
            
            best_idx = min_distances.argmax().item()
            selected.append(remaining.pop(best_idx))
        
        return selected

    def train_epoch(self, optimizer, criterion, batch_size=16):
        """Train for one epoch with proper validation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        all_images = []
        all_labels = []
        for img_tensor, label in self.labeled_data.values():
            all_images.append(img_tensor)
            all_labels.append(label)

        if len(all_images) == 0:
            raise ValueError("No labeled data available for training")

        X = torch.stack(all_images)
        y = torch.tensor(all_labels)

        indices = torch.randperm(len(all_images))
        batch_losses = []

        for i in range(0, len(all_images), batch_size):
            batch_indices = indices[i:min(i + batch_size, len(all_images))]
            batch_X = X[batch_indices].to(self.device)
            batch_y = y[batch_indices].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy
    
    def validate_model(self):
        """Perform validation on the validation set - improved for CSV uploads"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        labeled_validation = {}
        for idx, (tensor, label) in self.validation_data.items():
            if label is not None:
                labeled_validation[idx] = (tensor, label)
        
        if len(labeled_validation) == 0:
            print("Warning: No validation data available")
            return 0.0
        
        batch_size = 16
        with torch.no_grad():
            for idx, (img_tensor, label) in labeled_validation.items():
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == label).sum().item()
                total_samples += 1
                    
        validation_accuracy = 100.0 * total_correct / total_samples
        print(f"Validation Accuracy: {validation_accuracy:.2f}% ({total_correct}/{total_samples})")
        return validation_accuracy

    def validate_model_with_metrics(self):
        """Validate and return (accuracy, weighted_f1).
        
        Collects all predictions and ground-truth labels from the validation
        set (falling back to 20 % of labeled data when the validation set has
        no labels), then computes both accuracy and weighted-average F1 with
        sklearn so the scores are consistent and immune to class-imbalance.
        
        Returns:
            (accuracy: float, f1: float)  — both expressed as percentages 0-100.
        """
        self.model.eval()

        labeled_validation = {
            idx: (tensor, label)
            for idx, (tensor, label) in self.validation_data.items()
            if label is not None
        }

        if len(labeled_validation) == 0:
            print("Warning: No validation data available — returning (0, 0)")
            return 0.0, 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for idx, (img_tensor, label) in labeled_validation.items():
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.item())
                all_labels.append(int(label))

        total = len(all_labels)
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = 100.0 * correct / total

        try:
            f1 = sklearn_f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100.0
        except Exception as e:
            print(f"Warning: F1 computation failed ({e}), defaulting to 0")
            f1 = 0.0

        print(f"Validation — Accuracy: {accuracy:.2f}%  F1 (weighted): {f1:.2f}%  ({correct}/{total})")
        return accuracy, f1

    def validate(self):
        """Validate model performance - improved for active learning"""
        try:
            if len(self.validation_data) == 0:
                print("Warning: No validation data available")
                return 0.0
                    
            labeled_validation = [
                (img, label) for img, label in self.validation_data.values() 
                if label is not None
            ]
            
            if len(labeled_validation) == 0:
                print(f"Warning: No labeled validation data (0/{len(self.validation_data)} samples labeled)")
                print("Validation data exists but needs labels. Consider labeling some validation samples.")
                return 0.0

            self.model.eval()
            total_correct = 0
            total_samples = 0

            batch_size = 16
            with torch.no_grad():
                for i in range(0, len(labeled_validation), batch_size):
                    batch = labeled_validation[i:i + batch_size]
                    images = torch.stack([img for img, _ in batch]).to(self.device)
                    labels = torch.tensor([label for _, label in batch]).to(self.device)

                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            val_accuracy = 100.0 * total_correct / total_samples
            print(f"Validation Accuracy: {val_accuracy:.2f}% ({total_correct}/{total_samples})")
            return val_accuracy

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return 0.0

    def train(self, epochs: int, batch_size: int, learning_rate: float):
        """Train model on labeled data"""
        try:
            if len(self.labeled_data) == 0:
                raise HTTPException(status_code=400, detail="No labeled data available")

            if len(self.validation_data) == 0:
                raise HTTPException(status_code=400, detail="No validation data available")

            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            best_model = None
            
            for epoch in range(epochs):
                try:
                    train_loss, train_acc = self.train_epoch(optimizer, criterion)
                    val_acc = self.validate()

                    self.plot_epoch_xvalues.append(epoch + 1)
                    self.plot_epoch_yvalues.append(train_loss)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = self.model.state_dict().copy()

                    self.plot_training_progress(epoch + 1, train_loss, val_acc)
                except Exception as e:
                    print(f"Error in epoch {epoch}: {str(e)}")
                    raise

            self.plot_episode_xvalues.append(self.episode)
            self.plot_episode_yvalues.append(best_val_acc)
            
            self.episode += 1

            return {
                "status": "success",
                "epochs_completed": epochs,
                "final_accuracy": val_acc,
                "best_accuracy": best_val_acc
            }
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def plot_training_progress(self, epoch, loss, accuracy):
        """Plot and save training progress"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.plot_epoch_xvalues, self.plot_epoch_yvalues)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss (Episode {self.episode})")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.plot_episode_xvalues, self.plot_episode_yvalues)
        plt.xlabel("Episode")
        plt.ylabel("Validation Accuracy")
        plt.title("Active Learning Progress")
        
        plt.savefig(os.path.join(self.output_dir, f"progress_ep{self.episode}_e{epoch}.png"))
        plt.close()

    def submit_label(self, image_id: int, label: int):
        """Submit label for an image and track annotation source"""
        machine_prediction = None
        machine_confidence = None
        
        if image_id in self.unlabeled_data:
            try:
                self.model.eval()
                with torch.no_grad():
                    img_tensor = self.unlabeled_data[image_id].unsqueeze(0).to(self.device)
                    outputs = self.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    top_prob, top_class = torch.max(probs, dim=1)
                    machine_prediction = int(top_class.item())
                    machine_confidence = float(top_prob.item())
            except Exception as e:
                print(f"Could not get machine prediction: {e}")
        
        if image_id in self.validation_data:
            img_tensor = self.validation_data[image_id][0]
            self.validation_data[image_id] = (img_tensor, label)
        elif image_id in self.unlabeled_data:
            img_tensor = self.unlabeled_data.pop(image_id)
            self.labeled_data[image_id] = (img_tensor, label)
            
            self.annotation_tracking[image_id] = {
                'label': label,
                'source': 'human',
                'confidence': 1.0,
                'episode': self.episode,
                'machine_prediction': machine_prediction,
                'machine_confidence': machine_confidence
            }
        else:
            raise HTTPException(status_code=400, detail="Image not found")
        
        return {
            "status": "success",
            "labeled_count": len(self.labeled_data),
            "unlabeled_count": len(self.unlabeled_data),
            "validation_count": len([x for x in self.validation_data.values() if x[1] is not None])
        }
    
    def train_episode(self, epochs: int, batch_size: int, learning_rate: float):
        """Run a complete training episode with improved batch selection, checkpointing, and LR scheduling"""
        self.config["batch_size"] = int(batch_size)
        try:
            if len(self.labeled_data) == 0:
                raise ValueError("No labeled data available for training")
            
            self.rebuild_validation_split(self.config.get("val_split", 0.2))

            if self.checkpoint_manager is None:
                self.checkpoint_manager = CheckpointManager(self.output_dir)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            scheduler_config = self.training_config.get('scheduler', {
                'strategy': 'plateau',
                'params': {
                    'mode': 'max',
                    'factor': 0.1,
                    'patience': 5,
                    'verbose': True,
                    'min_lr': 1e-6
                }
            })
            
            if scheduler_config['strategy'] == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_config['params']
                )
            elif scheduler_config['strategy'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,
                    eta_min=scheduler_config['params'].get('min_lr', 0)
                )
            elif scheduler_config['strategy'] == 'warmup':
                steps_per_epoch = len(self.labeled_data) // batch_size
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_config['params'].get('max_lr', learning_rate * 10),
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    pct_start=scheduler_config['params'].get('warmup_pct', 0.3)
                )
            elif scheduler_config['strategy'] == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config['params'].get('step_size', max(1, epochs // 3)),
                    gamma=scheduler_config['params'].get('gamma', 0.1)
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.1, patience=5, verbose=True
                )
            criterion = nn.CrossEntropyLoss()
            best_val_acc = 0
            best_f1 = 0.0
            best_model_state = None
            lr_history = []

            for epoch in range(epochs):

                train_loss, train_acc = self.train_epoch(optimizer, criterion, batch_size)
                
                val_acc, val_f1 = self.validate_model_with_metrics()
                
                current_lr = optimizer.param_groups[0]['lr']
                if scheduler_config['strategy'] == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
                
                new_lr = optimizer.param_groups[0]['lr']
                lr_history.append({
                    'epoch': epoch + 1,
                    'old_lr': current_lr,
                    'new_lr': new_lr,
                    'val_acc': val_acc
                })
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    
                    try:
                        if self.checkpoint_manager:
                            state = {
                                'episode': self.episode,
                                'model_state': best_model_state,
                                'optimizer_state': optimizer.state_dict(),
                                'scheduler_state': scheduler.state_dict(),
                                'scheduler_config': scheduler_config,
                                'best_val_acc': best_val_acc,
                                'training_config': self.training_config,
                                'labeled_indices': list(self.labeled_data.keys()),
                                'unlabeled_indices': list(self.unlabeled_data.keys()),
                                'validation_indices': list(self.validation_data.keys()),
                                'lr_history': lr_history,
                                'metrics': {
                                    'episode_accuracies': {
                                        'x': self.plot_episode_xvalues,
                                        'y': self.plot_episode_yvalues
                                    },
                                    'epoch_losses': {
                                        'x': self.plot_epoch_xvalues,
                                        'y': self.plot_epoch_yvalues
                                    }
                                },
                                'episode_history': self.episode_history
                            }
                            self.checkpoint_manager.save_checkpoint(state, is_best=True)
                    except Exception as e:
                        print(f"Warning: Failed to save checkpoint: {str(e)}")
                        pass

                self.plot_epoch_xvalues.append(epoch + 1)
                self.plot_epoch_yvalues.append(train_loss)

                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Training Loss: {train_loss:.4f}")
                print(f"Training Accuracy: {train_acc:.2f}%")
                print(f"Validation Accuracy: {val_acc:.2f}%")
                print(f"Learning Rate: {new_lr:.6f}")

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                self.best_val_acc = best_val_acc
                self.best_model_state = best_model_state

            train_result = {
                "status": "success",
                "final_accuracy": val_acc,
                "best_accuracy": best_val_acc,
                "lr_history": lr_history
            }

            evaluation_data = None
            try:
                print("Generating evaluation data for model assessment...")
                evaluation_data = self.get_evaluation_batch(num_samples=10)
                if evaluation_data:
                    print(f"Generated evaluation data for {len(evaluation_data['predictions'])} images")
                else:
                    print("No evaluation data could be generated")
            except Exception as e:
                print(f"Warning: Could not generate evaluation data: {str(e)}")

            try:
                if evaluation_data is None:

                    annotation_batch_size = int(self.config.get("batch_size", batch_size))

                    next_batch = self.get_next_batch(
                        strategy=self.config.get("sampling_strategy", "least_confidence"),
                        batch_size=annotation_batch_size
                    )
                else:

                    next_batch = None
                    
                episode_metrics = {
                    'episode': self.episode,
                    'train_result': train_result,
                    'batch_size': len(next_batch) if next_batch else 0,
                    'strategy': self.training_config["sampling_strategy"],
                    'labeled_size': len(self.labeled_data),
                    'unlabeled_size': len(self.unlabeled_data),
                    'validation_size': len(self.validation_data),
                    'best_val_acc': best_val_acc,
                    'f1_score': best_f1,
                    'learning_rate': new_lr,
                    'lr_history': lr_history
                }
                
                self.episode_history.append(episode_metrics)
                
                self.plot_episode_xvalues.append(self.episode)
                self.plot_episode_yvalues.append(best_val_acc)
                
                if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager:
                    try:
                        state = {
                            'episode': self.episode,
                            'model_state': self.model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                            'scheduler_config': scheduler_config,
                            'best_val_acc': best_val_acc,
                            'training_config': self.training_config,
                            'labeled_indices': list(self.labeled_data.keys()),
                            'unlabeled_indices': list(self.unlabeled_data.keys()),
                            'validation_indices': list(self.validation_data.keys()),
                            'lr_history': lr_history,
                            'metrics': {
                                'episode_accuracies': {
                                    'x': self.plot_episode_xvalues,
                                    'y': self.plot_episode_yvalues
                                },
                                'epoch_losses': {
                                    'x': self.plot_epoch_xvalues,
                                    'y': self.plot_epoch_yvalues
                                }
                            },
                            'episode_history': self.episode_history
                        }
                        self.checkpoint_manager.save_checkpoint(state)
                    except Exception as checkpoint_error:
                        print(f"Warning: Failed to save episode checkpoint: {str(checkpoint_error)}")

                try:
                    self._save_episode_annotations_csv()
                except Exception as csv_error:
                    print(f"Warning: Failed to save episode CSV: {str(csv_error)}")

                self.episode += 1
                
                result = {
                    "status": "success",
                    "metrics": episode_metrics,
                    "final_val_acc": best_val_acc
                }
                
                if evaluation_data:
                    result["evaluation_data"] = evaluation_data
                    print("Returning episode result with evaluation data")
                else:
                    result["next_batch"] = next_batch
                    print("Returning episode result with next batch")
                    
                return result
                    
            except Exception as e:
                raise ValueError(f"Error selecting next batch: {str(e)}")
                
        except Exception as e:
            print(f"Error in train_episode: {str(e)}")
            raise

    def load_state(self, checkpoint_path: str = None):
        """Load complete model and training state"""
        if not self.checkpoint_manager:
            raise ValueError("Checkpoint manager not initialized")
            
        checkpoint = self.checkpoint_manager.load_checkpoint(
            self.model, 
            self.optimizer, 
            self.scheduler,
            checkpoint_path
        )
        
        if checkpoint:

            self.episode = checkpoint['episode']
            self.best_val_acc = checkpoint['best_val_acc']
            self.training_config = checkpoint['training_config']
            
            self.restore_data_splits(
                checkpoint['labeled_indices'],
                checkpoint['unlabeled_indices'],
                checkpoint['validation_indices']
            )
            
            metrics = checkpoint['metrics']
            self.plot_episode_xvalues = metrics['episode_accuracies']['x']
            self.plot_episode_yvalues = metrics['episode_accuracies']['y']
            self.plot_epoch_xvalues = metrics['epoch_losses']['x']
            self.plot_epoch_yvalues = metrics['epoch_losses']['y']
            
            self.lr_config = checkpoint['lr_config']
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

            self.episode_history = checkpoint['episode_history']
            
        return checkpoint

    def get_active_learning_batch(self, strategy: str, batch_size: int):
        """Enhanced active learning sampling with multiple strategies"""
        if strategy not in ["entropy", "margin", "least_confidence", "diversity", "random"]:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
            
        if strategy == "random":
            return self._random_sampling(batch_size)
            
        uncertainties = []
        self.model.eval()
        
        with torch.no_grad():
            for img_id, img_tensor in self.unlabeled_data.items():
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                uncertainty = self._compute_uncertainty(probs, strategy, img_tensor)
                uncertainties.append((img_id, uncertainty))
                
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected_batch = uncertainties[:batch_size]
        
        return self._prepare_batch_info(selected_batch)

    def _compute_uncertainty(self, probs: torch.Tensor, strategy: str, features: torch.Tensor = None) -> float:
        """
        Compute uncertainty score based on chosen strategy
        Args:
            probs: softmax probabilities from model
            strategy: sampling strategy to use
            features: feature representations (only needed for diversity strategy)
        Returns:
            uncertainty score between 0 and 1
        """
        try:
            if strategy == "least_confidence":

                return float(1 - torch.max(probs).item())
                
            elif strategy == "margin":

                sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                return float(sorted_probs[0][0] - sorted_probs[0][1]).item()
                
            elif strategy == "entropy":

                eps = 1e-10
                probs = probs.clamp(min=eps, max=1-eps)
                entropy = -(probs * torch.log(probs)).sum(dim=1)
                return float(entropy.item())
                
            elif strategy == "diversity":
                if features is None:
                    raise ValueError("Features required for diversity sampling")

                return self._compute_diversity_score(features)
                
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
                
        except Exception as e:
            print(f"Error computing uncertainty: {str(e)}")
            return 0.0

    def _compute_diversity_score(self, features: torch.Tensor) -> float:
        """
        Compute diversity score based on feature space distance from labeled pool
        """
        if len(self.labeled_data) == 0:
            return 1.0
            
        try:

            labeled_features = []
            self.model.eval()
            with torch.no_grad():
                for img_tensor, _ in self.labeled_data.values():
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)

                    feat = self._get_features(img_tensor)
                    labeled_features.append(feat)
                    
            labeled_features = torch.cat(labeled_features, dim=0)
            
            distances = torch.cdist(features, labeled_features)
            diversity_score = distances.mean().item()
            
            return min(1.0, diversity_score / 10.0)
            
        except Exception as e:
            print(f"Error computing diversity score: {str(e)}")
            return 1.0

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model's penultimate layer"""
        if not hasattr(self.model, 'get_features'):

            original_forward = self.model.forward
            
            def get_features(self, x: torch.Tensor) -> torch.Tensor:

                original_fc = self.fc
                self.fc = nn.Identity()
                features = self(x)
                self.fc = original_fc
                return features
                
            self.model.get_features = types.MethodType(get_features, self.model)
        
        return self.model.get_features(x)

    def _random_sampling(self, batch_size: int):
        """Implement random sampling strategy"""
        available_ids = list(self.unlabeled_data.keys())
        selected_ids = np.random.choice(
            available_ids, 
            size=min(batch_size, len(available_ids)), 
            replace=False
        )
        return [(id, 0.0) for id in selected_ids]

    def _prepare_batch_info(self, selected_batch):
        """Prepare detailed batch information including predictions"""
        batch_info = []
        for img_id, uncertainty in selected_batch:

            img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
            predictions = [
                {"label": f"Label {i}", "confidence": float(p)} 
                for i, p in enumerate(probs[0])
            ]
            
            batch_info.append({
                "image_id": img_id,
                "uncertainty": uncertainty,
                "predictions": predictions
            })
            
        return batch_info

    def save_checkpoint(self):
        """Save episode checkpoint"""
        checkpoint = {
            'episode': self.episode,
            'model_state': self.best_model_state,
            'best_val_acc': self.best_val_acc,
            'metrics': {
                'episode_accuracies': {
                    'x': self.plot_episode_xvalues,
                    'y': self.plot_episode_yvalues
                },
                'episode_history': self.episode_history
            }
        }
        
        checkpoint_path = os.path.join(
            self.output_dir, 
            f'checkpoint_ep{self.episode}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

    def get_validation_status(self):
        """Get validation set labeling status"""
        total = len(self.validation_data)
        labeled = len([1 for _, label in self.validation_data.values() if label is not None])
        return {
            "total": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
            "percent_labeled": (labeled / total * 100) if total > 0 else 0
        }
    
    def adapt_transformer_to_resnet(vit_state_dict, resnet_model, num_classes=None):
        """
        Adapt a Vision Transformer (ViT) model to a ResNet architecture
        by transferring compatible weights and knowledge.
        
        Args:
            vit_state_dict: State dict from a Vision Transformer model
            resnet_model: The target ResNet model
            num_classes: Number of output classes (if known)
        
        Returns:
            Tuple of (success flag, message)
        """

        if num_classes is not None:

            in_features = resnet_model.fc.in_features
            resnet_model.fc = torch.nn.Linear(in_features, num_classes)
        
        transferred_knowledge = {
            "visual_features": False,
            "classification_head": False
        }
        
        if 'patch_embed.proj.weight' in vit_state_dict:

            patch_weights = vit_state_dict['patch_embed.proj.weight']
            
            try:
                first_conv = None

                if hasattr(resnet_model, 'conv1'):
                    first_conv = resnet_model.conv1
                
                if first_conv is not None:

                    if patch_weights.shape[0] == first_conv.weight.shape[0]:

                        new_weights = torch.nn.functional.interpolate(
                            patch_weights, 
                            size=first_conv.weight.shape[2:],
                            mode='bilinear'
                        )
                        first_conv.weight.data = new_weights
                        transferred_knowledge["visual_features"] = True
            except Exception as e:
                print(f"Could not transfer patch embedding knowledge: {e}")
        
        if 'head.weight' in vit_state_dict and hasattr(resnet_model, 'fc'):
            try:
                vit_head_weight = vit_state_dict['head.weight']
                vit_head_bias = vit_state_dict.get('head.bias', None)
                
                if vit_head_weight.shape[0] == resnet_model.fc.weight.shape[0]:

                    if vit_head_weight.shape[1] != resnet_model.fc.weight.shape[1]:

                        projection = torch.zeros(
                            resnet_model.fc.weight.shape[1],
                            vit_head_weight.shape[1]
                        )
                        
                        min_dim = min(projection.shape[0], projection.shape[1])
                        projection[:min_dim, :min_dim] = torch.eye(min_dim)
                        
                        new_weights = torch.matmul(vit_head_weight, projection.t())
                        resnet_model.fc.weight.data = new_weights
                    else:

                        resnet_model.fc.weight.data = vit_head_weight
                    
                    if vit_head_bias is not None and hasattr(resnet_model.fc, 'bias'):
                        resnet_model.fc.bias.data = vit_head_bias
                    
                    transferred_knowledge["classification_head"] = True
            except Exception as e:
                print(f"Could not transfer classification head knowledge: {e}")
        
        if any(transferred_knowledge.values()):
            return True, f"Adapted transformer model to ResNet. Transferred: {', '.join([k for k, v in transferred_knowledge.items() if v])}"
        else:
            return False, "Could not transfer knowledge from transformer to ResNet. Will only use initialization."

    def adapt_pretrained_model(self, model_state, freeze_layers=True, adaptation_layers=None):
        """
        Adapt a pre-trained model for active learning by optionally freezing layers
        and preparing the model for fine-tuning.
        
        Args:
            model_state: State dict of the pretrained model
            freeze_layers: Whether to freeze early layers
            adaptation_layers: List of layer names to specifically adapt
        """
        try:

            is_vit = any(k in ['cls_token', 'pos_embed', 'patch_embed'] for k in model_state.keys())
        
            if is_vit:
                print("Detected ViT model (like RETFound). Performing specialized adaptation...")
                
                keys_to_remove = [k for k in model_state.keys() if 'head' in k.lower()]
                for key in keys_to_remove:
                    print(f"Removing original head layer: {key}")
                    del model_state[key]
                
                missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
                print(f"Loaded ViT weights. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                
                if freeze_layers:
                    for name, param in self.model.named_parameters():
                        if 'classifier' not in name and 'head' not in name:
                            param.requires_grad = False
                            
                    print("Froze ViT feature extraction layers, keeping classifier trainable")
                
                return True
            
            else:
                if hasattr(self.model, 'load_state_dict'):

                    try:
                        self.model.load_state_dict(model_state, strict=False)
                        print("Loaded pretrained model weights (non-strict)")
                    except Exception as e:
                        print(f"Error loading model directly: {str(e)}")
                        
                        fixed_state_dict = {}
                        for k, v in model_state.items():

                            if k.startswith('module.') and not any(key.startswith('module.') for key in self.model.state_dict()):
                                fixed_state_dict[k[7:]] = v
                            elif not k.startswith('module.') and any(key.startswith('module.') for key in self.model.state_dict()):
                                fixed_state_dict['module.' + k] = v
                            else:
                                fixed_state_dict[k] = v
                        
                        missing_keys, unexpected_keys = self.model.load_state_dict(fixed_state_dict, strict=False)
                        print(f"Loaded with key fixing. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                else:
                    print("Model doesn't have load_state_dict method")
                    return False
            
            if freeze_layers:

                if isinstance(self.model, (torch.nn.Module)):
                    for name, param in self.model.named_parameters():

                        if 'fc' not in name and 'classifier' not in name:
                            param.requires_grad = False
                        else:
                            print(f"Keeping {name} trainable")
                            
                print("Early layers frozen for transfer learning")
            
            if adaptation_layers:

                for layer_name in adaptation_layers:
                    if hasattr(self.model, layer_name):
                        layer = getattr(self.model, layer_name)
                        if layer_name == 'fc' and isinstance(layer, torch.nn.Linear):

                            in_features = layer.in_features
                            out_features = layer.out_features
                            dropout_layer = torch.nn.Dropout(0.5)
                            new_fc = torch.nn.Sequential(
                                dropout_layer,
                                torch.nn.Linear(in_features, out_features)
                            )
                            setattr(self.model, layer_name, new_fc)
                            print(f"Added dropout to {layer_name}")
                            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config['learning_rate']
            )
            
            return True
        
        except Exception as e:
            print(f"Error adapting pretrained model: {str(e)}")
            return False

class CheckpointManager:
    """Manages model checkpointing and state restoration"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, state, is_best=False):
        """Save checkpoint with correct episode numbering and better error handling"""
        try:

            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            episode = state.get('episode', 0)
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{episode:03d}.pt')
            
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
                
            safe_state = {
                'episode': episode,
                'model_state': state.get('model_state'),
                'best_val_acc': state.get('best_val_acc', 0),
                'training_config': state.get('training_config', {}),
                'labeled_indices': state.get('labeled_indices', []),
                'unlabeled_indices': state.get('unlabeled_indices', []),
                'validation_indices': state.get('validation_indices', []),
                'metrics': state.get('metrics', {}),
                'episode_history': state.get('episode_history', [])
            }
            
            if 'optimizer_state' in state and state['optimizer_state']:
                safe_state['optimizer_state'] = state['optimizer_state']
            
            if 'scheduler_state' in state and state['scheduler_state']:
                safe_state['scheduler_state'] = state['scheduler_state']
            else:
                safe_state['scheduler_state'] = {}
            
            try:
                torch.save(safe_state, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
                
                if is_best:
                    torch.save(safe_state, best_path)
                    print(f"Best model saved to: {best_path}")
                
                return checkpoint_path
                
            except Exception as save_error:
                print(f"Error saving checkpoint file: {str(save_error)}")

                alternative_path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{episode}_{int(time.time())}.pt')
                try:
                    torch.save(safe_state, alternative_path)
                    print(f"Checkpoint saved to alternative path: {alternative_path}")
                    return alternative_path
                except Exception as alt_error:
                    print(f"Alternative save also failed: {str(alt_error)}")
                    raise save_error
            
        except Exception as e:
            print(f"Error in save_checkpoint: {str(e)}")

            print("Warning: Checkpoint save failed, but continuing training...")
            return None
            
    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        """Load model checkpoint with safe handling of missing components"""
        try:
            if checkpoint_path is None:

                checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_ep*.pt'))
                if not checkpoints:
                    print("No checkpoints found")
                    return None
                checkpoint_path = max(checkpoints, key=os.path.getctime)
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                return None
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path, weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state'])
                    print("Model state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load model state: {e}")

                    try:
                        missing, unexpected = model.load_state_dict(checkpoint['model_state'], strict=False)
                        print(f"Model loaded with missing keys: {missing}, unexpected keys: {unexpected}")
                    except Exception as e2:
                        print(f"Error: Could not load model state even with strict=False: {e2}")
                        return None
            else:
                print("Warning: No model state found in checkpoint")
            
            if optimizer is not None and 'optimizer_state' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("Optimizer state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            elif optimizer is not None:
                print("No optimizer state found in checkpoint")
            
            if scheduler is not None and 'scheduler_state' in checkpoint and checkpoint['scheduler_state']:
                try:

                    scheduler_state = checkpoint['scheduler_state']
                    if scheduler_state and isinstance(scheduler_state, dict) and scheduler_state:

                        if hasattr(scheduler, 'load_state_dict'):
                            scheduler.load_state_dict(scheduler_state)
                            print("Scheduler state loaded successfully")
                        elif hasattr(scheduler, 'scheduler') and hasattr(scheduler.scheduler, 'load_state_dict'):

                            if 'scheduler_state' in scheduler_state:
                                scheduler.scheduler.load_state_dict(scheduler_state['scheduler_state'])
                            else:
                                scheduler.scheduler.load_state_dict(scheduler_state)
                            print("Wrapped scheduler state loaded successfully")
                        else:
                            print("Warning: Scheduler does not support state loading")
                    else:
                        print("Scheduler state is empty, skipping")
                except Exception as e:
                    print(f"Warning: Could not load scheduler state: {e}")
            elif scheduler is not None:
                print("No scheduler state found in checkpoint or scheduler_state is empty")
            
            print(f"Checkpoint loaded from episode {checkpoint.get('episode', 'unknown')}")
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def get_best_model_path(self):
        """Get path to best model checkpoint"""
        return os.path.join(self.checkpoint_dir, 'model_best.pt')

class AutomatedTrainingManager:
    def __init__(self, al_manager):
        self.al_manager = al_manager
        self.is_training = False
        self.stop_requested = False
        self.current_batch_labeled_count = 0
        self.current_batch_size = 0
        self.batch_complete = asyncio.Event()
        self.current_batch = []
        self.last_training_start = None
        self.training_timeout = 300
        
        self.training_config = {
            'sampling_strategy': 'least_confidence',
            'batch_size': 16,
            'epochs': 5,
            'learning_rate': 0.001
        }
        
        self.min_required_samples = 10

    def update_config(self, config):
        """Update training configuration safely"""
        if not isinstance(config, dict):
            print(f"Warning: config is not a dictionary: {type(config)}")
            return
            
        if 'sampling_strategy' in config and isinstance(config['sampling_strategy'], str):
            self.training_config['sampling_strategy'] = config['sampling_strategy']
            
        if 'batch_size' in config:
            try:
                batch_size = int(config['batch_size'])
                if batch_size > 0:
                    self.training_config['batch_size'] = batch_size
            except (ValueError, TypeError):
                print(f"Invalid batch_size: {config['batch_size']}")
                
        if 'epochs' in config:
            try:
                epochs = int(config['epochs'])
                if epochs > 0:
                    self.training_config['epochs'] = epochs
            except (ValueError, TypeError):
                print(f"Invalid epochs: {config['epochs']}")
                
        if 'learning_rate' in config:
            try:
                lr = float(config['learning_rate'])
                if lr > 0:
                    self.training_config['learning_rate'] = lr
            except (ValueError, TypeError):
                print(f"Invalid learning_rate: {config['learning_rate']}")
                
        print(f"Updated training config: {self.training_config}")
    
    def check_training_state(self):
        """Check if training state is stuck and reset if necessary"""
        if self.is_training and self.last_training_start:
            time_elapsed = time.time() - self.last_training_start
            if time_elapsed > self.training_timeout:
                print(f"Training state was stuck for {time_elapsed:.0f} seconds. Resetting...")
                self.is_training = False
                self.last_training_start = None
                return True
        return False

    def on_label_submitted(self):
        """Enhanced label submission handling"""
        self.current_batch_labeled_count += 1
        print(f"\n=== Automated Training Status ===")
        print(f"Labels in batch: {self.current_batch_labeled_count}/{self.current_batch_size}")
        print(f"Total labeled samples: {len(self.al_manager.labeled_data)}")
        
        self.check_training_state()
        
        batch_is_complete = self.current_batch_labeled_count >= self.current_batch_size
        has_enough_samples = len(self.al_manager.labeled_data) >= self.min_required_samples
        
        if (batch_is_complete and has_enough_samples and not self.is_training):
            print("Batch is complete! Starting training cycle...")
            asyncio.create_task(self._train_and_get_next_batch())
        else:
            if self.is_training:
                print("Training already in progress...")
            elif not batch_is_complete:
                print(f"Waiting for more labels before training ({self.current_batch_labeled_count}/{self.current_batch_size})")
            elif not has_enough_samples:
                print(f"Need more samples (have {len(self.al_manager.labeled_data)}, need {self.min_required_samples})")

    def update_config(self, config):
        """Update training configuration"""
        self.training_config.update(config)

    async def start_automated_training(self, config: dict):
        """Start automated training with improved state management"""
        try:
            print(f"Received config: {config}")
            
            if self.check_training_state():
                print("Reset stuck training state")
            
            if self.is_training:
                return {"status": "already_running"}
                
            self.training_config = {
                'epochs': int(config['epochs']),
                'batch_size': int(config['batch_size']),
                'sampling_strategy': str(config['sampling_strategy']),
                'learning_rate': float(config.get('learning_rate', 0.001))
            }
            
            print(f"Processed config: {self.training_config}")
            
            self.is_training = True
            self.stop_requested = False
            self.last_training_start = time.time()
            
            if not self.current_batch:
                self.current_batch = self.al_manager.get_next_batch(
                    strategy=self.training_config['sampling_strategy'],
                    batch_size=self.training_config['batch_size']
                )
                self.current_batch_size = len(self.current_batch)
                self.current_batch_labeled_count = 0
                
            return {
                "status": "success",
                "message": "Started automated training",
                "batch_size": self.current_batch_size,
                "config": self.training_config
            }
                
        except Exception as e:
            self.is_training = False
            self.last_training_start = None
            print(f"Error in start_automated_training: {str(e)}")
            raise

    async def _train_and_get_next_batch(self):
        """Training cycle with improved state management and evaluation"""
        try:
            print("\n=== Starting Training Cycle ===")
            self.last_training_start = time.time()
            
            if len(self.al_manager.labeled_data) < self.min_required_samples:
                print(f"Insufficient labeled data ({len(self.al_manager.labeled_data)} < {self.min_required_samples})")
                self.is_training = False
                return
            
            training_result = self.al_manager.train_episode(
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                learning_rate=self.training_config['learning_rate']
            )
            
            print(f"Training completed. Validation accuracy: {self.al_manager.best_val_acc:.2f}%")
            
            if 'evaluation_data' in training_result and training_result['evaluation_data']:
                print("Evaluation data found - evaluation screen should be shown")

                self.is_training = False
                self.last_training_start = None
                return {
                    "status": "success",
                    "evaluation_available": True,
                    "evaluation_data": training_result['evaluation_data'],
                    "training_result": training_result,
                    "validation_accuracy": self.al_manager.best_val_acc
                }
            
            if not self.stop_requested:
                print("No evaluation data - getting next batch...")
                self.current_batch = self.al_manager.get_next_batch(
                    strategy=self.training_config['sampling_strategy'],
                    batch_size=self.training_config['batch_size']
                )
                self.current_batch_size = len(self.current_batch)
                self.current_batch_labeled_count = 0
                
                self.al_manager.current_batch = [x["image_id"] for x in self.current_batch]
                self.is_training = False
                self.last_training_start = None
                
                return {
                    "status": "success",
                    "new_batch_available": True,
                    "batch_size": self.current_batch_size,
                    "training_result": training_result,
                    "validation_accuracy": self.al_manager.best_val_acc
                }
                    
        except Exception as e:
            print(f"Training error: {str(e)}")
            self.is_training = False
            self.last_training_start = None
            raise
    
    async def get_new_batch(self):
        """Manually request a new batch"""
        if not self.is_training:
            raise HTTPException(status_code=400, detail="Automated training not active")
        
        try:
            self.current_batch = self.al_manager.get_next_batch(
                strategy=self.training_config['sampling_strategy'],
                batch_size=self.training_config['batch_size']
            )
            self.current_batch_size = len(self.current_batch)
            self.current_batch_labeled_count = 0
            self.batch_complete.clear()
            
            return {
                "status": "success",
                "batch_size": self.current_batch_size
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def stop_automated_training(self):
        """Stop the automated training cycle"""
        self.stop_requested = True
        self.batch_complete.set()
            
    def get_training_status(self):
        """Get current automated training status"""
        return {
            "is_training": self.is_training,
            "current_episode": self.al_manager.episode,
            "labeled_count": len(self.al_manager.labeled_data),
            "unlabeled_count": len(self.al_manager.unlabeled_data),
            "current_batch": {
                "labeled": self.current_batch_labeled_count,
                "total": self.current_batch_size
            },
            "config": {
                "sampling_strategy": self.al_manager.config.get('sampling_strategy'),
                "batch_size": self.al_manager.config.get('batch_size'),
                "epochs": self.al_manager.config.get('epochs'),
                "learning_rate": self.al_manager.config.get('learning_rate')
            }
        }

    async def train_current_model(self):
        """Train the current model and collect metrics"""
        training_result = self.al_manager.train_episode(
            epochs=self.al_manager.config.get('epochs', 10),
            batch_size=self.al_manager.config.get('batch_size', 16),
            learning_rate=self.al_manager.config.get('learning_rate', 0.001)
        )
        d
        metrics = {
            'training_metrics': training_result,
            'episode_accuracies': {
                'x': self.al_manager.plot_episode_xvalues,
                'y': self.al_manager.plot_episode_yvalues
            },
            'current_epoch_losses': {
                'x': self.al_manager.plot_epoch_xvalues,
                'y': self.al_manager.plot_epoch_yvalues
            },
            'validation_accuracy': self.al_manager.best_val_acc,
            'episode': self.al_manager.episode
        }
        
        self.current_metrics = metrics
        return metrics

class LRSchedulerManager:
    """Manages different learning rate scheduling strategies"""
    
    def __init__(self, optimizer, strategy="plateau", **kwargs):
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_lr = kwargs.get('initial_lr', 0.001)
        self.history = []
        self.scheduler = self._create_scheduler(**kwargs)
        
    def get_lr(self):
        """Get current learning rate"""
        try:
            return self.optimizer.param_groups[0]['lr']
        except Exception as e:
            print(f"Error getting learning rate: {str(e)}")
            return self.initial_lr
    
    def get_status(self):
        """Get current scheduler status"""
        return {
            "strategy": self.strategy,
            "current_lr": self.get_lr(),
            "history": self.history,
            "initial_lr": self.initial_lr
        }
        
    def _create_scheduler(self, **kwargs):
        """Create scheduler based on strategy"""
        if self.strategy == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 5),
                verbose=kwargs.get('verbose', True),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif self.strategy == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('min_lr', 0)
            )
        elif self.strategy == "warmup":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=kwargs.get('max_lr', 0.1),
                epochs=kwargs.get('epochs', 30),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100),
                pct_start=kwargs.get('warmup_pct', 0.3)
            )
        elif self.strategy == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler strategy: {self.strategy}")
            
    def step(self, metric=None):
        """Update learning rate based on metric or epoch"""
        current_lr = self.get_lr()
        
        if self.strategy == "plateau":
            if metric is None:
                raise ValueError("Metric required for ReduceLROnPlateau scheduler")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
            
        new_lr = self.get_lr()
        
        self.history.append({
            'old_lr': current_lr,
            'new_lr': new_lr,
            'metric': metric
        })
        
        return new_lr
        
    def reset(self):
        """Reset learning rate to initial value"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
            
    def state_dict(self):
        """Get scheduler state for checkpointing"""
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'strategy': self.strategy,
            'history': self.history
        }
        
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.strategy = state_dict['strategy']
        self.history = state_dict['history']

app = FastAPI()
al_manager = ActiveLearningManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def adapt_pretrained_model(self, model_state, freeze_layers=True, adaptation_layers=None):
    """
    Adapt a pre-trained model for active learning by optionally freezing layers
    and preparing the model for fine-tuning.
    
    Args:
        model_state: State dict of the pretrained model
        freeze_layers: Whether to freeze early layers
        adaptation_layers: List of layer names to specifically adapt
    """
    try:

        if hasattr(self.model, 'load_state_dict'):

            try:
                self.model.load_state_dict(model_state, strict=False)
                print("Loaded pretrained model weights (non-strict)")
            except Exception as e:
                print(f"Error loading model directly: {str(e)}")
                
                fixed_state_dict = {}
                for k, v in model_state.items():

                    if k.startswith('module.') and not any(key.startswith('module.') for key in self.model.state_dict()):
                        fixed_state_dict[k[7:]] = v
                    elif not k.startswith('module.') and any(key.startswith('module.') for key in self.model.state_dict()):
                        fixed_state_dict['module.' + k] = v
                    else:
                        fixed_state_dict[k] = v
                
                missing_keys, unexpected_keys = self.model.load_state_dict(fixed_state_dict, strict=False)
                print(f"Loaded with key fixing. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        else:
            print("Model doesn't have load_state_dict method")
            return False
        
        if freeze_layers:

            if isinstance(self.model, (torch.nn.Module)):
                for name, param in self.model.named_parameters():

                    if 'fc' not in name and 'classifier' not in name:
                        param.requires_grad = False
                    else:
                        print(f"Keeping {name} trainable")
                        
            print("Early layers frozen for transfer learning")
        
        if adaptation_layers:

            for layer_name in adaptation_layers:
                if hasattr(self.model, layer_name):
                    layer = getattr(self.model, layer_name)
                    if layer_name == 'fc' and isinstance(layer, torch.nn.Linear):

                        in_features = layer.in_features
                        out_features = layer.out_features
                        dropout_layer = torch.nn.Dropout(0.5)
                        new_fc = torch.nn.Sequential(
                            dropout_layer,
                            torch.nn.Linear(in_features, out_features)
                        )
                        setattr(self.model, layer_name, new_fc)
                        print(f"Added dropout to {layer_name}")
        
        return True
    
    except Exception as e:
        print(f"Error adapting pretrained model: {str(e)}")
        return False

@app.post("/adapt-pretrained-model")
async def adapt_pretrained_model(
    freeze_layers: bool = Form(True),
    adaptation_type: str = Form("full_finetune")
):
    """
    Adapt a previously imported model for active learning
    """
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model has been imported yet")
        
        model_state = al_manager.model.state_dict()
        
        adaptation_layers = None
        if adaptation_type == "last_layer":
            adaptation_layers = ["fc"]
        elif adaptation_type == "mid_layers":
            adaptation_layers = ["layer4", "fc"]
        
        success = al_manager.adapt_pretrained_model(
            model_state=model_state,
            freeze_layers=freeze_layers,
            adaptation_layers=adaptation_layers
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to adapt model")
        
        return {
            "status": "success",
            "message": "Model adapted successfully",
            "adaptation_type": adaptation_type,
            "freeze_layers": freeze_layers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")

@app.post("/init")
async def initialize_project(request: ProjectInit):
    """Initialize new active learning project with improved error handling"""
    try:
        print(f"Initializing project with config: {request}")
        
        training_config = {
            'sampling_strategy': request.sampling_strategy if hasattr(request, 'sampling_strategy') else 'least_confidence',
            'batch_size': request.batch_size if hasattr(request, 'batch_size') else 16,
            'epochs': request.epochs if hasattr(request, 'epochs') else 10,
            'learning_rate': request.learning_rate if hasattr(request, 'learning_rate') else 0.001,
        }
        
        result = al_manager.initialize_project(
            project_name=request.project_name,
            model_name=request.model_type,
            num_classes=request.num_classes,
            config=training_config
        )
        
        return {
            "status": "success",
            "output_dir": result["output_dir"],
            "config": training_config
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"Error initializing project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_initial_data(
    files: List[UploadFile],
    val_split: float = None,
    initial_labeled_ratio: float = None
):
    """Upload and split initial dataset"""
    try:
        if val_split is not None:
            al_manager.config['val_split'] = val_split
        if initial_labeled_ratio is not None:
            al_manager.config['initial_labeled_ratio'] = initial_labeled_ratio
            
        return await al_manager.add_initial_data(files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-batch")
async def get_batch(request: BatchRequest):
    try:
        if not hasattr(al_manager, 'model') or al_manager.model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not initialized. Please initialize project first."
            )
            
        if len(al_manager.unlabeled_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="No unlabeled data available"
            )
        
        requested_batch_size = int(request.batch_size)

        # Save UI-selected annotation batch size for future episodes
        al_manager.config["batch_size"] = requested_batch_size
        automated_trainer.training_config["batch_size"] = requested_batch_size
        automated_trainer.training_config["sampling_strategy"] = request.strategy

        batch = al_manager.get_next_batch(
            strategy=request.strategy,
            batch_size=requested_batch_size
        )

        automated_trainer.current_batch_size = len(batch)
        automated_trainer.current_batch_labeled_count = 0

        return batch
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-label")
async def submit_label(submission: LabelSubmission):
    """Submit label for an image with enhanced logging"""
    try:
        print(f"\n=== Label Submission ===")
        print(f"Image ID: {submission.image_id}, Label: {submission.label}")
        
        result = al_manager.submit_label(
            image_id=submission.image_id,
            label=submission.label
        )

        total = automated_trainer.current_batch_size
        automated_trainer.current_batch_labeled_count += 1
        labeled = automated_trainer.current_batch_labeled_count % total if total > 0 else automated_trainer.current_batch_labeled_count
        if labeled == 0:
            labeled = total
        print(f"Current batch progress: {labeled}/{total}")

        return {
            **result,
            "batch_complete": total > 0 and labeled >= total,
            "is_training": automated_trainer.is_training,
            "current_progress": {
                "labeled": labeled,
                "total": total
            }
        }
    except Exception as e:
        print(f"Error in submit_label: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.001
):
    """Train/retrain model on labeled data"""
    try:
        if not al_manager.model:
            raise HTTPException(
                status_code=400, 
                detail="Model not initialized. Please initialize project first."
            )
            
        if len(al_manager.labeled_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="No labeled data available for training"
            )
            
        return al_manager.train(epochs, batch_size, learning_rate)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Training endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get current status of active learning process"""
    return {
        "project_name": al_manager.project_name,
        "current_episode": al_manager.episode,
        "labeled_count": len(al_manager.labeled_data),
        "unlabeled_count": len(al_manager.unlabeled_data),
        "validation_count": len(al_manager.validation_data),
        "current_batch_size": len(al_manager.current_batch),
    }

@app.get("/metrics")
async def get_metrics():
    try:

        episode_f1_scores = {
            "x": [ep["episode"] for ep in al_manager.episode_history if "f1_score" in ep],
            "y": [ep["f1_score"] for ep in al_manager.episode_history if "f1_score" in ep],
        }
        metrics = {
            "best_val_acc": al_manager.best_val_acc,
            "current_episode": al_manager.episode,
            "episode_accuracies": {
                "x": al_manager.plot_episode_xvalues,
                "y": al_manager.plot_episode_yvalues
            },
            "episode_f1_scores": episode_f1_scores,
            "current_epoch_losses": {
                "x": al_manager.plot_epoch_xvalues,
                "y": al_manager.plot_epoch_yvalues
            }
        }
        print("Sending metrics:", metrics)
        return metrics
    except Exception as e:
        print(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/{image_id}")
async def get_image(image_id: int):
    """Get image data for display with improved error handling and debugging"""
    try:
        print(f"Serving image {image_id}")
        
        if not isinstance(image_id, int) or image_id < 0:
            print(f"Invalid image ID format: {image_id}")
            raise HTTPException(status_code=400, detail="Invalid image ID format")
        
        tensor = None
        location = None
        
        if image_id in al_manager.unlabeled_data:
            tensor = al_manager.unlabeled_data[image_id]
            location = "unlabeled_data"
        elif image_id in al_manager.labeled_data:
            tensor = al_manager.labeled_data[image_id][0]
            location = "labeled_data"
        elif image_id in al_manager.validation_data:
            tensor = al_manager.validation_data[image_id][0]
            location = "validation_data"
        
        if tensor is None:
            print(f"Image {image_id} not found in any dataset")
            print(f"Available IDs in unlabeled: {list(al_manager.unlabeled_data.keys())[:5]}...")
            print(f"Available IDs in labeled: {list(al_manager.labeled_data.keys())[:5]}...")
            print(f"Available IDs in validation: {list(al_manager.validation_data.keys())[:5]}...")
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found in any dataset")
            
        print(f"Found image {image_id} in {location}")
            
        try:
            tensor = tensor.cpu()
        except Exception as e:
            print(f"Error moving tensor to CPU: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image tensor")
        
        print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        try:

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean
                
            img_array = tensor.numpy().transpose(1, 2, 0)
            img_array = np.clip(img_array, 0, 1)
            img_array = (img_array * 255).astype(np.uint8)
            
            print(f"Image array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")
            
            img = Image.fromarray(img_array)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            headers = {
                'Cache-Control': 'public, max-age=31536000',
                'ETag': f'"{hash(img_byte_arr)}"'
            }
            
            return Response(
                content=img_byte_arr, 
                media_type="image/png",
                headers=headers
            )
        except Exception as e:
            error_msg = f"Error converting tensor to image: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:

        raise
    except Exception as e:
        error_msg = f"Error serving image {image_id}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def get_model_num_classes(model):
    """
    Safely get the number of output classes from different model architectures
    """
    try:

        if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            return model.fc.out_features
        
        elif hasattr(model, 'classifier'):
            if hasattr(model.classifier, 'out_features'):
                return model.classifier.out_features
            elif isinstance(model.classifier, nn.Sequential):

                for layer in reversed(model.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
        
        elif hasattr(model, 'head') and hasattr(model.head, 'out_features'):
            return model.head.out_features
        
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            return model.classifier.out_features
        
        elif isinstance(model, nn.Sequential):
            for layer in reversed(model):
                if isinstance(layer, nn.Linear):
                    return layer.out_features
        
        print(f"Warning: Could not determine num_classes for model type {type(model)}")
        print(f"Model structure: {model}")
        
        return 2
        
    except Exception as e:
        print(f"Error getting num_classes: {str(e)}")
        return 2


@app.post("/cleanup-project")
async def cleanup_project():
    """
    Delete on-disk temp files for the current project (checkpoints, episode CSVs,
    progress plots) without touching in-memory state — model, labeled/unlabeled/
    validation data, and episode_history all stay exactly as they are.

    Call this from a dedicated 'Clear temp files' button, independent of export.
    Safe to call any time there's an active project; recreates an empty run
    folder afterward so training can continue immediately.
    """
    try:
        if not al_manager.project_name:
            raise HTTPException(status_code=400, detail="No active project to clean up")

        removed = []

        if os.path.exists("current_model_inspection.json"):
            os.remove("current_model_inspection.json")
            removed.append("current_model_inspection.json")

        project_root = os.path.join("output", al_manager.project_name)
        if os.path.isdir(project_root):
            shutil.rmtree(project_root, ignore_errors=True)
            removed.append(project_root)

        # recreate a fresh run folder immediately so the next save_checkpoint() /
        # progress plot write doesn't fail with a missing directory
        al_manager.output_dir = os.path.join(
            "output", al_manager.project_name, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(al_manager.output_dir, exist_ok=True)
        al_manager.checkpoint_manager = CheckpointManager(al_manager.output_dir)

        print(f"Cleaned up temp files for project '{al_manager.project_name}': {removed}")

        return {
            "status": "success",
            "removed": removed,
            "new_output_dir": al_manager.output_dir,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error cleaning up project: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export-project")
async def export_project():
    """Export complete project as ZIP file with model, data, and metadata"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
        
        print("Starting project export...")
        
        model_info = inspect_and_save_model_info(al_manager.model, "current_model_inspection.json")
        print("=== MODEL INSPECTION COMPLETE ===")
        print(f"Class name: {model_info.get('basic_info', {}).get('class_name', 'unknown')}")
        print(f"Detected type: {model_info.get('detection_result', {}).get('detected_type', 'unknown')}")
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            print("Created ZIP buffer...")
            
            with open("current_model_inspection.json", 'r') as f:
                zipf.writestr("model_inspection.json", f.read())
            
            # Prefer the model type stored during project initialization (training_config)
            # over the auto-detected type, since detection can't distinguish DINOv2 from ViT.
            _known_types = {'resnet18', 'resnet50', 'dinov2', 'vision-transformer', 'vit', 'efficientnet', 'custom'}
            stored_type = al_manager.training_config.get('model_type', '')
            if stored_type in _known_types:
                model_type_name = stored_type
            else:
                model_type_name = model_info.get('detection_result', {}).get('detected_type', 'custom')
                if model_type_name == 'resnet':
                    model_type_name = model_info.get('detection_result', {}).get('variant', 'resnet50')
                elif model_type_name in ('unknown', 'vision_transformer'):
                    model_type_name = 'vision-transformer'
                elif model_type_name not in _known_types:
                    model_type_name = 'custom'

            num_classes = get_model_num_classes(al_manager.model)
            is_vit = model_type_name in ('vision-transformer', 'vision_transformer', 'vit', 'dinov2')
            
            model_export = {
                'model_state': al_manager.model.state_dict(),
                'model_config': {
                    'project_name': al_manager.project_name,
                    'episode': al_manager.episode,
                    'model_type': model_type_name,
                    'model_class': model_info.get('basic_info', {}).get('class_name', 'unknown'),
                    'num_classes': num_classes,
                    'best_val_acc': al_manager.best_val_acc,
                    'is_vision_transformer': is_vit,
                    'detection_confidence': model_info.get('detection_result', {}).get('confidence', 'unknown'),
                    'detection_reasoning': model_info.get('detection_result', {}).get('reasoning', [])
                }
            }
        
            model_buffer = io.BytesIO()
            torch.save(model_export, model_buffer)
            model_buffer.seek(0)
            zipf.writestr("model.pt", model_buffer.getvalue())
            print("Added model to ZIP...")

            # csv_data = []
            
            # def get_image_info(img_id, img_tensor, label, split_type):
            #     original_path = al_manager.image_paths.get(img_id, f"image_{img_id}.jpg")
                
            #     return {
            #         'image_id': img_id,
            #         'image_path': original_path,
            #         'status': 'labeled' if label is not None else 'unlabeled',
            #         'label_index': label,
            #         'label_name': None,
            #         'split': split_type
            #     }
            
            # for img_id, (img_tensor, label) in al_manager.labeled_data.items():
            #     csv_data.append(get_image_info(img_id, img_tensor, label, 'train'))
            
            # for img_id, img_tensor in al_manager.unlabeled_data.items():
            #     csv_data.append(get_image_info(img_id, img_tensor, None, 'train'))
            
            # for img_id, (img_tensor, label) in al_manager.validation_data.items():
            #     csv_data.append(get_image_info(img_id, img_tensor, label, 'validation'))
            
            # if hasattr(al_manager, 'current_labels') and al_manager.current_labels:
            #     for item in csv_data:
            #         if item['label_index'] is not None and item['label_index'] < len(al_manager.current_labels):
            #             item['label_name'] = al_manager.current_labels[item['label_index']]
            
            # if csv_data:
            #     df = pd.DataFrame(csv_data)
            #     csv_buffer = io.StringIO()
            #     df.to_csv(csv_buffer, index=False)
            #     zipf.writestr("annotations.csv", csv_buffer.getvalue().encode('utf-8'))
            #     print("Added CSV with real paths to ZIP...")
            
            csv_data = []

            def get_prediction(img_tensor):
                """Return model prediction for one image tensor."""
                try:
                    al_manager.model.eval()
                    with torch.no_grad():
                        x = img_tensor.unsqueeze(0).to(al_manager.device)
                        outputs = al_manager.model(x)
                        probs = torch.softmax(outputs, dim=1)[0]

                        conf, pred_idx = torch.max(probs, dim=0)
                        pred_idx = int(pred_idx.item())
                        conf = float(conf.item())

                        pred_name = None
                        if hasattr(al_manager, "current_labels") and al_manager.current_labels:
                            if pred_idx < len(al_manager.current_labels):
                                pred_name = al_manager.current_labels[pred_idx]

                        return pred_idx, pred_name, conf, [float(p.item()) for p in probs]

                except Exception as e:
                    print(f"Prediction failed during export: {e}")
                    return None, None, None, None


            def get_image_info(img_id, img_tensor, label, split_type):
                original_path = al_manager.image_paths.get(img_id, f"image_{img_id}.jpg")

                pred_idx, pred_name, pred_conf, pred_probs = get_prediction(img_tensor)

                label_name = None
                if label is not None and hasattr(al_manager, "current_labels") and al_manager.current_labels:
                    if label < len(al_manager.current_labels):
                        label_name = al_manager.current_labels[label]

                return {
                    "image_id": img_id,
                    "image_path": original_path,
                    "status": "labeled" if label is not None else "unlabeled",
                    "label_index": label,
                    "label_name": label_name,
                    "split": split_type,

                    # model predictions
                    "prediction_index": pred_idx,
                    "prediction_name": pred_name,
                    "prediction_confidence": pred_conf,
                    "prediction_probabilities": json.dumps(pred_probs) if pred_probs is not None else None,
                }


            for img_id, (img_tensor, label) in al_manager.labeled_data.items():
                csv_data.append(get_image_info(img_id, img_tensor, label, "train"))

            for img_id, img_tensor in al_manager.unlabeled_data.items():
                csv_data.append(get_image_info(img_id, img_tensor, None, "train"))

            for img_id, (img_tensor, label) in al_manager.validation_data.items():
                csv_data.append(get_image_info(img_id, img_tensor, label, "validation"))

            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                zipf.writestr("annotations.csv", csv_buffer.getvalue().encode("utf-8"))
                print("Added annotations.csv with labels and model predictions to ZIP...")

            ###################

            training_config = automated_trainer.training_config if 'automated_trainer' in globals() else {}
            
            current_labels = getattr(al_manager, 'current_labels', [])
            
            if not current_labels:
                current_labels = [f"Class {i + 1}" for i in range(num_classes)]
            
            print(f"Exporting with labels: {current_labels}")
            
            metadata = {
               'project_info': {
                    'project_name': al_manager.project_name,
                    'export_timestamp': datetime.now().isoformat(),
                    'current_episode': max(0, al_manager.episode - 1),
                    'best_validation_accuracy': al_manager.best_val_acc,
                    'model_type': model_type_name,
                    'model_class': al_manager.model.__class__.__name__,
                    'is_vision_transformer': model_type_name in ('vision-transformer', 'vision_transformer', 'vit', 'dinov2'),
                    'num_classes': num_classes,
                },
                'labels': {
                    'label_names': current_labels,
                    'num_classes': len(current_labels)
                },
                'dataset_stats': {
                    'total_images': len(al_manager.labeled_data) + len(al_manager.unlabeled_data) + len(al_manager.validation_data),
                    'labeled_images': len(al_manager.labeled_data),
                    'unlabeled_images': len(al_manager.unlabeled_data),
                    'validation_images': len(al_manager.validation_data),
                    'validation_labeled': len([1 for _, label in al_manager.validation_data.values() if label is not None]),
                },
                'hyperparameters': {
                    'sampling_strategy': training_config.get('sampling_strategy', 'unknown'),
                    'batch_size': training_config.get('batch_size', 'unknown'),
                    'epochs': training_config.get('epochs', 'unknown'),
                    'learning_rate': training_config.get('learning_rate', 'unknown'),
                    'validation_split': al_manager.config.get('val_split', 0.2),
                    'initial_labeled_ratio': al_manager.config.get('initial_labeled_ratio', 0.1),
                },
                'training_metrics': {
                    'episode_accuracies': {
                        'episodes': al_manager.plot_episode_xvalues,
                        'accuracies': al_manager.plot_episode_yvalues
                    },
                    'epoch_losses': {
                        'epochs': al_manager.plot_epoch_xvalues,
                        'losses': al_manager.plot_epoch_yvalues
                    },
                    'episode_history': al_manager.episode_history
                },
                'episode_breakdown': []
            }
            
            for i, episode_data in enumerate(al_manager.episode_history):
                episode_info = {
                    'episode': i + 1,
                    'strategy_used': episode_data.get('strategy', 'unknown'),
                    'batch_size': episode_data.get('batch_size', 'unknown'),
                    'images_labeled_this_episode': episode_data.get('batch_size', 0),
                    'total_labeled_after_episode': episode_data.get('labeled_size', 0),
                    'validation_accuracy': episode_data.get('best_val_acc', 0),
                    'learning_rate': episode_data.get('learning_rate', 'unknown')
                }
                metadata['episode_breakdown'].append(episode_info)
            
            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr("metadata.json", metadata_json.encode('utf-8'))            
            print(f"Added metadata to ZIP with labels: {current_labels}")

            inference_export_status = add_inference_artifacts(
                zip_file=zipf,
                model=al_manager.model,
                model_type=model_type_name,
                label_names=current_labels,
                image_size=224,
                normalize_mean=(0.485, 0.456, 0.406),
                normalize_std=(0.229, 0.224, 0.225),
            )
            print(f"Inference export status: {inference_export_status}")

            csv_dir = os.path.join(al_manager.output_dir, 'episode_csvs')
            if os.path.exists(csv_dir):
                print("Adding episode CSV files to export...")
                for csv_file in os.listdir(csv_dir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(csv_dir, csv_file)
                        zipf.write(csv_path, os.path.join('episode_csvs', csv_file))
                        print(f"Added episode CSV: {csv_file}")
        
        zip_buffer.seek(0)
        zip_content = zip_buffer.getvalue()
        zip_filename = f"{al_manager.project_name}_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        print(f"ZIP file created successfully. Size: {len(zip_content)} bytes")
        
        return Response(
            content=zip_content,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}",
                "Content-Length": str(len(zip_content))
            }
        )
        
    except Exception as e:
        print(f"Error exporting project: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/import-model")
async def import_model(uploaded_file: UploadFile = File(...)):
    """Import a previously exported model"""
    try:

        content = await uploaded_file.read()

        if torch.cuda.is_available():
            model_data = torch.load(io.BytesIO(content), weights_only=False)
        else:
            model_data = torch.load(io.BytesIO(content), map_location=torch.device('cpu'), weights_only=False)

        config = model_data['model_config']
        training_config = config.get('training_config', {})

        if al_manager.model is None:
            num_classes = len(config.get('labels', [])) or config.get('num_classes', 2)
            al_manager.initialize_project(
                project_name=config['project_name'],
                model_name=config.get('model_type', 'resnet50'),
                num_classes=num_classes
            )
            
        al_manager.project_name = config['project_name']
        al_manager.episode = config['episode']
        
        metrics = config.get('metrics', {})
        episode_accuracies = metrics.get('episode_accuracies', {'x': [], 'y': []})
        al_manager.plot_episode_xvalues = episode_accuracies['x']
        al_manager.plot_episode_yvalues = episode_accuracies['y']
        
        al_manager.model.load_state_dict(model_data['model_state'])
        
        if 'automated_trainer' in globals():
            automated_trainer.training_config.update({
                'sampling_strategy': training_config.get('sampling_strategy', 'least_confidence'),
                'epochs': training_config.get('epochs', 5),
                'batch_size': training_config.get('batch_size', 16),
                'learning_rate': training_config.get('learning_rate', 0.001)
            })
        
        return {
            "status": "success",
            "project_name": al_manager.project_name,
            "episode": al_manager.episode,
            "training_config": training_config,
            "metrics": {
                "episode_accuracies": {
                    "x": al_manager.plot_episode_xvalues,
                    "y": al_manager.plot_episode_yvalues
                }
            }
        }
        
    except Exception as e:
        print(f"Error importing model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
def train_episode(self, epochs: int, batch_size: int, learning_rate: float):
    """Run a complete training episode with improved batch selection, checkpointing, LR scheduling, and evaluation"""
    try:
        if len(self.labeled_data) == 0:
            raise ValueError("No labeled data available for training")

        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(self.output_dir)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        scheduler_config = self.training_config.get('scheduler', {
            'strategy': 'plateau',
            'params': {
                'mode': 'max',
                'factor': 0.1,
                'patience': 5,
                'verbose': True,
                'min_lr': 1e-6
            }
        })
        
        if scheduler_config['strategy'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_config['params']
            )
        elif scheduler_config['strategy'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=scheduler_config['params'].get('min_lr', 0)
            )
        elif scheduler_config['strategy'] == 'warmup':
            steps_per_epoch = len(self.labeled_data) // batch_size
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_config['params'].get('max_lr', learning_rate * 10),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=scheduler_config['params'].get('warmup_pct', 0.3)
            )
        elif scheduler_config['strategy'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config['params'].get('step_size', max(1, epochs // 3)),
                gamma=scheduler_config['params'].get('gamma', 0.1)
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.1, patience=5, verbose=True
            )

        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0
        best_f1 = 0.0
        best_model_state = None
        lr_history = []

        for epoch in range(epochs):

            train_loss, train_acc = self.train_epoch(optimizer, criterion, batch_size)
            
            val_acc, val_f1 = self.validate_model_with_metrics()
            
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler_config['strategy'] == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            new_lr = optimizer.param_groups[0]['lr']
            lr_history.append({
                'epoch': epoch + 1,
                'old_lr': current_lr,
                'new_lr': new_lr,
                'val_acc': val_acc
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                
                try:
                    if self.checkpoint_manager:
                        state = {
                            'episode': self.episode,
                            'model_state': best_model_state,
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                            'scheduler_config': scheduler_config,
                            'best_val_acc': best_val_acc,
                            'training_config': self.training_config,
                            'labeled_indices': list(self.labeled_data.keys()),
                            'unlabeled_indices': list(self.unlabeled_data.keys()),
                            'validation_indices': list(self.validation_data.keys()),
                            'lr_history': lr_history,
                            'metrics': {
                                'episode_accuracies': {
                                    'x': self.plot_episode_xvalues,
                                    'y': self.plot_episode_yvalues
                                },
                                'epoch_losses': {
                                    'x': self.plot_epoch_xvalues,
                                    'y': self.plot_epoch_yvalues
                                }
                            },
                            'episode_history': self.episode_history
                        }
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(state, is_best=True)
                        if checkpoint_path:
                            print(f"Checkpoint saved successfully: {checkpoint_path}")
                        else:
                            print("Warning: Checkpoint save failed, but training continues...")
                except Exception as checkpoint_error:
                    print(f"Warning: Failed to save checkpoint: {str(checkpoint_error)}")
                    print("Training will continue without checkpoint...")

            self.plot_epoch_xvalues.append(epoch + 1)
            self.plot_epoch_yvalues.append(train_loss)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Accuracy: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            print(f"Learning Rate: {new_lr:.6f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.best_val_acc = best_val_acc
            self.best_model_state = best_model_state

        train_result = {
            "status": "success",
            "final_accuracy": val_acc,
            "best_accuracy": best_val_acc,
            "lr_history": lr_history
        }

        evaluation_data = None
        try:
            evaluation_data = self.get_evaluation_batch(num_samples=10)
            if evaluation_data:
                print(f"Generated evaluation data for {len(evaluation_data['predictions'])} images")
        except Exception as e:
            print(f"Warning: Could not generate evaluation data: {str(e)}")

        try:
            if evaluation_data is None:

                next_batch = self.get_next_batch(
                    strategy=self.training_config["sampling_strategy"],
                    batch_size=batch_size
                )
            else:

                next_batch = None
                
            episode_metrics = {
                'episode': self.episode,
                'train_result': train_result,
                'batch_size': len(next_batch) if next_batch else 0,
                'strategy': self.training_config["sampling_strategy"],
                'labeled_size': len(self.labeled_data),
                'unlabeled_size': len(self.unlabeled_data),
                'validation_size': len(self.validation_data),
                'best_val_acc': best_val_acc,
                'f1_score': best_f1,
                'learning_rate': new_lr,
                'lr_history': lr_history
            }
            
            self.episode_history.append(episode_metrics)
            
            self.plot_episode_xvalues.append(self.episode)
            self.plot_episode_yvalues.append(best_val_acc)
            
            if hasattr(self, 'checkpoint_manager'):
                state = {
                    'episode': self.episode,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'scheduler_config': scheduler_config,
                    'best_val_acc': best_val_acc,
                    'training_config': self.training_config,
                    'labeled_indices': list(self.labeled_data.keys()),
                    'unlabeled_indices': list(self.unlabeled_data.keys()),
                    'validation_indices': list(self.validation_data.keys()),
                    'lr_history': lr_history,
                    'metrics': {
                        'episode_accuracies': {
                            'x': self.plot_episode_xvalues,
                            'y': self.plot_episode_yvalues
                        },
                        'epoch_losses': {
                            'x': self.plot_epoch_xvalues,
                            'y': self.plot_epoch_yvalues
                        }
                    },
                    'episode_history': self.episode_history
                }
                self.checkpoint_manager.save_checkpoint(state)

            self.episode += 1
            
            result = {
                "status": "success",
                "metrics": episode_metrics,
                "final_val_acc": best_val_acc
            }
            
            if evaluation_data:
                result["evaluation_data"] = evaluation_data
                print("Returning episode result with evaluation data")
            else:
                result["next_batch"] = next_batch
                print("Returning episode result with next batch")
                
            return result
                
        except Exception as e:
            raise ValueError(f"Error after training: {str(e)}")
            
    except Exception as e:
        print(f"Error in train_episode: {str(e)}")
        raise

@app.get("/episode-history")
async def get_episode_history():
    """Return per-episode metrics including accuracy and F1 score.
    
    Each episode dict now contains:
      - episode         (int)   episode number
      - best_val_acc    (float) best validation accuracy in that episode, as %
      - f1_score        (float) weighted F1 at best-accuracy checkpoint, as %
      - labeled_size    (int)
      - unlabeled_size  (int)
      - strategy        (str)
      - batch_size      (int)
    """

    clean_episodes = []
    for ep in al_manager.episode_history:
        clean_episodes.append({
            "episode":         ep.get("episode"),
            "best_val_acc":    round(ep.get("best_val_acc", 0.0), 2),
            "f1_score":        round(ep["f1_score"], 2) if ep.get("f1_score") is not None else None,
            "labeled_size":    ep.get("labeled_size", 0),
            "unlabeled_size":  ep.get("unlabeled_size", 0),
            "validation_size": ep.get("validation_size", 0),
            "strategy":        ep.get("strategy", ""),
            "batch_size":      ep.get("batch_size", 0),
        })
    return {
        "episodes": clean_episodes,
        "current_episode": al_manager.episode
    }

@app.get("/validation-status")
async def get_validation_status():
    """Get validation set labeling status"""
    return al_manager.get_validation_status()

automated_trainer = AutomatedTrainingManager(al_manager)
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_training_state = {"running": False, "result": None, "error": None}

@app.post("/start-automated-training")
async def start_automated_training(config: TrainingConfig):
    """Start automated training with configuration"""
    try:
        await automated_trainer.start_automated_training(config.dict())
        return {"status": "success", "message": "Automated training started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-automated-training")
async def stop_automated_training():
    """Stop automated active learning cycle"""
    try:
        automated_trainer.stop_automated_training()
        return {"status": "success", "message": "Automated training stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/automated-training-status")
async def get_automated_training_status():
    """Get current automated training status"""
    return automated_trainer.get_training_status()

@app.post("/get-next-batch")
async def get_next_batch():
    """Manually get next batch during automated training with improved error handling"""
    try:
        print("\n=== Getting Next Batch ===")
        print(f"Current state: Labeled data: {len(al_manager.labeled_data)}, Unlabeled data: {len(al_manager.unlabeled_data)}")
        
        if len(al_manager.unlabeled_data) == 0:
            print("No unlabeled data available")
            return {
                "status": "error",
                "error": "No unlabeled data available for batch selection",
                "unlabeled_count": 0,
                "labeled_count": len(al_manager.labeled_data),
                "validation_count": len(al_manager.validation_data)
            }
        
        batch_size = int(
            automated_trainer.training_config.get(
            "batch_size",
            al_manager.config.get("batch_size", 16)
            )
            )

        try:
            config_batch_size = automated_trainer.training_config.get('batch_size')
            if config_batch_size is not None and isinstance(config_batch_size, (int, float)) and config_batch_size > 0:
                batch_size = int(config_batch_size)
        except (AttributeError, TypeError) as e:
            print(f"Error getting batch size from config: {str(e)}")
            print(f"Using default batch size: {batch_size}")
        
        if batch_size > len(al_manager.unlabeled_data):
            print(f"Batch size {batch_size} is larger than available unlabeled data {len(al_manager.unlabeled_data)}")

            batch_size = len(al_manager.unlabeled_data)
            print(f"Adjusted batch size to {batch_size}")
        
        strategy = "least_confidence"
        try:
            config_strategy = automated_trainer.training_config.get('sampling_strategy')
            if config_strategy is not None and isinstance(config_strategy, str):
                strategy = config_strategy
        except (AttributeError, TypeError) as e:
            print(f"Error getting strategy from config: {str(e)}")
            print(f"Using default strategy: {strategy}")
            
        print(f"Getting batch using strategy: {strategy}, batch size: {batch_size}")
        
        try:
            batch = al_manager.get_next_batch(strategy=strategy, batch_size=batch_size)
            
            try:
                automated_trainer.current_batch = batch
                automated_trainer.current_batch_size = len(batch)
                automated_trainer.current_batch_labeled_count = 0
            except Exception as config_error:
                print(f"Error updating trainer state: {str(config_error)}")

            print(f"Successfully got batch of {len(batch)} images")
            
            return {
                "status": "success",
                "batch_size": len(batch),
                "strategy": strategy
            }
        except Exception as e:
            print(f"Error getting batch: {str(e)}")
            traceback.print_exc()
            
            try:
                print("Trying fallback random sampling strategy")
                batch = al_manager.get_next_batch(strategy="random", batch_size=batch_size)
                
                try:
                    automated_trainer.current_batch = batch
                    automated_trainer.current_batch_size = len(batch)
                    automated_trainer.current_batch_labeled_count = 0
                except Exception as config_error:
                    print(f"Error updating trainer state: {str(config_error)}")
                
                print(f"Successfully got batch using fallback strategy with {len(batch)} images")
                
                return {
                    "status": "success",
                    "batch_size": len(batch),
                    "strategy": "random (fallback)"
                }
            except Exception as fallback_error:
                print(f"Fallback strategy also failed: {str(fallback_error)}")
                traceback.print_exc()
                raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-training-state")
async def reset_training_state():
    """Reset stuck training state"""
    if automated_trainer.check_training_state():
        return {"status": "success", "message": "Reset stuck training state"}
    return {"status": "success", "message": "Training state is not stuck"}

@app.post("/save-checkpoint")
async def save_checkpoint():
    """Save current model checkpoint manually"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        if not al_manager.output_dir:
            raise HTTPException(status_code=400, detail="No output directory set")
            
        if not hasattr(al_manager, 'checkpoint_manager') or al_manager.checkpoint_manager is None:
            al_manager.checkpoint_manager = CheckpointManager(al_manager.output_dir)
            
        state = {
            'episode': al_manager.episode,
            'model_state': al_manager.model.state_dict(),
            'best_val_acc': al_manager.best_val_acc,
            'training_config': getattr(automated_trainer, 'training_config', {}),
            'labeled_indices': list(al_manager.labeled_data.keys()),
            'unlabeled_indices': list(al_manager.unlabeled_data.keys()),
            'validation_indices': list(al_manager.validation_data.keys()),
            'metrics': {
                'episode_accuracies': {
                    'x': al_manager.plot_episode_xvalues,
                    'y': al_manager.plot_episode_yvalues
                },
                'epoch_losses': {
                    'x': al_manager.plot_epoch_xvalues,
                    'y': al_manager.plot_epoch_yvalues
                }
            },
            'episode_history': al_manager.episode_history
        }
        
        if hasattr(al_manager, 'optimizer') and al_manager.optimizer:
            try:
                state['optimizer_state'] = al_manager.optimizer.state_dict()
            except Exception as e:
                print(f"Warning: Could not save optimizer state: {e}")
                state['optimizer_state'] = {}
        else:
            state['optimizer_state'] = {}
            
        if hasattr(al_manager, 'lr_scheduler') and al_manager.lr_scheduler:
            try:
                if hasattr(al_manager.lr_scheduler, 'state_dict'):
                    state['scheduler_state'] = al_manager.lr_scheduler.state_dict()
                else:
                    state['scheduler_state'] = {}
            except Exception as e:
                print(f"Warning: Could not save scheduler state: {e}")
                state['scheduler_state'] = {}
        else:
            state['scheduler_state'] = {}
        
        checkpoint_path = al_manager.checkpoint_manager.save_checkpoint(state)
        
        return {
            "status": "success",
            "checkpoint_path": os.path.basename(checkpoint_path),
            "message": f"Checkpoint saved for episode {al_manager.episode}"
        }
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save checkpoint: {str(e)}")

@app.post("/load-checkpoint")
async def load_checkpoint(request: Request):
    """Load model checkpoint with proper request handling"""
    try:

        try:
            body = await request.json()
            checkpoint_path = body.get('checkpoint_path')
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON in request body: {str(e)}")
        
        if not checkpoint_path:
            raise HTTPException(status_code=422, detail="checkpoint_path is required")
        
        if not al_manager.output_dir:
            raise HTTPException(status_code=400, detail="No output directory set")
            
        if not hasattr(al_manager, 'checkpoint_manager') or al_manager.checkpoint_manager is None:
            al_manager.checkpoint_manager = CheckpointManager(al_manager.output_dir)
            
        full_checkpoint_path = os.path.join(al_manager.checkpoint_manager.checkpoint_dir, checkpoint_path)
        
        if not os.path.exists(full_checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

        if not al_manager.model:
            print("No model initialized — loading config from checkpoint to auto-initialize...")
            try:
                raw = torch.load(full_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
                saved_config = raw.get('training_config', {})
                model_name = saved_config.get('model_type', 'resnet18')
                num_classes = saved_config.get('num_classes', 2)
                project_name = saved_config.get('project_name', 'restored_project')
                print(f"Auto-initializing: model={model_name}, num_classes={num_classes}")
                al_manager.initialize_project(project_name, model_name, num_classes)
            except Exception as init_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"No model initialized and could not auto-initialize from checkpoint: {str(init_err)}"
                )
        
        print(f"Loading checkpoint from: {full_checkpoint_path}")
        
        checkpoint = al_manager.checkpoint_manager.load_checkpoint(
            al_manager.model, 
            getattr(al_manager, 'optimizer', None),
            getattr(al_manager, 'lr_scheduler', None),
            full_checkpoint_path
        )
        
        if checkpoint:
            al_manager.episode = checkpoint.get('episode', 0)
            al_manager.best_val_acc = checkpoint.get('best_val_acc', 0)
            
            metrics = checkpoint.get('metrics', {})
            al_manager.plot_episode_xvalues = metrics.get('episode_accuracies', {}).get('x', [])
            al_manager.plot_episode_yvalues = metrics.get('episode_accuracies', {}).get('y', [])
            al_manager.plot_epoch_xvalues = metrics.get('epoch_losses', {}).get('x', [])
            al_manager.plot_epoch_yvalues = metrics.get('epoch_losses', {}).get('y', [])
            
            al_manager.episode_history = checkpoint.get('episode_history', [])
            
            if 'labeled_indices' in checkpoint:
                labeled_count = len(checkpoint['labeled_indices'])
                unlabeled_count = len(checkpoint['unlabeled_indices'])
                validation_count = len(checkpoint['validation_indices'])
                print(f"Checkpoint had {labeled_count} labeled, {unlabeled_count} unlabeled, {validation_count} validation images")
            
            return {
                "status": "success",
                "episode": checkpoint.get('episode', 0),
                "best_val_acc": checkpoint.get('best_val_acc', 0),
                "message": f"Checkpoint {checkpoint_path} loaded successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Checkpoint file could not be loaded — check server logs for the exact error")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")
    
@app.get("/list-checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    try:
        if not al_manager.output_dir:
            return {"checkpoints": []}
            
        if not hasattr(al_manager, 'checkpoint_manager') or al_manager.checkpoint_manager is None:
            al_manager.checkpoint_manager = CheckpointManager(al_manager.output_dir)
            
        checkpoint_dir = al_manager.checkpoint_manager.checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            return {"checkpoints": []}
            
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_ep*.pt'))
        checkpoint_names = [os.path.basename(cp) for cp in checkpoints]
        
        return {"checkpoints": sorted(checkpoint_names)}
        
    except Exception as e:
        print(f"Error listing checkpoints: {str(e)}")
        return {"checkpoints": []}

@app.post("/configure-lr-scheduler")
async def configure_lr_scheduler(config: dict):
    """Configure learning rate scheduler"""
    try:
        al_manager.lr_config.update(config)
        return {"status": "success", "config": al_manager.lr_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lr-scheduler-status")
async def get_lr_scheduler_status():
    """Get current learning rate scheduler status"""
    try:
        if al_manager.lr_scheduler is None:
            return {
                "strategy": "plateau",
                "current_lr": 0.001,
                "history": [],
                "initial_lr": 0.001
            }
            
        return al_manager.lr_scheduler.get_status()
    except Exception as e:
        print(f"Error getting LR scheduler status: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get scheduler status: {str(e)}"
        )

@app.post("/import-pretrained-model")
async def import_pretrained_model(
    uploaded_file: UploadFile = File(...),
    model_type: str = Form(...),
    num_classes: int = Form(2),
    project_name: str = Form("imported_project")
):
    """
    Import a pre-trained model that wasn't created with this UI
    """
    try:

        content = await uploaded_file.read()
        
        supported_models = ["resnet18", "resnet50", "vision-transformer", "custom", "efficientnet", "densenet", "mobilenet"]
        
        if model_type not in supported_models:

            print(f"Model type '{model_type}' not in supported list, treating as 'custom'")
            model_type = "custom"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        state_dict = safe_load_model(tmp_path)
        
        if state_dict is None:
            os.unlink(tmp_path)
            raise ValueError("Failed to load model file. The file may be corrupted or in an unsupported format.")
        
        model_state = extract_model_state(state_dict)
        
        if model_type == "custom":
            print("Processing custom model...")
            
            try:
                detected_info = analyze_model_structure(state_dict)
                print(f"Detected model info: {detected_info}")
                
                if detected_info["detected_type"] != "unknown":
                    print(f"Auto-detected model type: {detected_info['detected_type']}")

                if detected_info["num_classes"] and detected_info["num_classes"] != num_classes:
                    print(f"Auto-detected {detected_info['num_classes']} classes, updating from {num_classes}")
                    num_classes = detected_info["num_classes"]
                    
            except Exception as detection_error:
                print(f"Could not auto-detect model structure: {detection_error}")
        
        if not al_manager.project_name:
            init_result = al_manager.initialize_project(
                project_name=project_name,
                model_name=model_type,
                num_classes=num_classes
            )
        
        if model_type == "custom":
            success = al_manager.load_custom_model_weights(model_state, num_classes)
            if not success:

                try:
                    missing_keys, unexpected_keys = al_manager.model.load_state_dict(model_state, strict=False)
                    print(f"Loaded custom model with missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
                except Exception as load_error:
                    print(f"Custom model loading failed: {load_error}")
                    raise ValueError(f"Failed to load custom model: {load_error}")
        else:

            try:
                al_manager.model.load_state_dict(model_state, strict=False)
                print("Loaded model state with non-strict matching")
            except Exception as e:
                print(f"Standard loading error: {str(e)}")
                
                fixed_state_dict = {}
                for k, v in model_state.items():
                    if k.startswith('module.'):
                        fixed_state_dict[k[7:]] = v
                    elif not k.startswith('module.') and f'module.{k}' in al_manager.model.state_dict():
                        fixed_state_dict[f'module.{k}'] = v
                    else:
                        fixed_state_dict[k] = v
                
                missing_keys, unexpected_keys = al_manager.model.load_state_dict(fixed_state_dict, strict=False)
                print(f"Loaded with key fixing. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        os.unlink(tmp_path)
        
        model_info = al_manager.get_model_info()
        
        return {
            "status": "success",
            "message": f"{'Custom' if model_type == 'custom' else model_type.title()} model imported successfully",
            "project_name": project_name,
            "model_type": model_type,
            "num_classes": num_classes,
            "model_info": model_info,
            "detected_architecture": detected_info.get("detected_type", "unknown") if model_type == "custom" else model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
    
@app.post("/verify-custom-model")
async def verify_custom_model(uploaded_file: UploadFile = File(...)):
    """
    Verify and analyze a custom model file before importing
    """
    try:
        content = await uploaded_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:

            state_dict = safe_load_model(tmp_path)
            
            if state_dict is None:
                os.unlink(tmp_path)
                return {
                    "status": "error",
                    "compatible": False,
                    "message": "Unable to load model file. File may be corrupted or in an unsupported format."
                }
            
            model_info = analyze_model_structure(state_dict)
            
            os.unlink(tmp_path)
            
            return {
                "status": "success",
                "compatible": True,
                "analysis": model_info,
                "recommended_model_type": model_info.get("detected_type", "custom"),
                "detected_classes": model_info.get("num_classes"),
                "message": f"Custom model analysis complete. Detected: {model_info.get('detected_type', 'unknown')} architecture"
            }
            
        except Exception as analysis_error:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            return {
                "status": "error", 
                "compatible": False,
                "message": f"Error analyzing custom model: {str(analysis_error)}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
    
def extract_model_state(state_dict):
    """
    Extract model state from various checkpoint formats
    Handles different model saving formats
    """
    if isinstance(state_dict, dict):

        if 'model_state' in state_dict:
            return state_dict['model_state']

        elif 'state_dict' in state_dict:
            return state_dict['state_dict']

        elif any(k.endswith('.weight') or k.endswith('.bias') for k in state_dict.keys()):
            return state_dict

        elif 'model' in state_dict and isinstance(state_dict['model'], dict):
            return state_dict['model']

        elif any(k.startswith('layer') or k.startswith('conv') or k.startswith('fc') 
                or k.startswith('features') or k.startswith('classifier') for k in state_dict.keys()):
            return state_dict
    
    if hasattr(state_dict, 'state_dict') and callable(getattr(state_dict, 'state_dict')):
        try:
            return state_dict.state_dict()
        except:
            pass
    
    return state_dict

def analyze_model_structure(state_dict):
    """
    Analyze the model structure to determine compatibility and required adaptations
    """
    result = {
        "compatible": False,
        "detected_type": "unknown",
        "num_classes": None,
        "adaptation_needed": True,
        "message": ""
    }
    
    try:

        model_state = extract_model_state(state_dict)
        
        if not model_state or not isinstance(model_state, dict):
            result["message"] = "Unable to extract model state dictionary. Invalid format."
            return result
            
        print(f"Keys in model state: {list(model_state.keys())[:10]}...")
        
        key_set = set([k.split('.')[0] for k in model_state.keys()])
        
        if any(k in ['cls_token', 'pos_embed', 'patch_embed'] for k in key_set):
            result["detected_type"] = "vision-transformer"
            
            head_keys = [k for k in model_state.keys() if 'head' in k.lower() and 'weight' in k]
            if head_keys:
                head_key = head_keys[0]
                head_shape = model_state[head_key].shape
                print(f"Found ViT head layer: {head_key} with shape: {head_shape}")
                
                if head_shape[0] == 512 or head_shape[0] > 100:
                    result["num_classes"] = None
                    result["message"] = f"Detected Vision Transformer (likely RETFound). Original head has {head_shape[0]} outputs (likely features). You can specify your desired number of classes."
                else:
                    result["num_classes"] = head_shape[0]
                    result["message"] = f"Detected Vision Transformer with {head_shape[0]} output classes."
            else:
                result["message"] = "Detected Vision Transformer. No classification head found - will add custom head."
                
        elif any(k.startswith('layer') for k in model_state.keys()):
            result["detected_type"] = "resnet"

            if 'fc.weight' in model_state:
                fc_shape = model_state['fc.weight'].shape
                result["num_classes"] = fc_shape[0]
                result["message"] = f"Detected ResNet with {fc_shape[0]} output classes."
            else:
                result["message"] = "Detected ResNet architecture."
                
        elif 'features' in key_set and 'classifier' in key_set:
            result["detected_type"] = "vgg-style"
        elif 'blocks' in key_set:
            result["detected_type"] = "mobilenet"
        else:
            result["detected_type"] = "custom"
        
        result["compatible"] = any(k.endswith('.weight') for k in model_state.keys())
        result["adaptation_needed"] = True
        
        if not result["compatible"]:
            result["message"] = "Model format not recognized. Unable to determine compatibility."
        
        return result
    
    except Exception as e:
        result["message"] = f"Error analyzing model: {str(e)}"
        return result

def detect_num_classes(state_dict):
    """
    Try to detect number of classes from model state dict
    Handles different naming conventions for final layer
    """

    output_patterns = [

        'fc.weight', 'classifier.weight', 'head.weight', 'output.weight',

        'head.weight', 'mlp_head.fc2.weight', 'cls_head.weight', 'classifier.weight',

        'roi_heads.box_predictor.cls_score.weight', 'bbox_pred.weight'
    ]
    
    for pattern in output_patterns:
        matching_keys = [k for k in state_dict.keys() if k.endswith(pattern)]
        if matching_keys:
            key = matching_keys[0]
            try:
                shape = state_dict[key].shape
                
                if len(shape) == 2:
                    out_features = shape[0]
                    return out_features
                elif len(shape) == 1:
                    return shape[0]
            except:
                continue
    
    try:

        weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
        
        classifier_patterns = ['fc', 'classifier', 'head', 'output', 'linear', 'pred']
        for pattern in classifier_patterns:
            candidates = [k for k in weight_keys if pattern in k.lower()]
            if candidates:

                key = candidates[-1]
                shape = state_dict[key].shape
                
                if len(shape) == 2:
                    return shape[0]
                elif len(shape) == 1:
                    return shape[0]
    except Exception as e:
        print(f"Error in flexible class detection: {e}")
    
    if 'patch_embed.proj.weight' in state_dict:

        try:

            if 'decoder_pred.weight' in state_dict:
                shape = state_dict['decoder_pred.weight'].shape
                return shape[0]
            
            if 'head.weight' in state_dict:
                shape = state_dict['head.weight'].shape
                return shape[0]
        except:
            pass
    
    return None

@app.post("/upload-csv-paths")
async def upload_csv_paths(
    file: UploadFile = File(...),
    delimiter: str = Form(","),
    val_split: float = Form(0.2),
    initial_labeled_ratio: float = Form(0.0),
):
    try:
        content = await file.read()
        text = content.decode("utf-8-sig")

        df = pd.read_csv(io.StringIO(text), delimiter=delimiter)

        path_columns = [
            "file_path", "filepath", "path", "filename",
            "image", "image_path", "img_path"
        ]

        path_col = None
        for col in path_columns:
            if col in df.columns:
                path_col = col
                break

        if path_col is None:
            path_col = df.columns[0]

        loaded_images = []

        for _, row in df.iterrows():
            image_path = str(row[path_col]).strip()

            if not image_path or image_path.lower() == "nan":
                continue

            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            try:
                img = Image.open(image_path).convert("RGB")
                img_tensor = al_manager.transform(img)

                img_id = len(al_manager.image_paths)
                al_manager.image_paths[img_id] = image_path
                al_manager.unlabeled_data[img_id] = img_tensor
                loaded_images.append(img_id)

            except Exception as e:
                print(f"Could not load image {image_path}: {e}")
                continue

        if len(loaded_images) == 0:
            raise HTTPException(
                status_code=400,
                detail="Could not find or process any images from the CSV"
            )

        # IMPORTANT:
        # Path-only CSV has no labels.
        # Do not create labeled_data.
        # Do not create validation_data.
        # Validation will be rebuilt later from labeled_data before training.
        split_info = {
            "total_images": len(loaded_images),
            "validation": 0,
            "initial_labeled": 0,
            "unlabeled": len(al_manager.unlabeled_data),
        }

        return {
            "status": "success",
            "message": "CSV image paths loaded as unlabeled data.",
            "split_info": split_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"CSV path upload error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def create_consistent_label_mapping(csv_content, label_column, delimiter, predefined_labels=None):
    """
    Create a consistent label mapping that respects predefined label order
    """
    import csv
    from io import StringIO
    
    if predefined_labels:
        label_to_index = {label: idx for idx, label in enumerate(predefined_labels)}
        return label_to_index
    
    csv_reader = csv.DictReader(StringIO(csv_content), delimiter=delimiter)
    unique_labels = set()
    
    for row in csv_reader:
        label_str = row.get(label_column, "").strip()
        if label_str:
            unique_labels.add(label_str)
    
    sorted_labels = sorted(list(unique_labels))
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    
    return label_to_index

@app.post("/upload-combined-with-labels")
async def upload_combined_with_labels(
    files: List[UploadFile],
    val_split: float = Form(0.2),
    initial_labeled_ratio: float = Form(0.4),
    label_column: str = Form("label")
):
    """
    Process a combined upload of a CSV file with both file paths and class labels, plus image files
    """
    try:

        csv_files = [f for f in files if f.filename.endswith(('.csv', '.tsv', '.txt'))]
        image_files = [f for f in files if f.content_type and f.content_type.startswith('image/')]
        
        if not csv_files:
            raise HTTPException(status_code=400, detail="No CSV file found in upload")
        
        if not image_files:
            raise HTTPException(status_code=400, detail="No image files found in upload")
        
        csv_file = csv_files[0]
        csv_content = await csv_file.read()
        csv_text = csv_content.decode('utf-8', errors='replace')
        
        import csv
        from io import StringIO
        
        first_line = csv_text.split('\n')[0]
        delimiter = ','
        for potential_delimiter in [',', '\t', ';', '|']:
            if potential_delimiter in first_line:
                delimiter = potential_delimiter
                break
        
        image_map = {}
        for img_file in image_files:

            image_map[img_file.filename] = img_file
            base_name = os.path.basename(img_file.filename)
            image_map[base_name] = img_file
        
        csv_reader = csv.DictReader(StringIO(csv_text), delimiter=delimiter)
        file_path_column = None
        label_column_name = label_column
        
        path_column_names = ['file_path', 'filepath', 'path', 'filename', 'image', 'file']

        label_column_names = ['label', 'class', 'category', 'target', 'y', 'classification']
        
        for field in csv_reader.fieldnames:
            if field.lower() in path_column_names:
                file_path_column = field
                break
        
        if label_column_name not in csv_reader.fieldnames:
            for field in csv_reader.fieldnames:
                if field.lower() in label_column_names:
                    label_column_name = field
                    break
        
        if not file_path_column:
            raise HTTPException(status_code=400, detail="Could not find file path column in CSV")
            
        if label_column_name not in csv_reader.fieldnames:
            print(f"Warning: No label column found. Available columns: {csv_reader.fieldnames}")
            has_labels = False
        else:
            has_labels = True
            print(f"Found label column: {label_column_name}")
        
        csv_reader = csv.DictReader(StringIO(csv_text), delimiter=delimiter)
        
        labeled_images = []
        unlabeled_images = []
        label_to_index = {}

        for row in csv_reader:
            path = row.get(file_path_column, "").strip()
            if not path:
                continue
                
            filename = os.path.basename(path)
            
            if filename in image_map:
                img_file = image_map[filename]
                content = await img_file.read()
                
                try:

                    img = Image.open(io.BytesIO(content)).convert('RGB')
                    img_tensor = al_manager.transform(img)
                    
                    img_id = len(al_manager.unlabeled_data) + len(al_manager.labeled_data) + len(al_manager.validation_data)

                    if has_labels:
                        label_str = row.get(label_column_name, "").strip()
                        if label_str:

                            if label_str not in label_to_index:
                                label_to_index[label_str] = len(label_to_index)
                            
                            label_idx = label_to_index[label_str]

                            al_manager.labeled_data[img_id] = (img_tensor, label_idx)
                            labeled_images.append(img_id)
                            continue
                    
                    al_manager.unlabeled_data[img_id] = img_tensor
                    unlabeled_images.append(img_id)
                    
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
                    continue
        
        if not (labeled_images or unlabeled_images):
            raise HTTPException(status_code=400, detail="Failed to process any images from the upload")
        
        print(f"Processed {len(labeled_images)} labeled images and {len(unlabeled_images)} unlabeled images")
        print(f"Label mapping: {label_to_index}")
        
        val_size = int(len(unlabeled_images) * val_split)
        val_indices = unlabeled_images[:val_size]
        remaining_unlabeled = unlabeled_images[val_size:]
        
        for idx in val_indices:
            if idx in al_manager.unlabeled_data:
                img_tensor = al_manager.unlabeled_data.pop(idx)
                al_manager.validation_data[idx] = (img_tensor, None)
        
        return {
            "status": "success",
            "message": f"Successfully processed images with labels from CSV",
            "stats": {
                "labeled": len(labeled_images),
                "unlabeled": len(remaining_unlabeled),
                "validation": len(val_indices),
                "total": len(labeled_images) + len(remaining_unlabeled) + len(val_indices)
            },
            "label_mapping": label_to_index
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV with labels: {str(e)}")

@app.post("/debug-csv-file")
async def debug_csv_file(csv_file: UploadFile = File(...)):
    """
    Debug CSV file to examine the paths and content
    """
    try:

        content = await csv_file.read()
        text_content = content.decode('utf-8', errors='replace')
        
        import csv
        from io import StringIO
        
        delimiter = ','
        if '\t' in text_content[:1000]:
            delimiter = '\t'
        
        csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
        
        sample_rows = []
        for i, row in enumerate(csv_reader):
            if i < 5:
                sample_rows.append(dict(row))
            else:
                break
                
        columns = csv_reader.fieldnames if csv_reader.fieldnames else []
        
        cwd = os.getcwd()
        search_dirs = [
            cwd,
            os.path.join(cwd, 'data'),
            os.path.join(cwd, 'images'),
            os.path.join(cwd, 'uploads')
        ]
        
        return {
            "columns": columns,
            "sample_rows": sample_rows,
            "delimiter_used": delimiter,
            "current_directory": cwd,
            "search_directories": search_dirs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV debug failed: {str(e)}")

@app.post("/upload-csv-paths-with-labels")
async def upload_csv_paths_with_labels(
    csv_file: UploadFile = File(...),
    label_column: str = Form(default="annotation"),
    delimiter: str = Form(default=","),
    val_split: float = Form(default=0.2),
    initial_labeled_ratio: float = Form(default=1.0),
    expected_label_mapping: str = Form(default=None)
):
    """
    Process a CSV file with both image paths and labels.
    Images with a label in the CSV go to labeled_data.
    Images without a label go to unlabeled_data.
    A val_split fraction of labeled images is held out as the validation set.
    """
    try:

        print(f"Received request:")
        print(f"  File: {csv_file.filename}")
        print(f"  Label column: {label_column}")
        print(f"  Delimiter: {delimiter}")
        print(f"  Expected label mapping: {expected_label_mapping}")
        
        label_to_index = {}
        if expected_label_mapping and expected_label_mapping.strip():
            try:
                import json
                label_to_index = json.loads(expected_label_mapping)
                print(f"Using expected label mapping from frontend: {label_to_index}")
            except Exception as e:
                print(f"Error parsing expected label mapping: {e}")
                label_to_index = {}
        
        if not csv_file or not csv_file.filename:
            raise HTTPException(status_code=400, detail="No CSV file provided")
            
        try:
            content = await csv_file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
                
            text_content = content.decode('utf-8', errors='replace')
            print(f"Successfully read {len(text_content)} characters from CSV")
            
        except Exception as read_error:
            print(f"Error reading CSV file: {str(read_error)}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(read_error)}")
        
        if delimiter == "\\t" or delimiter.lower() == "tab":
            delimiter = "\t"
        
        try:
            csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
            fieldnames = csv_reader.fieldnames
            
            if not fieldnames:
                raise HTTPException(status_code=400, detail="CSV file has no headers")
                
            print(f"Found CSV columns: {fieldnames}")
                
        except Exception as csv_error:
            print(f"Error parsing CSV: {str(csv_error)}")
            raise HTTPException(status_code=400, detail=f"Error parsing CSV with delimiter '{delimiter}': {str(csv_error)}")
        
        file_path_column = None
        path_column_names = ['file_path', 'filepath', 'path', 'filename', 'image', 'file', 'image_path']
        
        for field in fieldnames:
            if field.lower() in path_column_names:
                file_path_column = field
                break
        
        if not file_path_column and len(fieldnames) > 0:
            file_path_column = fieldnames[0]
            print(f"Using first column as file path: '{file_path_column}'")
        
        if not file_path_column:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not identify file path column. Available columns: {fieldnames}"
            )
        
        if label_column not in fieldnames:

            label_alternatives = ['annotation', 'label', 'class', 'category', 'target', 'classification', 'diagnosis']
            found_alternative = None
            
            for field in fieldnames:
                if field.lower() in label_alternatives:
                    found_alternative = field
                    break
            
            if found_alternative:
                print(f"Using '{found_alternative}' instead of '{label_column}' for labels")
                label_column = found_alternative
            elif len(fieldnames) > 1:
                label_column = fieldnames[1]
                print(f"Using second column '{label_column}' as label column")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Label column '{label_column}' not found. Available columns: {fieldnames}"
                )
        
        print(f"Using file path column: '{file_path_column}'")
        print(f"Using label column: '{label_column}'")
        
        if not label_to_index:
            print("No expected mapping provided. Scanning CSV for all labels...")
            csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
            unique_labels = set()
            
            for row in csv_reader:
                label_str = row.get(label_column, "").strip()
                if label_str:
                    unique_labels.add(label_str)
            
            sorted_labels = sorted(list(unique_labels))
            label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
            print(f"Created alphabetical label mapping: {label_to_index}")
        
        csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
        
        labeled_images = []
        unlabeled_images = []
        failed_paths = []
        
        processed_count = 0
        for row_index, row in enumerate(csv_reader):
            if not row or all(not val for val in row.values()):
                continue
                
            file_path = row.get(file_path_column, "").strip()
            if not file_path:
                continue
                
            label_str = row.get(label_column, "").strip() if label_column in row else None
            found_path = None
            
            if os.path.isabs(file_path) and os.path.exists(file_path):
                found_path = file_path
                if row_index < 3:
                    print(f"✓ Using absolute path: {file_path}")
            else:

                filename = os.path.basename(file_path)
                
                cwd = os.getcwd()
                search_dirs = [
                    cwd,
                    os.path.join(cwd, 'data'),
                    os.path.join(cwd, 'images'),
                    os.path.join(cwd, 'uploads'),
                    os.path.join(cwd, 'static'),
                ]
                
                for search_dir in search_dirs:
                    test_path = os.path.join(search_dir, filename)
                    if os.path.exists(test_path):
                        found_path = test_path
                        if row_index < 3:
                            print(f"✓ Found by filename in: {test_path}")
                        break
                
                if not found_path:
                    for search_dir in search_dirs:
                        test_path = os.path.join(search_dir, file_path)
                        if os.path.exists(test_path):
                            found_path = test_path
                            if row_index < 3:
                                print(f"✓ Found by relative path in: {test_path}")
                            break
            
            if not found_path:
                failed_paths.append(file_path)
                if len(failed_paths) <= 3:
                    print(f"✗ Could not find: {file_path}")
                continue
            
            try:
                img = Image.open(found_path).convert('RGB')
                img_tensor = al_manager.transform(img)
                
                img_id = len(al_manager.unlabeled_data) + len(al_manager.labeled_data) + len(al_manager.validation_data)
                
                al_manager.image_paths[img_id] = found_path
                
                if label_str:
                    label_idx = None
                    
                    try:
                        numeric_label = int(label_str)

                        if numeric_label in label_to_index.values():
                            label_idx = numeric_label
                            if row_index < 3:
                                print(f"Using numeric label directly: {numeric_label}")
                        else:

                            if label_str in label_to_index:
                                label_idx = label_to_index[label_str]
                    except ValueError:

                        if label_str in label_to_index:
                            label_idx = label_to_index[label_str]
                    
                    if label_idx is not None:
                        al_manager.ground_truth_labels[img_id] = {
                            "label_idx": label_idx,
                            "label_name": label_str,
                        }
                        al_manager.labeled_data[img_id] = (img_tensor, label_idx)
                        labeled_images.append(img_id)
                        if row_index < 3:
                            print(f"Labeled image {img_id} with label index {label_idx}")
                    else:
                        print(f"Warning: Label '{label_str}' could not be mapped. Expected mapping: {label_to_index}")
                        al_manager.unlabeled_data[img_id] = img_tensor
                        unlabeled_images.append(img_id)
                else:
                    al_manager.unlabeled_data[img_id] = img_tensor
                    unlabeled_images.append(img_id)
                
                processed_count += 1
                
            except Exception as img_error:
                print(f"Error processing image {found_path}: {str(img_error)}")
                failed_paths.append(file_path)
                continue
        
        if processed_count == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not process any images. Failed paths: {failed_paths[:5]}"
            )
        
        al_manager.config["val_split"] = val_split
        al_manager.rebuild_validation_split(val_split)

        remaining_unlabeled = unlabeled_images
        
        print(f"Final label mapping used: {label_to_index}")
        print(f"Processing complete: {len(labeled_images)} labeled, {len(remaining_unlabeled)} unlabeled, {len(al_manager.validation_data)} validation")
        
        return {
            "status": "success",
            "message": f"Successfully processed {processed_count} images from CSV",
            "stats": {
                "labeled": len(labeled_images),
                "unlabeled": len(remaining_unlabeled),
                "validation": len(al_manager.validation_data),
                "validation_labeled": len([1 for _, label in al_manager.validation_data.values() if label is not None]),
                "total": processed_count,
                "failed": len(failed_paths)
            },
            "label_mapping": label_to_index,
            "failed_paths": failed_paths[:10] if failed_paths else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload_csv_paths_with_labels: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recover-batch")
async def recover_batch(request: BatchRequest):
    """Emergency batch recovery with simple random sampling"""
    try:
        print("\n=== EMERGENCY BATCH RECOVERY ===")
        print(f"Using strategy: {request.strategy}, batch size: {request.batch_size}")
        
        if len(al_manager.unlabeled_data) == 0:
            raise HTTPException(status_code=400, detail="No unlabeled data available")
            
        batch_size = min(request.batch_size, len(al_manager.unlabeled_data))
        
        image_ids = list(al_manager.unlabeled_data.keys())
        selected_ids = random.sample(image_ids, batch_size)
        
        selected_samples = []
        for img_id in selected_ids:
            selected_samples.append({
                "image_id": img_id,
                "uncertainty": 0.5,
                "predictions": [{"label": "Unknown", "confidence": 0.0}]
            })
        
        al_manager.current_batch = [x["image_id"] for x in selected_samples]
        
        print(f"Successfully recovered batch with {len(selected_samples)} images")
        return selected_samples
        
    except Exception as e:
        print(f"Error in recover_batch: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/import-project")
async def import_project(uploaded_file: UploadFile = File(...)):
    """Import a complete project from ZIP file and prepare for continuation"""
    try:

        content = await uploaded_file.read()
        
        with tempfile.TemporaryDirectory() as temp_dir:

            zip_path = os.path.join(temp_dir, "imported_project.zip")
            with open(zip_path, "wb") as f:
                f.write(content)
            
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_dir)
            
            project_files = {}
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file in ['model.pt', 'annotations.csv', 'metadata.json']:
                        project_files[file] = os.path.join(root, file)
            
            required_files = ['model.pt', 'metadata.json']
            missing_files = [f for f in required_files if f not in project_files]
            if missing_files:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required files: {missing_files}"
                )
            
            with open(project_files['metadata.json'], 'r') as f:
                metadata = json.load(f)
            
            project_info = metadata['project_info']
            dataset_stats = metadata['dataset_stats']
            hyperparameters = metadata['hyperparameters']
            training_metrics = metadata['training_metrics']
            labels_info = metadata.get('labels', {})
            
            model_type = project_info.get('model_type', 'resnet50')
            is_vit = project_info.get('is_vision_transformer', False)
            model_class = project_info.get('model_class', '')
            
            print(f"Importing project with model_type: {model_type}, is_vit: {is_vit}, class: {model_class}")
            
            if torch.cuda.is_available():
                model_data = torch.load(project_files['model.pt'], weights_only=False)
            else:
                model_data = torch.load(project_files['model.pt'], map_location=torch.device('cpu'), weights_only=False)
            
            model_config = model_data['model_config']
            model_state = model_data['model_state']
            
            _vit_family = {'vision-transformer', 'vision_transformer', 'vit', 'dinov2'}
            if not is_vit and model_type not in _vit_family and model_type != 'custom':
                vit_indicators = ['cls_token', 'pos_embed', 'patch_embed']
                if any(key in model_state.keys() for key in vit_indicators):
                    print("Detected ViT model from state dict")
                    is_vit = True
                    model_type = 'vision-transformer'

            if model_type in _vit_family:
                is_vit = True
                # Distinguish DINOv2 (patch-14) from ViT-B/16 (patch-16) by patch weight shape
                patch_key = 'patch_embed.proj.weight'
                if patch_key in model_state:
                    patch_size = model_state[patch_key].shape[2]
                    if patch_size == 14 and model_type != 'dinov2':
                        print(f"Detected patch size 14 — upgrading model_type to 'dinov2'")
                        model_type = 'dinov2'
            
            num_classes = determine_num_classes_from_state(model_state)
            if not num_classes:
                num_classes = project_info.get('num_classes', 2)
            
            print(f"Detected {num_classes} classes from model structure")
            
            al_manager.project_name = project_info['project_name']
            imported_episode = project_info['current_episode']
            al_manager.episode = imported_episode + 1

            print(f"Imported project completed episode {imported_episode}, ready for episode {al_manager.episode}")

            al_manager.best_val_acc = project_info['best_validation_accuracy']
            
            al_manager.config.update({
                'val_split': hyperparameters.get('validation_split', 0.2),
                'initial_labeled_ratio': hyperparameters.get('initial_labeled_ratio', 0.1),
            })
            
            if model_type == 'dinov2':
                print("Initializing DINOv2 model")
                init_result = al_manager.initialize_project(
                    project_name=al_manager.project_name,
                    model_name='dinov2',
                    num_classes=num_classes
                )
            elif is_vit or model_type == 'vision-transformer':
                print("Initializing Vision Transformer model")
                init_result = al_manager.initialize_project(
                    project_name=al_manager.project_name,
                    model_name='vision-transformer',
                    num_classes=num_classes
                )
            else:
                print(f"Initializing {model_type} model")
                if 'resnet' in model_type.lower():
                    if '18' in model_type:
                        model_name = 'resnet18'
                    else:
                        model_name = 'resnet50'
                else:
                    model_name = 'resnet50'
                init_result = al_manager.initialize_project(
                    project_name=al_manager.project_name,
                    model_name=model_name,
                    num_classes=num_classes
                )
            
            try:
                al_manager.model.load_state_dict(model_state, strict=True)
                print("Model loaded with strict=True")
            except RuntimeError as e:
                print(f"Strict loading failed: {e}")
                print("Attempting to adapt model structure...")
                
                adapted_state = adapt_model_state_dict(model_state, al_manager.model.state_dict())
                missing_keys, unexpected_keys = al_manager.model.load_state_dict(adapted_state, strict=False)
                
                if missing_keys:
                    print(f"Missing keys after adaptation: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys after adaptation: {unexpected_keys}")
                
                print("Model loaded with adapted state dict")
            
            al_manager.plot_episode_xvalues = training_metrics['episode_accuracies']['episodes']
            al_manager.plot_episode_yvalues = training_metrics['episode_accuracies']['accuracies']
            al_manager.plot_epoch_xvalues = training_metrics['epoch_losses']['epochs']
            al_manager.plot_epoch_yvalues = training_metrics['epoch_losses']['losses']
            al_manager.episode_history = training_metrics.get('episode_history', [])
            
            if 'automated_trainer' in globals():
                automated_trainer.training_config.update({
                    'sampling_strategy': hyperparameters.get('sampling_strategy', 'least_confidence'),
                    'epochs': hyperparameters.get('epochs', 10),
                    'batch_size': hyperparameters.get('batch_size', 16),
                    'learning_rate': hyperparameters.get('learning_rate', 0.001)
                })
            
            al_manager.labeled_data.clear()
            al_manager.unlabeled_data.clear()
            al_manager.validation_data.clear()
            
            if not hasattr(al_manager, 'image_paths'):
                al_manager.image_paths = {}
            else:
                al_manager.image_paths.clear()
            
            al_manager.output_dir = os.path.join("output", al_manager.project_name, 
                datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(al_manager.output_dir, exist_ok=True)
            
            loaded_images_count = 0
            project_ready = False
            
            if 'annotations.csv' in project_files:
                print("Found annotations.csv, attempting to load existing data...")
                try:

                    import pandas as pd
                    df = pd.read_csv(project_files['annotations.csv'])
                    
                    print(f"Annotations CSV contains {len(df)} entries")
                    
                    search_paths = [
                        os.getcwd(),
                        os.path.join(os.getcwd(), 'data'),
                        os.path.join(os.getcwd(), 'images'),
                        os.path.join(os.getcwd(), 'uploads'),
                        extract_dir,
                        os.path.dirname(project_files['annotations.csv'])
                    ]
                    
                    failed_paths = []
                    
                    for _, row in df.iterrows():
                        try:
                            original_path = row['image_path']
                            image_id = int(row['image_id'])
                            label_index = row['label_index'] if pd.notna(row['label_index']) else None
                            split_type = row['split']
                            
                            image_found = False
                            actual_path = None
                            
                            if os.path.exists(original_path):
                                actual_path = original_path
                                image_found = True
                            else:

                                filename = os.path.basename(original_path)
                                for search_path in search_paths:
                                    candidate_path = os.path.join(search_path, filename)
                                    if os.path.exists(candidate_path):
                                        actual_path = candidate_path
                                        image_found = True
                                        break
                            
                            if image_found:

                                img = Image.open(actual_path).convert('RGB')
                                img_tensor = al_manager.transform(img)
                                
                                al_manager.image_paths[image_id] = actual_path
                                
                                if split_type == 'validation':
                                    al_manager.validation_data[image_id] = (img_tensor, int(label_index) if label_index is not None else None)
                                elif label_index is not None:
                                    al_manager.labeled_data[image_id] = (img_tensor, int(label_index))
                                else:
                                    al_manager.unlabeled_data[image_id] = img_tensor
                                
                                loaded_images_count += 1
                            else:
                                failed_paths.append(original_path)
                                
                        except Exception as e:
                            print(f"Error loading image {row.get('image_path', 'unknown')}: {str(e)}")
                            failed_paths.append(row.get('image_path', 'unknown'))
                            continue
                    
                    print(f"Successfully loaded {loaded_images_count} images from annotations")
                    if failed_paths:
                        print(f"Failed to load {len(failed_paths)} images. First few: {failed_paths[:5]}")
                    
                    project_ready = loaded_images_count > 0
                    
                except Exception as csv_error:
                    print(f"Error loading annotations.csv: {str(csv_error)}")
                    import traceback
                    traceback.print_exc()
            
            final_model_type = model_type
            
            if project_ready:
                message = f"Project '{al_manager.project_name}' ({final_model_type}) imported successfully with {loaded_images_count} existing images loaded. Project is ready for active learning!"
            else:
                message = f"Project '{al_manager.project_name}' ({final_model_type}) imported successfully. Upload new images to continue training with the existing model."
            
            return {
                "status": "success",
                "project_info": {
                    **project_info,
                    "model_type": final_model_type,
                    "is_vision_transformer": is_vit,
                    "num_classes": num_classes
                },
                "dataset_stats": {
                    **dataset_stats,
                    "current_labeled": len(al_manager.labeled_data),
                    "current_unlabeled": len(al_manager.unlabeled_data),
                    "current_validation": len(al_manager.validation_data),
                    "loaded_from_annotations": loaded_images_count
                },
                "hyperparameters": hyperparameters,
                "labels": labels_info,
                "training_config": automated_trainer.training_config if 'automated_trainer' in globals() else {},
                "model_ready": True,
                "project_ready": project_ready,
                "images_loaded": loaded_images_count > 0,
                "message": message
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error importing project: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to import project: {str(e)}")

def determine_num_classes_from_state(state_dict):
    """Determine number of classes from model state dict"""
    try:

        if 'fc.weight' in state_dict:
            return state_dict['fc.weight'].shape[0]
        elif 'fc.1.weight' in state_dict:
            return state_dict['fc.1.weight'].shape[0]
        elif 'fc.2.weight' in state_dict:
            return state_dict['fc.2.weight'].shape[0]
        
        elif 'classifier.weight' in state_dict:
            return state_dict['classifier.weight'].shape[0]
        elif 'head.weight' in state_dict:
            return state_dict['head.weight'].shape[0]
        
        fc_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
        if fc_keys:
            return state_dict[fc_keys[0]].shape[0]
        
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
        if classifier_keys:
            return state_dict[classifier_keys[0]].shape[0]
        
        return None
    except Exception as e:
        print(f"Error determining num_classes: {e}")
        return None

def adapt_model_state_dict(saved_state, target_state):
    """Adapt saved model state dict to match target model structure"""
    adapted_state = {}
    
    print("Adapting model state dict...")
    print(f"Saved state keys sample: {list(saved_state.keys())[:10]}...")
    print(f"Target state keys sample: {list(target_state.keys())[:10]}...")
    
    for key in saved_state.keys():
        if not any(classifier_key in key for classifier_key in ['fc.', 'classifier.', 'head.']):
            if key in target_state:
                adapted_state[key] = saved_state[key]
            else:
                print(f"Skipping key not in target: {key}")
    
    target_classifier_keys = [k for k in target_state.keys() if any(c in k for c in ['fc.', 'classifier.', 'head.'])]
    saved_classifier_keys = [k for k in saved_state.keys() if any(c in k for c in ['fc.', 'classifier.', 'head.'])]
    
    print(f"Target classifier keys: {target_classifier_keys}")
    print(f"Saved classifier keys: {saved_classifier_keys}")
    
    if 'fc.weight' in target_state and 'fc.bias' in target_state:

        if 'fc.1.weight' in saved_state and 'fc.1.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['fc.1.weight']
            adapted_state['fc.bias'] = saved_state['fc.1.bias']
            print("Mapped fc.1 -> fc")
        elif 'fc.2.weight' in saved_state and 'fc.2.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['fc.2.weight']
            adapted_state['fc.bias'] = saved_state['fc.2.bias']
            print("Mapped fc.2 -> fc")
        elif 'fc.weight' in saved_state and 'fc.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['fc.weight']
            adapted_state['fc.bias'] = saved_state['fc.bias']
            print("Direct fc mapping")
        elif 'classifier.weight' in saved_state and 'classifier.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['classifier.weight']
            adapted_state['fc.bias'] = saved_state['classifier.bias']
            print("Mapped classifier -> fc")
    
    elif 'fc.1.weight' in target_state and 'fc.1.bias' in target_state:

        if 'fc.weight' in saved_state and 'fc.bias' in saved_state:
            adapted_state['fc.1.weight'] = saved_state['fc.weight']
            adapted_state['fc.1.bias'] = saved_state['fc.bias']
            print("Mapped fc -> fc.1")
        elif 'fc.1.weight' in saved_state and 'fc.1.bias' in saved_state:
            adapted_state['fc.1.weight'] = saved_state['fc.1.weight']
            adapted_state['fc.1.bias'] = saved_state['fc.1.bias']
            print("Direct fc.1 mapping")
    
    elif 'classifier.weight' in target_state and 'classifier.bias' in target_state:

        if 'fc.weight' in saved_state and 'fc.bias' in saved_state:
            adapted_state['classifier.weight'] = saved_state['fc.weight']
            adapted_state['classifier.bias'] = saved_state['fc.bias']
            print("Mapped fc -> classifier")
        elif 'fc.1.weight' in saved_state and 'fc.1.bias' in saved_state:
            adapted_state['classifier.weight'] = saved_state['fc.1.weight']
            adapted_state['classifier.bias'] = saved_state['fc.1.bias']
            print("Mapped fc.1 -> classifier")
        elif 'classifier.weight' in saved_state and 'classifier.bias' in saved_state:
            adapted_state['classifier.weight'] = saved_state['classifier.weight']
            adapted_state['classifier.bias'] = saved_state['classifier.bias']
            print("Direct classifier mapping")
    
    print(f"Adapted state final keys: {len(adapted_state)} keys")
    return adapted_state

def evaluate_model_on_unlabeled(self, num_samples=10):
    """
    Evaluate model performance on a sample of unlabeled data
    Returns predictions with confidence scores for the next batch of images
    """
    try:
        if not self.model or len(self.unlabeled_data) == 0:
            return None
            
        self.model.eval()
        
        sample_size = min(num_samples, len(self.unlabeled_data))
        sample_ids = list(self.unlabeled_data.keys())[:sample_size]
        
        predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for img_id in sample_ids:
                img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                top_prob, top_class = torch.max(probs, dim=1)
                confidence = float(top_prob.item())
                predicted_class = int(top_class.item())
                
                all_probs = []
                for i, prob in enumerate(probs[0]):
                    all_probs.append({
                        'class_index': i,
                        'probability': float(prob.item())
                    })
                
                all_probs.sort(key=lambda x: x['probability'], reverse=True)
                
                predictions.append({
                    'image_id': img_id,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_probabilities': all_probs
                })
                
                all_confidences.append(confidence)
        
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        return {
            'predictions': predictions,
            'overall_confidence': overall_confidence,
            'num_evaluated': len(predictions),
            'episode_info': {
                'episode': self.episode,
                'validation_accuracy': self.best_val_acc
            }
        }
        
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return None
    
def get_evaluation_batch(self, num_samples=10):
    """
    Get the next batch of unlabeled images for evaluation display
    Similar to get_next_batch but focused on evaluation metrics
    """
    try:
        if not self.model or len(self.unlabeled_data) == 0:
            return None
            
        evaluation_data = self.evaluate_model_on_unlabeled(num_samples)
        
        if evaluation_data:

            for pred in evaluation_data['predictions']:

                pred['uncertainty'] = 1 - pred['confidence']
                
                pred['predictions'] = [
                    {
                        'label': f"Class {i}",
                        'confidence': prob['probability']
                    }
                    for i, prob in enumerate(pred['all_probabilities'])
                ]
        
        return evaluation_data
        
    except Exception as e:
        print(f"Error getting evaluation batch: {str(e)}")
        return None

def inspect_and_save_model_info(model, output_path="model_inspection.json"):
    """
    Comprehensive model inspection - saves all model info to JSON for debugging
    """
    try:
        state_dict = model.state_dict()
        
        model_info = {
            "basic_info": {
                "class_name": model.__class__.__name__,
                "module_name": model.__class__.__module__,
                "model_type": str(type(model)),
                "model_repr": str(model)[:1000] + "..." if len(str(model)) > 1000 else str(model)
            },
            "state_dict_analysis": {
                "total_parameters": len(state_dict),
                "parameter_shapes": {k: list(v.shape) for k, v in state_dict.items()},
                "parameter_names": list(state_dict.keys()),
                "first_20_keys": list(state_dict.keys())[:20],
                "last_20_keys": list(state_dict.keys())[-20:]
            },
            "architecture_detection": {
                "has_resnet_layers": any(key.startswith('layer') for key in state_dict.keys()),
                "has_vit_indicators": any(indicator in key for key in state_dict.keys() 
                                        for indicator in ['cls_token', 'pos_embed', 'patch_embed', 'blocks']),
                "has_attention": any('attn' in key or 'attention' in key for key in state_dict.keys()),
                "has_transformer": any('transformer' in key for key in state_dict.keys()),
                "resnet_layer_keys": [k for k in state_dict.keys() if k.startswith('layer')],
                "vit_keys": [k for k in state_dict.keys() if any(indicator in k for indicator in ['cls_token', 'pos_embed', 'patch_embed', 'blocks'])],
                "attention_keys": [k for k in state_dict.keys() if 'attn' in k or 'attention' in k],
                "classifier_keys": [k for k in state_dict.keys() if any(c in k for c in ['classifier', 'head', 'fc'])]
            },
            "layer_analysis": {
                "conv_layers": [k for k in state_dict.keys() if 'conv' in k and 'weight' in k],
                "linear_layers": [k for k in state_dict.keys() if any(layer_type in k for layer_type in ['fc', 'linear', 'classifier', 'head']) and 'weight' in k],
                "norm_layers": [k for k in state_dict.keys() if any(norm_type in k for norm_type in ['bn', 'norm', 'layer_norm']) and 'weight' in k],
                "embedding_layers": [k for k in state_dict.keys() if 'embed' in k]
            },
            "model_structure": {},
            "pytorch_model_info": {}
        }
        
        try:

            children = list(model.children())
            named_children = list(model.named_children())
            modules = list(model.modules())
            named_modules = list(model.named_modules())
            
            model_info["model_structure"] = {
                "num_children": len(children),
                "num_modules": len(modules),
                "named_children": [(name, str(type(child))) for name, child in named_children],
                "child_types": [str(type(child)) for child in children],
                "has_fc": hasattr(model, 'fc'),
                "has_classifier": hasattr(model, 'classifier'),
                "has_head": hasattr(model, 'head'),
                "has_features": hasattr(model, 'features')
            }
            
            if hasattr(model, 'fc'):
                fc_layer = model.fc
                model_info["model_structure"]["fc_info"] = {
                    "type": str(type(fc_layer)),
                    "repr": str(fc_layer),
                    "has_in_features": hasattr(fc_layer, 'in_features'),
                    "has_out_features": hasattr(fc_layer, 'out_features'),
                    "in_features": getattr(fc_layer, 'in_features', None),
                    "out_features": getattr(fc_layer, 'out_features', None)
                }
            
            if hasattr(model, 'classifier'):
                classifier = model.classifier
                model_info["model_structure"]["classifier_info"] = {
                    "type": str(type(classifier)),
                    "repr": str(classifier),
                    "has_in_features": hasattr(classifier, 'in_features'),
                    "has_out_features": hasattr(classifier, 'out_features'),
                    "in_features": getattr(classifier, 'in_features', None),
                    "out_features": getattr(classifier, 'out_features', None)
                }
            
            if hasattr(model, 'head'):
                head = model.head
                model_info["model_structure"]["head_info"] = {
                    "type": str(type(head)),
                    "repr": str(head),
                    "has_in_features": hasattr(head, 'in_features'),
                    "has_out_features": hasattr(head, 'out_features'),
                    "in_features": getattr(head, 'in_features', None),
                    "out_features": getattr(head, 'out_features', None)
                }
                
        except Exception as struct_error:
            model_info["model_structure"]["error"] = str(struct_error)
        
        try:
            import torch
            model_info["pytorch_model_info"] = {
                "device": str(next(model.parameters()).device) if list(model.parameters()) else "no_parameters",
                "dtype": str(next(model.parameters()).dtype) if list(model.parameters()) else "no_parameters",
                "requires_grad": any(p.requires_grad for p in model.parameters()),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "is_training": model.training
            }
        except Exception as torch_error:
            model_info["pytorch_model_info"]["error"] = str(torch_error)
        
        detection_result = {
            "detected_type": "unknown",
            "confidence": "low",
            "reasoning": []
        }
        
        if model_info["architecture_detection"]["has_resnet_layers"]:
            detection_result["detected_type"] = "resnet"
            detection_result["confidence"] = "high"
            detection_result["reasoning"].append("Found ResNet layer structure (layer1, layer2, etc.)")
            
            if model_info["model_structure"].get("fc_info", {}).get("in_features") == 512:
                detection_result["variant"] = "resnet18"
            elif model_info["model_structure"].get("fc_info", {}).get("in_features") == 2048:
                detection_result["variant"] = "resnet50"
        
        elif model_info["architecture_detection"]["has_vit_indicators"]:
            detection_result["detected_type"] = "vision_transformer"
            detection_result["confidence"] = "high"
            detection_result["reasoning"].append("Found ViT indicators (cls_token, pos_embed, patch_embed, blocks)")
        
        elif model_info["architecture_detection"]["has_attention"]:
            detection_result["detected_type"] = "transformer"
            detection_result["confidence"] = "medium"
            detection_result["reasoning"].append("Found attention mechanisms")
        
        class_name = model_info["basic_info"]["class_name"].lower()
        if 'resnet' in class_name:
            detection_result["class_name_suggests"] = "resnet"
        elif any(vit_term in class_name for vit_term in ['vit', 'vision', 'transformer']):
            detection_result["class_name_suggests"] = "vision_transformer"
        else:
            detection_result["class_name_suggests"] = "unknown"
        
        model_info["detection_result"] = detection_result
        
        import json
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print(f"Model inspection saved to: {output_path}")
        print(f"Detected model type: {detection_result['detected_type']} (confidence: {detection_result['confidence']})")
        
        return model_info
        
    except Exception as e:
        error_info = {
            "error": str(e),
            "basic_class_name": model.__class__.__name__ if hasattr(model, '__class__') else "unknown"
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"Error during model inspection: {e}")
        return error_info

def determine_model_type_for_export(model):
    """Determine the model type for export based on model structure"""
    
    state_dict = model.state_dict()
    class_name = model.__class__.__name__
    
    print(f"=== MODEL TYPE DETECTION DEBUG ===")
    print(f"Model class name: {class_name}")
    print(f"State dict keys (first 10): {list(state_dict.keys())[:10]}")
    
    if any(vit_indicator in class_name for vit_indicator in ['ViT', 'Vision', 'Transformer', 'SimpleViTClassifier']):
        print("Detected ViT from class name")
        return 'vision-transformer'
    
    vit_indicators = ['cls_token', 'pos_embed', 'patch_embed', 'blocks.', 'norm.weight', 'head.weight']
    found_vit_keys = [key for key in state_dict.keys() if any(indicator in key for indicator in vit_indicators)]
    
    if found_vit_keys:
        print(f"Detected ViT from state dict keys: {found_vit_keys[:5]}")
        return 'vision-transformer'
    
    attention_keys = [key for key in state_dict.keys() if any(pattern in key for pattern in ['attn', 'attention', 'self_attention'])]
    if attention_keys:
        print(f"Detected transformer from attention keys: {attention_keys[:3]}")
        return 'vision-transformer'
    
    if any(key.startswith('layer') for key in state_dict.keys()):
        print("Detected ResNet from layer structure")

        feature_count = None
        for key in state_dict.keys():
            if ('fc' in key and 'weight' in key) or ('classifier' in key and 'weight' in key):
                weight_shape = state_dict[key].shape
                if len(weight_shape) == 2:
                    feature_count = weight_shape[1]
                    break
        
        if feature_count == 512:
            return 'resnet18'
        elif feature_count == 2048:
            return 'resnet50'
        else:
            return 'resnet50'
    
    if 'features' in [key.split('.')[0] for key in state_dict.keys()]:
        print("Detected VGG-style model")
        return 'vgg'
    
    classifier_keys = [k for k in state_dict.keys() if any(c in k for c in ['classifier', 'head', 'fc'])]
    if classifier_keys:
        print(f"Found classifier keys: {classifier_keys}")

        if not any(key.startswith('layer') for key in state_dict.keys()):
            print("Assuming custom ViT due to classifier without ResNet layers")
            return 'vision-transformer'
    
    print("Defaulting to custom")
    return 'custom'

@app.post("/load-existing-annotations")
async def load_existing_annotations():
    """Load existing annotations from imported project and start active learning"""
    try:
        if not al_manager.project_name:
            raise HTTPException(status_code=400, detail="No project loaded")
        
        total_data = len(al_manager.labeled_data) + len(al_manager.unlabeled_data) + len(al_manager.validation_data)
        
        if total_data == 0:
            return {
                "status": "no_data",
                "message": "No existing data found in project. Please upload new images."
            }
        
        if len(al_manager.unlabeled_data) > 0:
            try:

                strategy = getattr(automated_trainer, 'training_config', {}).get('sampling_strategy', 'least_confidence')
                batch_size = getattr(automated_trainer, 'training_config', {}).get('batch_size', 16)
                
                batch = al_manager.get_next_batch(strategy, min(batch_size, len(al_manager.unlabeled_data)))
                
                return {
                    "status": "success", 
                    "message": f"Loaded existing project data. Ready for active learning with {len(batch)} images.",
                    "data_stats": {
                        "labeled": len(al_manager.labeled_data),
                        "unlabeled": len(al_manager.unlabeled_data), 
                        "validation": len(al_manager.validation_data)
                    },
                    "batch_ready": True
                }
            except Exception as e:
                return {
                    "status": "success",
                    "message": f"Loaded existing project data, but couldn't get batch: {str(e)}",
                    "data_stats": {
                        "labeled": len(al_manager.labeled_data),
                        "unlabeled": len(al_manager.unlabeled_data),
                        "validation": len(al_manager.validation_data)
                    },
                    "batch_ready": False
                }
        else:
            return {
                "status": "success",
                "message": "Project loaded successfully. All data is labeled - ready for final training.",
                "data_stats": {
                    "labeled": len(al_manager.labeled_data),
                    "unlabeled": len(al_manager.unlabeled_data),
                    "validation": len(al_manager.validation_data)
                },
                "batch_ready": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load existing annotations: {str(e)}")

@app.post("/update-project-labels")
async def update_project_labels(request: dict):
    """Update the current project labels"""
    try:
        labels = request.get('labels', [])

        al_manager.current_labels = labels
        return {"status": "success", "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate-model")
async def evaluate_model(num_samples: int = 10):
    """Get model evaluation on unlabeled data"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        evaluation_data = al_manager.evaluate_model_on_unlabeled(num_samples)
        
        if not evaluation_data:
            raise HTTPException(status_code=400, detail="Unable to evaluate model - no unlabeled data available")
            
        return evaluation_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/evaluation-batch") 
async def get_evaluation_batch(num_samples: int = 10):
    """Get evaluation batch for display"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        evaluation_data = al_manager.get_evaluation_batch(num_samples)
        
        if not evaluation_data:
            raise HTTPException(status_code=400, detail="No evaluation data available")
            
        return evaluation_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation batch: {str(e)}")

@app.post("/continue-from-evaluation")
async def continue_from_evaluation():
    """Continue training after evaluation screen"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        if len(al_manager.unlabeled_data) == 0:
            return {
                "status": "complete",
                "message": "No more unlabeled data available. Training complete!"
            }
            
        strategy = getattr(automated_trainer, 'training_config', {}).get('sampling_strategy', 'least_confidence')
        batch_size = getattr(automated_trainer, 'training_config', {}).get('batch_size', 16)
        
        batch = al_manager.get_next_batch(strategy, min(batch_size, len(al_manager.unlabeled_data)))
        
        al_manager.current_batch = [x["image_id"] for x in batch]
        
        return {
            "status": "success",
            "message": f"Ready to continue with {len(batch)} new images for labeling",
            "batch_size": len(batch),
            "batch": batch
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to continue from evaluation: {str(e)}")

@app.post("/train-episode")
async def train_episode(request: TrainEpisodeRequest):
    try:
        epochs = request.epochs
        batch_size = request.batch_size
        learning_rate = request.learning_rate
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="Model not initialized.")
        if len(al_manager.labeled_data) == 0:
            raise HTTPException(status_code=400, detail="No labeled data available for training.")
        if _training_state["running"]:
            raise HTTPException(status_code=409, detail="Training already in progress.")

        _training_state["running"] = True
        _training_state["result"] = None
        _training_state["error"] = None

        loop = asyncio.get_event_loop()

        def run_training():
            try:
                result = al_manager.train_episode(epochs, batch_size, learning_rate)
                _training_state["result"] = result
                _training_state["error"] = None
            except Exception as e:
                _training_state["error"] = str(e)
                _training_state["result"] = None
            finally:
                _training_state["running"] = False

        loop.run_in_executor(_thread_pool, run_training)

        return {"status": "started", "message": "Training started in background. Poll /training-status for completion."}

    except HTTPException:
        raise
    except Exception as e:
        _training_state["running"] = False
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-status")
async def training_status():
    if _training_state["running"]:
        return {"status": "running", "result": None, "error": None}
    if _training_state["error"]:
        return {"status": "error", "result": None, "error": _training_state["error"]}
    if _training_state["result"] is not None:
        return {"status": "complete", "result": _training_state["result"], "error": None}
    return {"status": "idle", "result": None, "error": None}

@app.get("/label-hint/{image_id}")
async def get_label_hint(image_id: int):
    hint = al_manager.ground_truth_labels.get(image_id)
    full_path = al_manager.image_paths.get(image_id, "")
    file_name = os.path.basename(full_path) if full_path else f"image_{image_id}"
    if hint:
        return {
            "available": True,
            "label_idx":  hint["label_idx"],
            "label_name": hint["label_name"],
            "file_name":  file_name,
        }
    return {"available": False, "label_idx": None, "label_name": None, "file_name": file_name}

@app.get("/")
async def main():
    return {"message": "Welcome to Active Learning API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
