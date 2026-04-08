import random
from models import Observation, Action


class Simulator:
    """
    Realistic OffroadSegNet ML workflow simulator.
    
    Simulates:
    - Diminishing returns from capacity increases
    - Training plateau after sufficient epochs
    - Overfitting curves from repeated training
    - Instability from suboptimal learning rates
    - Class imbalance effects
    - Augmentation improving generalization
    - Training time cost per epoch
    - Deterministic behavior with seed control
    """
    
    def __init__(self, target_iou: float = 0.50, initial_compute: int = 30, seed: int = 42):
        self.target_iou = target_iou
        self.initial_compute = initial_compute
        self.seed = seed
        random.seed(seed)
        self.reset()
    
    def reset(self, seed: int = None):
        """Initialize simulator to baseline state."""
        if seed is not None:
            self.seed = seed
            random.seed(seed)
        else:
            random.seed(self.seed)
        self.backbone_depth = 18  # ResNet-18 baseline
        self.feature_channels = 256  # Default feature channels
        self.regularization_strength = 0.0
        self.learning_rate = 0.01  # Suboptimal initial LR
        self.optimizer = "sgd"
        self.batch_size = 32  # Standard batch size
        self.augmentation_level = 0  # No augmentation
        self.mixed_precision = False
        self.class_balancing = False
        self.loss_weights = {"road": 1.0, "vegetation": 1.0, "sky": 1.0}
        
        # Training State
        self.epoch = 0
        self.remaining_compute = self.initial_compute
        self.early_stopped = False
        self.total_epochs_trained = 0
        
        # Performance Metrics
        self.road_iou = 0.40  # Road class baseline
        self.vegetation_iou = 0.10  # Vegetation baseline
        self.sky_iou = 0.00  # Sky missing from initial training
        self.train_loss = 1.2
        self.val_loss = 1.5  # Already shows overfitting risk
        
        # Status Flags
        self.overfitting_flag = False
        self.instability_flag = True  # Bad initial LR causes instability
        self.training_plateau = False
        self.class_imbalance_detected = True  # Sky class missing
        
        # Tracking
        self.last_action = None
        self.repeating_penalty = 0.0
        self.action_history = []
        self.iou_history = [self.mean_iou]
        self.loss_history = [self.val_loss]
        self.epoch_improvements = []  # Track improvement per epoch
    
    @property
    def mean_iou(self) -> float:
        """Calculate mean IoU across all classes (always bounded [0,1])."""
        raw_mean = (self.road_iou + self.vegetation_iou + self.sky_iou) / 3.0
        return max(0.0, min(1.0, raw_mean))
    
    def step(self, action: Action) -> dict:
        """Execute one step with given action."""
        info = {
            "msg": "",
            "iou_improvement": 0.0,
            "repeating_penalty": 0.0
        }
        old_iou = self.mean_iou
        
        # Check for repeated configuration actions (idempotent but wasteful)
        is_repeat = (
            action.action == self.last_action and 
            action.action not in ["run_training_epoch", "early_stop_training"]
        )
        
        if is_repeat:
            self.repeating_penalty = -0.3
            info["repeating_penalty"] = self.repeating_penalty
            info["msg"] = "WARNING: Repeated configuration action (already enabled)."
        else:
            self.repeating_penalty = 0.0
        
        self.last_action = action.action
        self.action_history.append(action.action)
        
        # Configuration Actions (cost 1 compute each)
        if action.action == "adjust_learning_rate":
            if self.remaining_compute > 0:
                self.remaining_compute = max(0, self.remaining_compute - 1)  # Compute cost
            self._adjust_learning_rate()
            info["msg"] = "OK: Learning rate optimized to 0.001. Instability resolved."
            
        elif action.action == "enable_augmentation":
            if self.remaining_compute > 0:
                self.remaining_compute = max(0, self.remaining_compute - 1)  # Compute cost
            self._enable_augmentation()
            info["msg"] = "OK: Data augmentation enabled (rotation, flip, color jitter)."
            
        elif action.action == "increase_regularization":
            if self.remaining_compute > 0:
                self.remaining_compute = max(0, self.remaining_compute - 1)  # Compute cost
            self._increase_regularization()
            info["msg"] = "OK: L2 regularization increased to 0.0005. Overfitting risk reduced."
            
        elif action.action == "switch_optimizer":
            if self.remaining_compute > 0:
                self.remaining_compute = max(0, self.remaining_compute - 1)  # Compute cost
            self._switch_optimizer()
            info["msg"] = "OK: Switched to AdamW optimizer. Better convergence expected."
            
        elif action.action == "enable_class_balancing":
            if self.remaining_compute > 0:
                self.remaining_compute = max(0, self.remaining_compute - 1)  # Compute cost
            self._enable_class_balancing()
            info["msg"] = "OK: Focal loss with class reweighting enabled."
            
        elif action.action == "reduce_batch_size":
            if self.remaining_compute > 0:
                self.remaining_compute = max(0, self.remaining_compute - 1)  # Compute cost
            self._reduce_batch_size()
            info["msg"] = "WARNING: Batch size reduced to 16. More memory efficient but slower."
            
        elif action.action == "early_stop_training":
            self.early_stopped = True
            info["msg"] = "STOP: Training stopped. Epoch budget preserved."
            
        elif action.action == "run_training_epoch":
            self._run_training_epoch()
            info["msg"] = f"Epoch {self.epoch} ({self.total_epochs_trained} total) completed."
        
        # Update status flags
        self._update_status_flags()
        
        # Track improvement
        iou_improvement = self.mean_iou - old_iou
        info["iou_improvement"] = iou_improvement
        self.epoch_improvements.append(iou_improvement)
        
        return info
    
    def _adjust_learning_rate(self):
        """Reduce learning rate from poor default to reasonable value."""
        self.learning_rate = 0.001
        self.instability_flag = False
    
    def _enable_augmentation(self):
        """Enable data augmentation strategy."""
        if self.augmentation_level == 0:
            self.augmentation_level = 2
            # Boost generalization metrics
            self.road_iou *= 1.05
            self.vegetation_iou *= 1.08
    
    def _increase_regularization(self):
        """Add L2 regularization to prevent overfitting."""
        self.regularization_strength = 0.0005
    
    def _switch_optimizer(self):
        """Switch from SGD to AdamW."""
        self.optimizer = "adamw"
    
    def _enable_class_balancing(self):
        """Enable focal loss and class reweighting for imbalanced classes."""
        if not self.class_balancing:
            self.class_balancing = True
            self.loss_weights = {"road": 1.0, "vegetation": 1.5, "sky": 3.0}
            self.class_imbalance_detected = False
    
    def _reduce_batch_size(self):
        """Reduce batch size for memory efficiency."""
        if self.batch_size > 8:
            self.batch_size = 16
    
    def _run_training_epoch(self):
        """
        Simulate one training epoch with realistic learning curves.
        """
        # Cost calculation
        base_cost = 5
        cost_modifier = 1.0
        if self.batch_size < 32:
            cost_modifier = 1.2  # Smaller batches = more iterations
        cost = int(base_cost * cost_modifier)
        
        if self.remaining_compute < cost:
            return
        
        self.remaining_compute = max(0, self.remaining_compute - cost)
        self.epoch += 1
        self.total_epochs_trained += 1
        
        # Simulate training with realistic dynamics
        if self.instability_flag:
            # Bad learning rate causes divergence
            self._epoch_with_instability()
        elif self.training_plateau:
            # Already converged, diminishing returns
            self._epoch_with_plateau()
        else:
            # Normal learning curve
            self._epoch_normal_training()
        
        # Overfitting risk increases with epochs
        self._evaluate_overfitting()
        self.iou_history.append(self.mean_iou)
        self.loss_history.append(self.val_loss)
    
    def _epoch_with_instability(self):
        """Training degrades without proper learning rate - IoU drops."""
        decay = 0.05 + (random.random() * 0.03)  # Unstable
        self.road_iou = max(0.10, min(1.0, self.road_iou - decay))
        self.vegetation_iou = max(0.00, min(1.0, self.vegetation_iou - decay * 0.5))
        self.sky_iou = max(0.00, min(1.0, self.sky_iou - decay * 0.3))
        self.train_loss += random.uniform(0.1, 0.3)
        self.val_loss += random.uniform(0.1, 0.3)
    
    def _epoch_with_plateau(self):
        """Minimal improvement after convergence - diminishing returns."""
        # Very small improvement on plateau
        improvement = random.uniform(0.0005, 0.002) * (1.0 + self.augmentation_level * 0.1)
        # Apply diminishing returns: improvement *= (1 - mean_iou)
        improvement *= (1.0 - self.mean_iou)
        
        self.road_iou = max(0.0, min(0.75, self.road_iou + improvement * 0.5))
        self.vegetation_iou = max(0.0, min(0.75, self.vegetation_iou + improvement * 0.3))
        if self.class_balancing:
            self.sky_iou = max(0.0, min(0.75, self.sky_iou + improvement * 0.2))
        self.train_loss *= 0.98
        self.val_loss *= 0.99
    
    def _epoch_normal_training(self):
        """Normal training with learning curve dynamics and diminishing returns."""
        base_improvement = 0.05
        
        # Learning curve starts steep, then plateaus (diminishing)
        progress_factor = min(1.0, (self.total_epochs_trained / 8.0))
        improvement = base_improvement * (1.0 - progress_factor * 0.6)
        
        # Apply diminishing returns based on current IoU (harder to improve near ceiling)
        improvement *= (1.0 - self.mean_iou)
        
        # Optimizer matters
        if self.optimizer == "adamw":
            improvement *= 1.15
        
        # Augmentation helps generalization
        if self.augmentation_level > 0:
            improvement *= 1.10
        
        # Regularization slows learning but prevents overfitting
        if self.regularization_strength > 0:
            improvement *= 0.95
        
        # Apply improvements with realistic class learning rates
        # Road class learns fastest (easier segmentation problem)
        self.road_iou = max(0.0, min(0.75, self.road_iou + improvement * 0.7))
        
        # Vegetation medium speed
        self.vegetation_iou = max(0.0, min(0.75, self.vegetation_iou + improvement * 0.4))
        
        # Sky class only learns with balancing, and much slower (harder class)
        if self.class_balancing:
            self.sky_iou = max(0.0, min(0.75, self.sky_iou + improvement * 0.2))
        else:
            # Without balancing, sky class stays at 0
            self.sky_iou = 0.0
        
        # Loss improvements
        self.train_loss *= 0.92
        self.val_loss *= 0.93
        
        # Check plateau threshold
        if self.total_epochs_trained >= 5 and improvement < 0.005:
            self.training_plateau = True
    
    def _evaluate_overfitting(self):
        """Evaluate whether model is overfitting based on loss divergence."""
        if len(self.loss_history) > 2:
            recent_train = self.train_loss
            recent_val = self.val_loss
            loss_gap = recent_val - recent_train
            
            # Overfitting detected if validation loss > training loss by margin
            if loss_gap > 0.3:
                self.overfitting_flag = True
            else:
                # Regularization helps prevent overfitting
                if self.regularization_strength > 0:
                    self.overfitting_flag = False
    
    def _update_status_flags(self):
        """Update status flags based on current state."""
        # Class imbalance detected if sky still at 0 without balancing
        if not self.class_balancing and self.sky_iou == 0.0:
            self.class_imbalance_detected = True
        else:
            self.class_imbalance_detected = False
    
    def get_observation(self) -> Observation:
        """
        Generate structured experiment report as observation.
        """
        report = self._generate_report()
        
        return Observation(
            report=report,
            mean_iou=self.mean_iou,
            road_iou=self.road_iou,
            vegetation_iou=self.vegetation_iou,
            sky_iou=self.sky_iou,
            epoch=self.epoch,
            compute_remaining_minutes=self.remaining_compute,
            overfitting=self.overfitting_flag,
            instability=self.instability_flag,
            early_stopped=self.early_stopped
        )
    
    def _generate_report(self) -> str:
        """Generate realistic ML experiment report."""
        lines = [
            "=" * 70,
            "OFFROAD SEGMENTATION EXPERIMENT REPORT",
            "=" * 70,
            f"Model: OffroadSegNet ResNet-{self.backbone_depth}",
            f"Dataset: Offroad Terrain Segmentation (Train + Val)",
            f"Epoch: {self.total_epochs_trained} | Compute Budget: {self.remaining_compute}m remaining",
            "",
            "CURRENT METRICS:",
            f"  Mean IoU:        {self.mean_iou:.4f}",
            f"  Road IoU:        {self.road_iou:.4f} (critical for navigation)",
            f"  Vegetation IoU:  {self.vegetation_iou:.4f}",
            f"  Sky IoU:         {self.sky_iou:.4f} (missing coverage)" if self.class_imbalance_detected else f"  Sky IoU:         {self.sky_iou:.4f}",
            f"  Train Loss:      {self.train_loss:.4f}",
            f"  Val Loss:        {self.val_loss:.4f}",
            "",
            "TRAINING CONFIGURATION:",
            f"  Optimizer:             {self.optimizer.upper()}",
            f"  Learning Rate:         {self.learning_rate}",
            f"  Batch Size:            {self.batch_size}",
            f"  Backbone Depth:        {self.backbone_depth}",
            f"  Feature Channels:      {self.feature_channels}",
            f"  L2 Regularization:     {self.regularization_strength}",
            f"  Data Augmentation:     {'Enabled' if self.augmentation_level > 0 else 'Disabled'}",
            f"  Class Balancing:       {'Enabled' if self.class_balancing else 'Disabled'}",
            f"  Mixed Precision:       {'Enabled' if self.mixed_precision else 'Disabled'}",
            "",
            "ISSUES DETECTED:",
        ]
        
        issues = []
        if self.instability_flag:
            issues.append("  WARNING: Training instability - suboptimal learning rate (try 'adjust_learning_rate')")
        if self.overfitting_flag:
            issues.append("  WARNING: Overfitting detected - validation loss diverging (try 'increase_regularization')")
        if self.training_plateau:
            issues.append("  INFO: Training plateau - gains diminishing after convergence")
        if self.class_imbalance_detected:
            issues.append("  WARNING: Sky class not learning - severe class imbalance (try 'enable_class_balancing')")
        if not self.class_balancing and self.class_imbalance_detected:
            issues.append("      Without balancing, background classes receive no gradient signals")
        
        if not issues:
            issues.append("  OK: No major issues detected")
        
        lines.extend(issues)
        
        lines.extend([
            "",
            "RECOMMENDATIONS:",
            f"  - Target IoU: {self.target_iou:.2f} | Gap: {max(0, self.target_iou - self.mean_iou):.4f}",
            "  - Run training epochs to improve metrics ('run_training_epoch')",
            "  - Adjust configuration to resolve issues (see above)",
            "=" * 70,
        ])
        
        return "\n".join(lines)
