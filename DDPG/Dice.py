import torch


class DiceCoefficient:
    def __init__(self, epsilon: float = 1e-6, multi_class: bool = False):
        """
        Initializes the Dice Coefficient class.

        Args:
        - epsilon (float): A small value to prevent division by zero in Dice calculation.
        - multi_class (bool): Whether to handle multi-class segmentation. Defaults to False (binary segmentation).
        """
        self.epsilon = epsilon
        self.multi_class = multi_class

    def calculate(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Dice coefficient between prediction and label for a segmentation task.

        Args:
        - prediction (torch.Tensor): The predicted mask with shape [batch_size, height, width].
        - label (torch.Tensor): The ground truth mask with shape [batch_size, height, width].

        Returns:
        - dice (torch.Tensor): The Dice coefficient for each sample in the batch (if multi_class=False, otherwise averaged across classes).
        """
        if self.multi_class:
            return self._multi_class_dice(prediction, label)
        else:
            return self._binary_dice(prediction, label)

    import torch

    def dice_coefficient_torch(self, pred, label, epsilon=1e-6):
        """
        Computes the Dice coefficient for segmentation tasks using PyTorch.

        Args:
            pred (torch.Tensor): Predicted segmentation mask of shape (batch, image_size).
            label (torch.Tensor): Ground truth segmentation mask of shape (batch, image_size).
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            torch.Tensor: Dice coefficient averaged over the batch.
        """
        # Flatten the image dimensions
        pred = pred.view(pred.size(0), -1)
        label = label.view(label.size(0), -1)

        # Compute the intersection and union
        intersection = (pred * label).sum(dim=1)
        union = pred.sum(dim=1) + label.sum(dim=1)

        # Compute Dice coefficient for each batch
        dice = (2. * intersection + epsilon) / (union + epsilon)

        # Return the mean Dice coefficient
        return dice.mean()

    def _binary_dice(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice coefficient for binary segmentation (one class).

        Args:
        - prediction (torch.Tensor): Predicted binary mask.
        - label (torch.Tensor): Ground truth binary mask.

        Returns:
        - dice (torch.Tensor): Dice coefficient for each sample in the batch.
        """
        # Flattening to perform element-wise operations over each pixel in the image
        prediction_flat = prediction.view(prediction.size(0), -1)
        label_flat = label.view(label.size(0), -1)

        # Compute intersection and union (binary)
        intersection = (prediction_flat * label_flat).sum(dim=1)
        union = prediction_flat.sum(dim=1) + label_flat.sum(dim=1)

        # Dice coefficient
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return dice

    def _multi_class_dice(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice coefficient for multi-class segmentation.

        Args:
        - prediction (torch.Tensor): Predicted mask with shape [batch_size, num_classes, height, width].
        - label (torch.Tensor): Ground truth mask with shape [batch_size, num_classes, height, width].

        Returns:
        - dice (torch.Tensor): Dice coefficient for each class averaged across the batch.
        """
        # Ensure prediction and label have the same shape
        assert prediction.shape == label.shape, "Prediction and label must have the same shape."

        # Flatten to handle the multi-class scenario
        batch_size, num_classes, height, width = prediction.shape
        prediction_flat = prediction.view(batch_size, num_classes, -1)
        label_flat = label.view(batch_size, num_classes, -1)

        # Compute intersection and union for each class
        intersection = (prediction_flat * label_flat).sum(dim=2)
        union = prediction_flat.sum(dim=2) + label_flat.sum(dim=2)

        # Dice coefficient for each class
        dice_per_class = (2. * intersection + self.epsilon) / (union + self.epsilon)

        # Return the mean Dice score across all classes (per batch)
        return dice_per_class.mean(dim=0)

    def batch_dice(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice coefficient averaged across the batch (useful for evaluation).

        Args:
        - prediction (torch.Tensor): Predicted mask with shape [batch_size, height, width] or [batch_size, num_classes, height, width].
        - label (torch.Tensor): Ground truth mask with shape [batch_size, height, width] or [batch_size, num_classes, height, width].

        Returns:
        - mean_dice (torch.Tensor): The average Dice score over the entire batch.
        """
        dice_scores = self.calculate(prediction, label)
        return dice_scores.mean()

    def __call__(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        This method allows the class instance to be called directly like a function.
        """
        return self.dice_coefficient_torch(prediction, label)
