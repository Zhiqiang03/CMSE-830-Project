import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MLP(nn.Module):
    """Multi-Layer Perceptron for taxi tip classification"""

    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (e.g., [128, 64, 32])
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_mlp_model(model_path, device='cpu') -> MLP:
    """Load a saved MLP model"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    model = MLP(config['input_size'], config['hidden_sizes'], config['num_classes'], config['dropout_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def train_mlp(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """
    Train the MLP model

    Args:
        model: The MLP model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on

    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]', leave=False)

        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]', leave=False)

            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch summary
        print(f'Epoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print()

    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path='mlp_training_history.png'):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(val_losses, label='Validation Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracies
    axes[1].plot(train_accuracies, label='Train Accuracy', marker='o', markersize=3)
    axes[1].plot(val_accuracies, label='Validation Accuracy', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.show()


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model on test data"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc='Evaluating'):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%\n")

    # Per-class accuracy
    print("Per-Class Accuracy:")
    for i in range(3):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == all_labels[mask]).sum() / mask.sum()
            class_name = 'Low Tip' if i==0 else 'Mid Tip' if i==1 else 'High Tip'
            print(f"  Class {i} ({class_name}): {class_acc * 100:.2f}%")

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions,
                                target_names=['Low Tip (0)', 'Middle Tip (1)', 'High Tip (2)']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low (0)', 'Middle (1)', 'High (2)'],
                yticklabels=['Low (0)', 'Middle (1)', 'High (2)'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to mlp_confusion_matrix.png")
    plt.show()

    return accuracy, all_predictions, all_labels

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    data_path = 'data/processed_taxi_data.pt'
    data = torch.load(data_path)

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(torch.unique(y_train))}\n")

    # 2. Create DataLoaders
    batch_size = 64

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Batch size: {batch_size}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}\n")

    # 3. Model Configuration
    input_size = X_train.shape[1]
    hidden_sizes = [256, 128, 64, 32]  # Three hidden layers
    num_classes = len(torch.unique(y_train))
    dropout_rate = 0.2

    # 4. Initialize Model
    model = MLP(input_size, hidden_sizes, num_classes, dropout_rate).to(device)
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # 5. Loss Function and Optimizer
    class_counts = torch.bincount(y_train.long())
    total_samples = len(y_train)

    print("\nClass Distribution:")
    for i, count in enumerate(class_counts):
        percentage = 100 * count / total_samples
        print(f"  Class {i} (Tip {'Low' if i==0 else 'Mid' if i==1 else 'High'}): {count:,} samples ({percentage:.2f}%)")

    imbalance_ratio = class_counts.max().item() / class_counts.min().item()
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

    # Calculate class weights using inverse frequency
    # This gives higher weight to minority classes
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(device)

    print(f"\nClass Weights Applied:")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f}x")
    print("="*60 + "\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 6. Train Model
    print("Starting training...\n")
    num_epochs = 20

    train_losses, val_losses, train_accuracies, val_accuracies = train_mlp(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=num_epochs, device=device
    )

    # Update learning rate based on validation loss
    # Note: In a real scenario, you'd use a separate validation set
    # scheduler.step(val_losses[-1])

    # 7. Plot Training History
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    # 8. Evaluate Model
    accuracy, predictions, labels = evaluate_model(model, test_loader, device)

    # 9. Save Model
    model_path = 'models/mlp_tip_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'accuracy': accuracy,
        'model_config': {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate
        }
    }, model_path)
    print(f"\nModel saved to {model_path}")


