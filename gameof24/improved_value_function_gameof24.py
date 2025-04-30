import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import ast

df = pd.read_csv("gameof24/24game_problems.csv")
game_of_24_quads = [ast.literal_eval(x) for x in df['numbers']]

class EnhancedValueNetwork(nn.Module):
    def __init__(self):
        super(EnhancedValueNetwork, self).__init__()
        # Input features: 4 numbers + 6 possible operations between pairs + statistics
        self.fc1 = nn.Linear(16, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        # First 4 values are the numbers
        numbers = x[:, :4]
        
        # Extract features
        features = self.extract_features(numbers)
        
        # Combined input
        x = torch.cat([numbers, features], dim=1)
        
        # Forward pass
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x
    
    def extract_features(self, numbers):
        batch_size = numbers.shape[0]
        features = torch.zeros(batch_size, 12, device=numbers.device)
        
        # For each state, calculate all possible pair operations
        for i in range(batch_size):
            nums = numbers[i]
            
            # Get non-zero numbers
            valid_nums = nums[nums != 0]
            if len(valid_nums) < 2:
                continue
                
            # Calculate features for each possible pair of numbers
            feature_idx = 0
            
            # Calculate all possible pair operations
            for j in range(len(valid_nums)):
                for k in range(j+1, len(valid_nums)):
                    a, b = valid_nums[j], valid_nums[k]
                    
                    # Check if any operation gets us closer to 24
                    if feature_idx < 12:  # Ensure we don't exceed feature dimensions
                        # How close addition gets us to 24
                        features[i, feature_idx] = 1.0 - min(abs(a + b - 24) / 24, 1.0)
                        feature_idx += 1
                        
                    if feature_idx < 12:
                        # How close multiplication gets us to 24
                        features[i, feature_idx] = 1.0 - min(abs(a * b - 24) / 24, 1.0)
                        feature_idx += 1
                        
                    if feature_idx < 12:
                        # How close subtraction gets us to 24
                        features[i, feature_idx] = 1.0 - min(abs(a - b - 24) / 24, 1.0)
                        feature_idx += 1
                        
                    if feature_idx < 12:
                        # How close division gets us to 24 (if possible)
                        if b != 0:
                            features[i, feature_idx] = 1.0 - min(abs(a / b - 24) / 24, 1.0)
                        feature_idx += 1
                    
                    if feature_idx >= 12:
                        break
                if feature_idx >= 12:
                    break
        
        return features

def simulate_paths(quad, max_depth = 3):
  """
  Simulates all possible paths that the model can take from a single set of four numbers.
  Uses every possible pairing of numbers with every operation.
  --------
  Implementation Details
  We aren't using a visited dictionary because we want to visit the same state multiple times.
  (idk if this is actually a good idea)
  The same state could be on different sides of the tree and thus higher values
  mean that there are more ways to achieve 24. By the same token, if some state only has a single
  way to reach 24, it won't get a high score.
  """
  labels = {} # key:state, value: score
  path_stack = []

  def dfs(state, depth):
    # sorted ensures that states are order agnostic and we aren't learning different
    # weights for states that are logically the same
    state_sorted = tuple(sorted(float(x) for x in state))

    path_stack.append(state_sorted)
    if state_sorted not in labels:
      labels[state_sorted] = 0

    if len(state) == 1:
      # if we're using floats we need to allow some wiggle room
      hit_24 = abs(state[0] - 24) < 1e-6
      # mark parents as success (this is why we're doing dfs not bfs)
      if hit_24:
        for idx, s in enumerate(reversed(path_stack)):
           if idx == 0:
              reward = 1.0
           elif idx == 1:
              reward = 0.75
           elif idx == 2:
              reward = 0.5
           else:
              break
           labels[s] = labels.get(s, 0.0) + reward
      path_stack.pop()
      return

    # get all pairs of numbers
    for (i, j) in itertools.combinations(range(len(state)), 2):
      a, b = state[i], state[j]

      remaining_numbers = [state[k] for k in range(len(state)) if k != i and k != j]
      candidates = []
      candidates.append(a + b)
      candidates.append(a - b)
      candidates.append(b - a)
      candidates.append(a * b)
      if a != 0:
        candidates.append(b / a)
      if b != 0:
        candidates.append(a / b)

      for new_num in candidates:
        new_state = remaining_numbers + [new_num]
        dfs(new_state, depth - 1)

    path_stack.pop()

  dfs(quad, max_depth)

  if labels:
    max_val = max(labels.values())
    if max_val > 0:
        for s in labels:
            labels[s] /= max_val
  return labels

def train_enhanced_value_network(X_train, y_train, epochs=20, batch_size=64):
    model = EnhancedValueNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    loss_fn = nn.BCELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Class weights to handle imbalance
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)])
    weighted_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training/validation split
    indices = torch.randperm(len(X_train))
    train_idx, val_idx = indices[:int(0.8*len(indices))], indices[int(0.8*len(indices)):]
    
    train_x, train_y = X_train[train_idx], y_train[train_idx]
    val_x, val_y = X_train[val_idx], y_train[val_idx]
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Create batches with random permutation
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = train_x[idx]
            batch_y = train_y[idx]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = weighted_loss(preds, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = weighted_loss(val_preds, val_y).item()
            
            # Binary predictions for accuracy calculation
            binary_preds = (val_preds > 0.5).float()
            accuracy = (binary_preds == val_y).float().mean().item()
            
            # Calculate precision, recall, and F1 for positive class
            true_positives = ((binary_preds == 1) & (val_y == 1)).sum().item()
            predicted_positives = (binary_preds == 1).sum().item()
            actual_positives = (val_y == 1).sum().item()
            
            precision = true_positives / max(predicted_positives, 1)
            recall = true_positives / max(actual_positives, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        avg_loss = total_loss / len(train_x)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Accuracy={accuracy:.4f}, F1={f1:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    return model

def pad(state):
  state = list(state)
  return [0] * (4 - len(state)) + state

def generate_improved_training_data(data, max_depth=3):
    """Generate more diverse training data with augmentation techniques"""
    X = []
    Y = []

    for quad in data:
        reachable_states = simulate_paths(quad, max_depth)
        for state, value in reachable_states.items():
            # Store the original state
            X.append(pad(state))
            Y.append(value)
            
            # Add permutations of the state for more robust training
            if len(state) > 1:
                for _ in range(3):  # Add 3 random permutations
                    perm_state = list(state)
                    random.shuffle(perm_state)
                    X.append(pad(tuple(perm_state)))
                    Y.append(value)
    
    return X, Y

States, Values = generate_improved_training_data(game_of_24_quads)

# Separate positives and negatives
positives = [x for x, y in zip(States, Values) if y > 0]
positive_labels = [y for y in Values if y > 0]

negatives = [x for x, y in zip(States, Values) if y == 0]
negative_labels = [0 for _ in negatives]

# Oversample positives to boost their ratio (e.g., make up 25% of dataset)
import random
pos_sample_count = int(0.25 * len(negatives))  # adjust as needed
pos_sampled = random.choices(list(zip(positives, positive_labels)), k=pos_sample_count)
pos_X, pos_Y = zip(*pos_sampled)

# Combine
X_final = negatives + list(pos_X)
Y_final = negative_labels + list(pos_Y)

model = train_enhanced_value_network(States, Values, epochs=10, batch_size=32)
torch.save(model.state_dict(), 'weighted_loss_value_network.pth')