# Part 3: Ethics & Optimization

## 1. Ethical Considerations

### Potential Biases in Models

**MNIST Model:**
- Demographic bias: Handwriting style variations by age/education/culture
- Geographic bias: Underrepresentation of non-Western digit writing styles
- Stylistic bias: Performance drops on unconventional digit formations

**Amazon Reviews Model:**
- Linguistic bias: Better performance on formal English vs. slang/AAVE/non-native English
- Product category bias: Overrepresentation of popular categories (electronics)
- Sentiment bias: Difficulty with sarcasm, cultural context, and emoji-based sentiment

### Bias Mitigation Tools

**TensorFlow Fairness Indicators:**
```python
# Example implementation
fairness_indicators.evaluate(
    model=model,
    data=test_dataset,
    sensitive_features=test_labels['demographic_metadata'],
    metrics=['accuracy', 'false_positive_rate']
)
```
- Quantifies performance gaps across demographic groups
- Generates disparity visualization dashboards
- Identifies error rate differences (e.g., +15% misclassification for elderly users)

**spaCy Rule-Based Systems:**
```python
# Example slang normalization
matcher.add("SLANG", [[{"LOWER": {"IN": ["dope", "fire", "mid"]}}]])
def normalize_text(text):
    doc = nlp(text)
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        text = text.replace(span.text, "excellent" if span.text in ["dope","fire"] else "average")
    return text
```
- Standardizes linguistic variations (slang, cultural expressions)
- Anonymizes demographic references via entity recognition
- Augments training data through rule-based paraphrasing

## 2. Troubleshooting Challenge

### Buggy TensorFlow Code
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28)),  # Error 1
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Error 2
    metrics=['accuracy']
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255

model.fit(
    x_train, 
    y_train,  # Error 3
    epochs=5
)
```

### Identified Errors & Fixes

1. **Input Shape Mismatch**:
   - ❌ Missing channel dimension: `input_shape=(28,28)`
   - ✅ Fix: `input_shape=(28,28,1)` 

2. **Loss Function Mismatch**:
   - ❌ `categorical_crossentropy` requires one-hot labels
   - ✅ Fix: Use `sparse_categorical_crossentropy` for integer labels

3. **Label Shape Incompatibility**:
   - ❌ `y_train` shape (60000,) vs required (60000,10)
   - ✅ Fix: Either:
     - Change loss to `sparse_categorical_crossentropy` OR
     - One-hot encode: `tf.keras.utils.to_categorical(y_train)`

### Fixed Code
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # Fixed
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Fixed
    metrics=['accuracy']
)

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255  # Dynamic reshaping

model.fit(
    x_train,
    y_train,  # Now compatible with sparse loss
    epochs=5
)
```

### Debugging Principles
1. **Shape Verification**: Always check layer input/output dimensions
2. **Loss-Label Compatibility**:
   - `categorical_crossentropy` → one-hot encoded labels
   - `sparse_categorical_crossentropy` → integer labels
3. **Channel Awareness**: Grayscale images require explicit channel dimension
4. **Dynamic Reshaping**: Use `-1` in reshape to handle arbitrary batch sizes
