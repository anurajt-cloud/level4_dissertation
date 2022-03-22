import tensorflow as tf
import numpy as np


def interpolate_signal(baseline, signal, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(signal, axis=0)
    delta = input_x - baseline_x
    signals = baseline_x + alphas_x * delta
    return signals

def compute_gradients(model, signals, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(signals)
        logits = model(signals)
        probs = tf.nn.softmax(logits, axis=-1)[:,target_class_idx]
    return tape.gradient(probs, signals)

def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def integrated_gradients(model, baseline, signal, target_class_idx, m_steps=50, batch_size=32):
    # 1. generate alphas
    alphas = tf.cast(tf.linspace(0.0, 1.0, m_steps+1), tf.float32)
    
    # Initialize tensorarray outside loop to collect gradients.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
    
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        
        # 2. Generate interpolated signals between baseline and input.
        interpolated_path_input_batch = interpolate_signal(baseline, signal, alpha_batch)
        print(target_class_idx.shape)
        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(model, interpolated_path_input_batch, target_class_idx)
        
        # Write batch indices and gradients to extend TensorArray
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)
    
    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()
    
    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(total_gradients)
    
    # 5 Scale integrated gradients with respect to input.
    integrated_gradients = (signal - baseline) * avg_gradients
    # integrated_gradients = integrated_gradients*(tf.reduce_max(signal)/tf.reduce_max(integrated_gradients))
    # Converting 2D tensor to 1D vetor.
    # attribution_mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1)
    return integrated_gradients #attribution_mask
    

def calculate_ig(model, beats, class_indexes):
    # print("calig", class_indexes.shape)
    #converting beats & class_indexes into float32
    # beats = tf.cast(beats, tf.float32)
    class_indexes = tf.cast(class_indexes, tf.int64)
    
    
    baseline = tf.zeros((920,1), dtype=tf.float32)
    # alphas = tf.cast(tf.linspace(0.0, 1.0, 51), tf.float32)
    # signals = interpolate_signal(baseline, signal, alphas)
    
    
    igs = np.array([integrated_gradients(model, baseline, b, c, 50, 32) for b,c in zip(beats,class_indexes)])
    # print(igs.shape)
    return igs
    
