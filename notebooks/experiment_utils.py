"""Shared TensorFlow vehicle-damage experiment helpers.

Images enter preprocessing as RGB pixels in [0, 255]. Models output logits.
Localization uses the selected annotation mask at original crop resolution.
Grad-CAM caches store small native maps to keep memory use modest on a Mac.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from pycocotools.coco import COCO
from scipy.stats import spearmanr

CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
SPLIT_FOLDERS = {'train': 'train2017', 'val': 'val2017', 'test': 'test2017'}
GRADCAM_LAYER = 'conv5_block3_out'


def save_json(path, value):
    """Write a JSON artifact, rejecting nonfinite values."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def load_manifest(path, split=None):
    """Read and validate crop metadata; preserve CSV row order."""
    df = pd.read_csv(path)
    required = {'sample_id', 'split', 'image_id', 'annotation_id', 'source_path',
                'category_id', 'category', 'class_index', 'left', 'top', 'right', 'bottom',
                'crop_width', 'crop_height', 'padding_fraction_per_side'}
    if missing := required - set(df.columns):
        raise ValueError(f'Missing manifest columns: {sorted(missing)}')
    if df[list(required)].isna().any().any() or not df.sample_id.is_unique:
        raise ValueError('Manifest has missing values or duplicate sample IDs.')
    if not set(df.split).issubset(SPLIT_FOLDERS):
        raise ValueError('Unknown dataset split.')
    geometry = df[['left', 'top', 'right', 'bottom', 'crop_width', 'crop_height']]
    if not np.isfinite(geometry.to_numpy()).all() or not (geometry == np.floor(geometry)).all().all():
        raise ValueError('Crop coordinates must be finite integers.')
    if ((df.left < 0) | (df.top < 0) | (df.right <= df.left) | (df.bottom <= df.top)
        | (df.crop_width != df.right - df.left) | (df.crop_height != df.bottom - df.top)).any():
        raise ValueError('Invalid crop geometry.')
    if not all(0 <= i < len(CLASS_NAMES) and CLASS_NAMES[int(i)] == c and i == int(i)
               for i, c in zip(df.class_index, df.category)):
        raise ValueError('Class order differs from CLASS_NAMES.')
    return df.loc[df.split == split].reset_index(drop=True) if split else df.reset_index(drop=True)


def load_annotations(data_dir, splits=('train', 'val', 'test')):
    """Return split-specific COCO indexes (IDs need not be globally unique)."""
    return {s: COCO(str(Path(data_dir) / 'annotations' / f'instances_{SPLIT_FOLDERS[s]}.json'))
            for s in splits}


def load_crop(row):
    """Return uint8 RGB crop without resizing or normalization."""
    box = tuple(int(row[k]) for k in ('left', 'top', 'right', 'bottom'))
    with Image.open(row['source_path']) as image:
        if box[2] > image.width or box[3] > image.height:
            raise ValueError(f'Crop exceeds image bounds: {row["sample_id"]}')
        return np.asarray(image.convert('RGB').crop(box))


def load_damage_mask(row, annotations):
    """Decode the selected instance, then crop with identical image coordinates."""
    coco = annotations[row['split']]
    ann = coco.anns[int(row['annotation_id'])]
    if ann['image_id'] != int(row['image_id']) or ann['category_id'] != int(row['category_id']):
        raise ValueError(f'Annotation mismatch: {row["sample_id"]}')
    mask = coco.annToMask(ann)[int(row['top']):int(row['bottom']),
                              int(row['left']):int(row['right'])].astype(bool)
    if mask.shape != (int(row['crop_height']), int(row['crop_width'])) or not mask.any():
        raise ValueError(f'Empty or misaligned damage mask: {row["sample_id"]}')
    return mask


def preprocess_crop(crop, target_size=(224, 224)):
    """Bilinear resize (height, width), then ResNet50 preprocessing exactly once."""
    image = tf.image.resize(tf.cast(crop, tf.float32), target_size)
    return tf.keras.applications.resnet50.preprocess_input(image)


def make_dataset(samples, batch_size=32, target_size=(224, 224), training=False, seed=42):
    """Build ordered evaluation data or shuffled, augmented training data.

    Training augmentation affects only classification inputs. Localization always
    reloads unaugmented crops/masks. A single mapping worker limits Mac memory use.
    """
    if samples.empty:
        raise ValueError('Cannot build an empty dataset.')
    rows = list(samples.to_dict('records'))
    def generator():
        for row in rows:
            yield load_crop(row), np.int32(row['class_index'])
    ds = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec((None, None, 3), tf.uint8), tf.TensorSpec((), tf.int32)))
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(rows)))
    if training:
        ds = ds.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal', seed=seed),
        tf.keras.layers.RandomRotation(0.03, fill_mode='reflect', seed=seed + 1),
        tf.keras.layers.RandomContrast(0.15, seed=seed + 2),
    ]) if training else None
    def prepare(image, label):
        image = tf.image.resize(tf.cast(image, tf.float32), target_size)
        if augmentation is not None:
            image = tf.clip_by_value(augmentation(image, training=True), 0., 255.)
        return tf.keras.applications.resnet50.preprocess_input(image), label
    return ds.map(prepare, num_parallel_calls=1, deterministic=True).batch(batch_size).prefetch(1)


def plot_class_distribution(samples):
    counts = pd.crosstab(samples.category, samples.split).reindex(CLASS_NAMES, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    counts.plot.bar(ax=axes[0], ylabel='Crops', title='Class counts')
    (counts.div(counts.sum(axis=0), axis=1) * 100).plot.bar(
        ax=axes[1], ylabel='Percent', title='Class distribution within each split')
    for ax in axes:
        ax.set_xlabel(''); ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    return fig, counts


def calculate_class_weights(train_samples):
    if set(train_samples.split) != {'train'}:
        raise ValueError('Class weights must use training samples only.')
    counts = np.bincount(train_samples.class_index, minlength=len(CLASS_NAMES))
    if (counts == 0).any():
        raise ValueError('Every class needs training samples.')
    return {i: float(len(train_samples) / (len(counts) * n)) for i, n in enumerate(counts)}


def build_resnet50(target_size=(224, 224), num_classes=6, weights='imagenet', dropout=0.3):
    """Flat functional graph exposes the target convolution layer for Grad-CAM."""
    backbone = tf.keras.applications.ResNet50(include_top=False, weights=weights,
                                             input_shape=(*target_size, 3))
    backbone.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_pool')(backbone.output)
    x = tf.keras.layers.Dropout(dropout, name='head_dropout')(x)
    logits = tf.keras.layers.Dense(num_classes, name='class_logits')(x)
    return tf.keras.Model(backbone.input, logits, name='damage_resnet50')


def configure_fine_tuning(model, fine_tune=False, learning_rate=1e-3):
    """Train head only, or head plus conv5; keep BatchNorm statistics frozen."""
    for layer in model.layers:
        layer.trainable = layer.name == 'class_logits' or (fine_tune and layer.name.startswith('conv5_'))
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])


def plot_training_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    offset = 0
    for stage, history in histories.items():
        n = len(history['loss']); epochs = np.arange(offset + 1, offset + n + 1)
        for ax, metric in zip(axes, ('loss', 'accuracy')):
            ax.plot(epochs, history[metric], label=f'{stage}: train')
            ax.plot(epochs, history[f'val_{metric}'], label=f'{stage}: validation')
            ax.set(xlabel='Epoch', ylabel=metric); ax.legend()
        offset += n
    fig.tight_layout()
    return fig


def evaluate_classification(model, dataset, samples, class_names=CLASS_NAMES):
    """Return per-sample probabilities, all-class metrics and confusion matrix.

    Dataset must be unshuffled and correspond to samples in the same order.
    Macro metrics include every class; zero denominators produce zero scores.
    """
    logits = model.predict(dataset, verbose=1)
    if len(logits) != len(samples):
        raise ValueError('Prediction count does not match manifest.')
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    true = samples.class_index.to_numpy(dtype=int); predicted = probs.argmax(axis=1)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    tp = matrix.diagonal(); support = matrix.sum(axis=1); predicted_count = matrix.sum(axis=0)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp, dtype=float), where=predicted_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    report = pd.DataFrame({'class': class_names, 'precision': precision, 'recall': recall,
                           'f1': f1, 'support': support})
    metrics = {'accuracy': float(tp.sum() / support.sum())}
    for name, values in [('precision', precision), ('recall', recall), ('f1', f1)]:
        metrics[f'macro_{name}'] = float(values.mean())
        metrics[f'weighted_{name}'] = float(np.average(values, weights=support))
    predictions = samples[['sample_id', 'image_id', 'annotation_id', 'category', 'class_index']].copy()
    predictions['predicted_index'] = predicted
    predictions['predicted_category'] = [class_names[i] for i in predicted]
    predictions['correct'] = true == predicted
    for i, name in enumerate(class_names):
        predictions[f'probability_{name}'] = probs[:, i]
    return predictions, report, metrics, matrix


def plot_confusion_matrix(matrix, class_names=CLASS_NAMES, normalize=False):
    values = matrix.astype(float)
    if normalize:
        values = np.divide(values, values.sum(axis=1, keepdims=True), out=np.zeros_like(values),
                           where=values.sum(axis=1, keepdims=True) != 0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, cmap='Blues'); fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted class', ylabel='True class')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{values[i,j]:.2f}' if normalize else str(matrix[i,j]), ha='center',
                    color='white' if values[i,j] > values.max() / 2 else 'black')
    fig.tight_layout()
    return fig


def _gradcam_model(model, layer_name):
    return tf.keras.Model(model.input, [model.get_layer(layer_name).output, model.output])


def _gradcam_batch(grad_model, images, targets=None):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as tape:
        # Watching inputs also makes frozen backbone operations differentiable.
        tape.watch(images)
        features, logits = grad_model(images, training=False)
        indices = tf.argmax(logits, axis=1, output_type=tf.int32) if targets is None else tf.convert_to_tensor(targets, tf.int32)
        scores = tf.reduce_sum(tf.gather(logits, indices, axis=1, batch_dims=1))
    gradients = tape.gradient(scores, features)
    if gradients is None:
        raise ValueError('Target layer is disconnected from class logits.')
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    maps = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1)).numpy()
    finite = np.isfinite(maps).all(axis=(1, 2))
    maxima = np.max(np.where(np.isfinite(maps), maps, 0), axis=(1, 2))
    informative = finite & (maxima > 1e-12)
    maps = np.divide(maps, maxima[:, None, None], out=np.zeros_like(maps),
                     where=informative[:, None, None])
    maps[~informative] = 0
    return maps.astype(np.float32), logits.numpy(), informative


def resize_heatmap(heatmap, shape):
    return np.asarray(Image.fromarray(np.asarray(heatmap, dtype=np.float32)).resize(
        (int(shape[1]), int(shape[0])), Image.Resampling.BILINEAR))


def generate_gradcam(model, crop, target_class=None, target_size=(224, 224), layer_name=GRADCAM_LAYER):
    """Return normalized Grad-CAM at crop resolution; target=None uses prediction."""
    maps, _, _ = _gradcam_batch(_gradcam_model(model, layer_name),
        preprocess_crop(crop, target_size)[None], None if target_class is None else [target_class])
    return resize_heatmap(maps[0], crop.shape[:2])


def collect_gradcam(model, samples, target_size=(224, 224), batch_size=32,
                    target_mode='ground_truth', layer_name=GRADCAM_LAYER):
    """Cache native heatmaps and sample IDs; incorrect predictions are included."""
    if target_mode not in ('ground_truth', 'predicted') or samples.empty:
        raise ValueError('Choose ground_truth/predicted and nonempty samples.')
    grad_model = _gradcam_model(model, layer_name)
    maps, logits, valid, targets = [], [], [], []
    for start in range(0, len(samples), batch_size):
        batch = samples.iloc[start:start + batch_size]
        images = tf.stack([preprocess_crop(load_crop(row), target_size) for row in batch.to_dict('records')])
        indices = batch.class_index.to_numpy() if target_mode == 'ground_truth' else None
        hm, scores, informative = _gradcam_batch(grad_model, images, indices)
        maps.append(hm); logits.append(scores); valid.append(informative)
        targets.append(indices if indices is not None else scores.argmax(axis=1))
        if start == 0 or (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(samples):
            print(f'Grad-CAM: {min(start + batch_size, len(samples))}/{len(samples)}')
    return {'sample_ids': samples.sample_id.to_numpy(dtype=str), 'heatmaps': np.concatenate(maps),
            'logits': np.concatenate(logits), 'informative': np.concatenate(valid),
            'target_classes': np.concatenate(targets), 'target_mode': np.asarray(target_mode),
            'layer_name': np.asarray(layer_name)}


def _check_cache(samples, cache):
    if not np.array_equal(samples.sample_id.to_numpy(dtype=str), cache['sample_ids']):
        raise ValueError('Grad-CAM cache and sample order differ.')


def calculate_iou(heatmap, mask, threshold):
    if not 0 < threshold <= 1:
        raise ValueError('Threshold must be in (0, 1].')
    mask = np.asarray(mask, dtype=bool)
    if heatmap.shape != mask.shape or not mask.any():
        raise ValueError('Heatmap and nonempty mask must have the same shape.')
    if not np.isfinite(heatmap).all() or heatmap.max() <= 1e-12:
        return 0.0
    predicted = heatmap >= threshold
    return float(np.logical_and(predicted, mask).sum() / np.logical_or(predicted, mask).sum())


def pointing_game(heatmap, mask):
    """Empty/nonfinite maps fail. For ties use the first maximum in row order."""
    if heatmap.shape != mask.shape or not np.asarray(mask).any():
        raise ValueError('Heatmap and nonempty mask must have the same shape.')
    if not np.isfinite(heatmap).all() or heatmap.max() <= 1e-12:
        return False
    return bool(mask[np.unravel_index(np.argmax(heatmap), heatmap.shape)])


def evaluate_localization(samples, annotations, cache, thresholds):
    """Score original-resolution masks; return each sample/threshold and summaries."""
    _check_cache(samples, cache)
    thresholds = sorted(set(float(t) for t in thresholds))
    if not thresholds or any(not 0 < t <= 1 for t in thresholds):
        raise ValueError('Provide thresholds in (0, 1].')
    records = []
    for i, row in enumerate(samples.to_dict('records')):
        mask = load_damage_mask(row, annotations)
        heatmap = resize_heatmap(cache['heatmaps'][i], mask.shape)
        valid = bool(cache['informative'][i])
        hit = pointing_game(heatmap, mask) if valid else False
        for threshold in thresholds:
            records.append({'sample_id': row['sample_id'], 'category': row['category'],
                'true_index': int(row['class_index']), 'predicted_index': int(cache['logits'][i].argmax()),
                'target_class': int(cache['target_classes'][i]), 'target_mode': str(cache['target_mode']),
                'threshold': threshold, 'iou': calculate_iou(heatmap, mask, threshold) if valid else 0.,
                'pointing_hit': hit, 'informative': valid, 'whole_crop_iou': float(mask.mean())})
    results = pd.DataFrame(records)
    summary = results.groupby('threshold', as_index=False).agg(
        mean_iou=('iou', 'mean'), pointing_accuracy=('pointing_hit', 'mean'),
        informative_fraction=('informative', 'mean'), samples=('sample_id', 'size'),
        whole_crop_mean_iou=('whole_crop_iou', 'mean'))
    macro = results.groupby(['threshold', 'category']).iou.mean().groupby('threshold').mean()
    summary['macro_class_iou'] = summary.threshold.map(macro)
    return results, summary.sort_values(['mean_iou', 'threshold'], ascending=[False, True]).reset_index(drop=True)


def select_validation_threshold(samples, annotations, cache, thresholds):
    """Rank ONLY validation thresholds by mean IoU; ties favor lower threshold."""
    if set(samples.split) != {'val'}:
        raise ValueError('Threshold selection must use validation samples only.')
    return evaluate_localization(samples, annotations, cache, thresholds)


def balanced_sample(samples, per_class=2, seed=42):
    if per_class <= 0:
        raise ValueError('per_class must be positive.')
    return pd.concat([g.sample(min(per_class, len(g)), random_state=seed)
                      for _, g in samples.groupby('class_index')]).reset_index(drop=True)


def subset_cache(samples, cache):
    positions = {sid: i for i, sid in enumerate(cache['sample_ids'])}
    indices = [positions[sid] for sid in samples.sample_id]
    return {k: np.asarray(v)[indices] if np.asarray(v).ndim > 0 else v for k, v in cache.items()}


def plot_gradcam(crop, heatmap, mask=None, threshold=None, title='Grad-CAM'):
    columns = 2 + int(mask is not None) + int(threshold is not None)
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    axes[0].imshow(crop); axes[0].set_title('Image crop')
    axes[1].imshow(crop); axes[1].imshow(heatmap, cmap='jet', alpha=.45, vmin=0, vmax=1)
    axes[1].set_title(title)
    position = 2
    if mask is not None:
        axes[position].imshow(mask, cmap='gray'); axes[position].set_title('Selected damage mask'); position += 1
    if threshold is not None:
        axes[position].imshow(heatmap >= threshold, cmap='gray')
        axes[position].set_title(f'Threshold = {threshold:.2f}')
    for ax in axes: ax.axis('off')
    fig.tight_layout()
    return fig


def plot_threshold_comparison(samples, annotations, cache, thresholds):
    """Use identical samples for every candidate threshold."""
    _check_cache(samples, cache)
    thresholds = list(thresholds)
    fig, axes = plt.subplots(len(samples), 3 + len(thresholds),
        figsize=(3 * (3 + len(thresholds)), 3 * len(samples)), squeeze=False)
    for i, row in enumerate(samples.to_dict('records')):
        crop = load_crop(row); mask = load_damage_mask(row, annotations)
        heatmap = resize_heatmap(cache['heatmaps'][i], mask.shape)
        axes[i, 0].imshow(crop); axes[i, 0].set_title(row['category'])
        axes[i, 1].imshow(mask, cmap='gray'); axes[i, 1].set_title('Damage mask')
        axes[i, 2].imshow(crop); axes[i, 2].imshow(heatmap, cmap='jet', alpha=.45, vmin=0, vmax=1)
        axes[i, 2].set_title('Grad-CAM')
        for j, threshold in enumerate(thresholds, 3):
            axes[i, j].imshow(heatmap >= threshold, cmap='gray')
            axes[i, j].set_title(f'{threshold:.2f} | IoU {calculate_iou(heatmap, mask, threshold):.3f}')
        for ax in axes[i]: ax.axis('off')
    fig.tight_layout()
    return fig


def compare_heatmaps(original, randomized):
    """Signed Spearman correlation and MAE; undefined correlation returns None."""
    a, b = np.asarray(original).ravel(), np.asarray(randomized).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Heatmaps must be finite and have matching shapes.')
    correlation = None
    if np.ptp(a) > 1e-12 and np.ptp(b) > 1e-12:
        correlation = float(spearmanr(a, b).statistic)
    return {'spearman': correlation, 'mae': float(np.mean(np.abs(a - b)))}


def run_randomization_sanity_check(model, samples, target_size=(224, 224), batch_size=32,
                                   layer_name=GRADCAM_LAYER, seed=42, baseline_cache=None,
                                   output_dir=None):
    """Cascading randomization: head -> conv5 -> conv4 -> conv3 -> conv2 -> conv1.

    Reset kernels with each layer's configured initializer and BatchNorm state
    to initialization defaults. Use a fresh clone; original weights are untouched.
    Target classes stay fixed at baseline. Save each stage before continuing so
    long test runs retain completed results. Correlation is computed at input size.
    """
    if samples.empty:
        raise ValueError('Sanity sample is empty.')
    baseline = baseline_cache or collect_gradcam(model, samples, target_size, batch_size, layer_name=layer_name)
    _check_cache(samples, baseline)
    clone = tf.keras.models.clone_model(model); clone.set_weights(model.get_weights())
    tf.keras.utils.set_random_seed(seed)
    groups = [('head', lambda n: n == 'class_logits')]
    groups += [(stage, lambda n, stage=stage: n.startswith(stage + '_'))
               for stage in ('conv5', 'conv4', 'conv3', 'conv2', 'conv1')]
    records, previews = [], {}
    if output_dir is not None:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for stage, matches in groups:
        reset_names = []
        for layer in clone.layers:
            if matches(layer.name) and layer.get_weights():
                fresh = layer.__class__.from_config(layer.get_config())
                fresh.build(layer.input.shape)
                layer.set_weights(fresh.get_weights()); reset_names.append(layer.name)
        if not reset_names:
            raise ValueError(f'No weights randomized for {stage}.')
        grad_model = _gradcam_model(clone, layer_name)
        stage_maps = []
        for start in range(0, len(samples), batch_size):
            batch = samples.iloc[start:start + batch_size]
            images = tf.stack([preprocess_crop(load_crop(row), target_size) for row in batch.to_dict('records')])
            maps, _, valid = _gradcam_batch(grad_model, images, baseline['target_classes'][start:start + len(batch)])
            stage_maps.append(maps)
            for j, row in enumerate(batch.to_dict('records')):
                index = start + j
                a = resize_heatmap(baseline['heatmaps'][index], target_size)
                b = resize_heatmap(maps[j], target_size)
                records.append({'sample_id': row['sample_id'], 'category': row['category'], 'stage': stage,
                    'target_class': int(baseline['target_classes'][index]),
                    'baseline_informative': bool(baseline['informative'][index]),
                    'randomized_informative': bool(valid[j]), **compare_heatmaps(a, b)})
            if start == 0 or (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(samples):
                print(f'Sanity {stage}: {min(start + batch_size, len(samples))}/{len(samples)}')
        stage_maps = np.concatenate(stage_maps)
        previews[stage] = stage_maps[:min(6, len(samples))]
        if output_dir is not None:
            np.savez_compressed(output_dir / f'{stage}_heatmaps.npz', sample_ids=baseline['sample_ids'], heatmaps=stage_maps)
            pd.DataFrame(records).to_csv(output_dir / 'sanity_per_sample.csv', index=False)
    results = pd.DataFrame(records)
    summary = results.groupby('stage', sort=False).agg(
        mean_spearman=('spearman', 'mean'), mean_mae=('mae', 'mean'),
        valid_correlations=('spearman', 'count'), samples=('sample_id', 'size'),
        randomized_informative_fraction=('randomized_informative', 'mean')).reset_index()
    if output_dir is not None:
        summary.to_csv(output_dir / 'sanity_summary.csv', index=False)
        save_json(output_dir / 'sanity_protocol.json', {'seed': seed, 'stages': [g[0] for g in groups],
            'randomization': 'cascading configured initializers including BatchNorm state',
            'target_mode': str(baseline['target_mode']), 'comparison_size': list(target_size),
            'layer_name': layer_name, 'samples': len(samples)})
    return results, summary, previews


def plot_sanity_results(samples, baseline, summary, previews, target_size=(224, 224)):
    stages = list(previews)
    n = min(6, len(samples))
    fig, axes = plt.subplots(n, 2 + len(stages), figsize=(3 * (2 + len(stages)), 3 * n), squeeze=False)
    for i, row in enumerate(samples.iloc[:n].to_dict('records')):
        crop = load_crop(row)
        axes[i, 0].imshow(crop); axes[i, 0].set_title(row['category'])
        for j, (name, heatmap) in enumerate([('Trained', baseline['heatmaps'][i])] +
                [(stage, previews[stage][i]) for stage in stages], 1):
            axes[i, j].imshow(resize_heatmap(heatmap, target_size), cmap='jet', vmin=0, vmax=1)
            axes[i, j].set_title(name)
        for ax in axes[i]: ax.axis('off')
    fig.tight_layout()
    metric_fig, metric_axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column in zip(metric_axes, ('mean_spearman', 'mean_mae')):
        ax.plot(summary.stage, summary[column], marker='o'); ax.set_ylabel(column)
        ax.tick_params(axis='x', rotation=30)
    metric_fig.tight_layout()
    return fig, metric_fig
