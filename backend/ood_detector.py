# ================================================================
# RETINASCAN AI — Out-of-Distribution (OOD) Detector
# ================================================================
# Feature-space OOD detection using ConvNeXt-Tiny penultimate layer.
#
# Strategy:
#   1. Extract feature vectors from the model's penultimate layer
#      (after global average pooling, before classifier head).
#   2. At startup, store individual reference feature vectors from
#      the demo OCT images (one per class).
#   3. At inference time, compute cosine similarity between the
#      test image features and EACH reference vector, then use the
#      MAXIMUM similarity (nearest-class match).
#   4. Combine with OCT-specific structural analysis (vertical
#      intensity profile, speckle noise pattern).
#   5. Apply a multi-signal gating check: the image must satisfy
#      multiple OCT-characteristic tests simultaneously.
#
# KEY INSIGHT: Because ConvNeXt is trained for 4-way classification,
# the 4 class clusters are maximally separated in feature space
# (negative cosine similarity between classes). A centroid-based
# approach fails because averaging vectors pointing in opposite
# directions produces a near-zero vector. Instead, we use the
# MAX similarity to any individual reference (nearest-class match).
#
# This does NOT modify the model or retrain anything.
# ================================================================

import numpy as np
import cv2
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F


# ================================================================
# FEATURE EXTRACTION HOOK
# ================================================================
class FeatureExtractor:
    """Extract features from ConvNeXt penultimate layer."""

    def __init__(self, model):
        self.model = model
        self.features = None
        # Hook into the adaptive average pool (classifier[0] is LayerNorm,
        # classifier[1] is Flatten — we hook after features[-1])
        self._hook = model.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features = output.detach()

    def extract(self, input_tensor):
        """Run forward pass and return pooled feature vector."""
        with torch.no_grad():
            _ = self.model(input_tensor)
        # features shape: (1, C, 1, 1) after avgpool
        return self.features.flatten(1)  # (1, C)

    def cleanup(self):
        self._hook.remove()


# ================================================================
# OCT STRUCTURAL ANALYSIS
# ================================================================
def analyze_oct_structure(image_array):
    """
    Analyze whether the image has OCT B-scan structural characteristics.

    OCT B-scans have a distinctive pattern:
      - Horizontal layered structure (retinal layers)
      - Dark regions above (vitreous) and below (choroid/sclera)
      - Bright band in the middle (retinal layers)
      - Speckle noise texture characteristic of interferometry
      - Relatively uniform horizontal bands
      - High horizontal edge energy relative to vertical
      - Characteristic aspect ratio of intensity distribution

    Returns:
        dict with structural analysis scores
    """
    # Convert to grayscale if needed
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()

    # Resize to consistent size for analysis
    gray = cv2.resize(gray, (224, 224))

    scores = {}

    # --- Check 1: Vertical Intensity Profile ---
    vertical_profile = gray.mean(axis=1)
    profile_normalized = vertical_profile / (vertical_profile.max() + 1e-8)

    fifths = np.array_split(profile_normalized, 5)
    top_mean = (fifths[0].mean() + fifths[1].mean()) / 2
    mid_mean = fifths[2].mean()
    bot_mean = (fifths[3].mean() + fifths[4].mean()) / 2

    layer_contrast = mid_mean - (top_mean + bot_mean) / 2
    scores['layer_contrast'] = float(layer_contrast)

    profile_std = float(np.std(profile_normalized))
    scores['vertical_profile_std'] = profile_std

    # --- Check 2: Horizontal Edge Dominance ---
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    h_energy = np.sum(sobel_h ** 2)
    v_energy = np.sum(sobel_v ** 2)
    total_energy = h_energy + v_energy + 1e-10
    h_dominance = float(h_energy / total_energy)
    scores['h_edge_dominance'] = h_dominance

    # --- Check 3: Speckle Noise Measure ---
    blur = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
    local_var = np.mean((gray.astype(np.float64) - blur) ** 2)
    mean_intensity = gray.mean() + 1e-8
    speckle_index = float(np.sqrt(local_var) / mean_intensity)
    scores['speckle_index'] = speckle_index

    # --- Check 4: Row-wise Correlation ---
    row_correlations = []
    for i in range(0, gray.shape[0] - 1, 4):
        row_a = gray[i].astype(np.float64)
        row_b = gray[i + 1].astype(np.float64)
        if row_a.std() > 0 and row_b.std() > 0:
            corr = np.corrcoef(row_a, row_b)[0, 1]
            if not np.isnan(corr):
                row_correlations.append(corr)
    avg_row_corr = float(np.mean(row_correlations)) if row_correlations else 0.0
    scores['row_correlation'] = avg_row_corr

    # --- Check 5: Intensity Distribution ---
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    dark_fraction = float(hist[:80].sum())
    scores['dark_fraction'] = dark_fraction

    # --- Check 6: Horizontal Band Continuity ---
    row_edge_energies = np.sum(np.abs(sobel_h), axis=1)
    total_row_energy = np.sum(row_edge_energies) + 1e-10
    sorted_energies = np.sort(row_edge_energies)[::-1]
    top_30pct_rows = int(0.3 * len(sorted_energies))
    energy_concentration = float(
        np.sum(sorted_energies[:top_30pct_rows]) / total_row_energy
    )
    scores['edge_concentration'] = energy_concentration

    # --- Check 7: Texture Complexity ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mean = float(np.mean(np.abs(laplacian)))
    lap_std = float(np.std(laplacian))
    texture_ratio = lap_std / (lap_mean + 1e-8)
    scores['texture_ratio'] = texture_ratio

    # --- Check 8: Vertical Symmetry of Edge Pattern ---
    top_half_edges = np.sum(np.abs(sobel_h[:112]))
    bot_half_edges = np.sum(np.abs(sobel_h[112:]))
    total_half = top_half_edges + bot_half_edges + 1e-10
    edge_balance = 1.0 - abs(top_half_edges - bot_half_edges) / total_half
    scores['edge_balance'] = float(edge_balance)

    return scores


def compute_structural_oct_score(structural_scores):
    """
    Combine structural analysis scores into a single OCT likelihood score.
    Calibrated from real OCT demo images and various non-OCT images.

    Returns:
        float: 0.0 (not OCT-like) to 1.0 (very OCT-like)
    """
    score = 0.0
    total_weight = 0.0

    # Layer contrast (OCT should have positive contrast — bright middle band)
    # NOTE: Some OCT classes (DRUSEN) can have negative layer_contrast
    w1 = 2.0
    lc = structural_scores.get('layer_contrast', 0)
    if lc > 0.08:
        score += w1 * min(lc / 0.18, 1.0)
    elif lc > -0.05:
        score += w1 * 0.4
    elif lc > -0.20:
        score += w1 * 0.2
    total_weight += w1

    # Horizontal edge dominance (OCT typically > 0.60)
    w2 = 2.5
    he = structural_scores.get('h_edge_dominance', 0.5)
    if he > 0.60:
        score += w2 * min((he - 0.50) / 0.15, 1.0)
    elif he > 0.50:
        score += w2 * 0.2
    total_weight += w2

    # Speckle index (OCT has moderate speckle, roughly 0.15-0.35)
    # This is key: B&W photos have very low speckle (<0.05)
    w3 = 2.5
    si = structural_scores.get('speckle_index', 0)
    if 0.12 <= si <= 0.45:
        speckle_score = 1.0 - abs(si - 0.25) / 0.25
        score += w3 * max(0, speckle_score)
    elif 0.08 <= si < 0.12:
        score += w3 * 0.15  # very slight credit for borderline
    total_weight += w3

    # Row correlation (OCT: 0.45-0.90, photos: 0.0 or 0.99+)
    w4 = 2.0
    rc = structural_scores.get('row_correlation', 0)
    if 0.30 <= rc <= 0.92:
        # OCT has moderate-to-high row correlation
        score += w4 * min((rc - 0.20) / 0.50, 1.0)
    total_weight += w4

    # Dark fraction (OCT images usually have > 40% dark pixels)
    w5 = 1.0
    df = structural_scores.get('dark_fraction', 0)
    if df > 0.35:
        score += w5 * min((df - 0.25) / 0.40, 1.0)
    total_weight += w5

    # Edge concentration (OCT has concentrated edge energy)
    w6 = 1.5
    ec = structural_scores.get('edge_concentration', 0)
    if ec > 0.50:
        score += w6 * min((ec - 0.45) / 0.25, 1.0)
    total_weight += w6

    # Vertical profile variation (OCT has high vertical variation, > 0.12)
    w7 = 1.0
    vpstd = structural_scores.get('vertical_profile_std', 0)
    if vpstd > 0.12:
        score += w7 * min((vpstd - 0.08) / 0.20, 1.0)
    total_weight += w7

    return float(score / total_weight) if total_weight > 0 else 0.0


def count_oct_indicators(structural_details):
    """
    Count how many OCT-characteristic indicators the image satisfies.
    An image must satisfy MULTIPLE indicators to be considered OCT.
    
    This is a multi-signal gating check that makes it very hard for
    non-OCT images to pass — they'd need to simultaneously match
    several OCT-specific characteristics.
    
    Calibrated from measurements:
        OCT speckle_idx: 0.16-0.28 | Photos: <0.05 | Noise: 0.35
        OCT h_edge_dom:  0.62-0.89 | Photos: varies
        OCT dark_frac:   0.74-0.82 | Photos: varies  
        OCT row_corr:    0.49-0.86 | Lines: 0.0 | Photos: varies
        OCT edge_conc:   0.51-0.67 | Non-OCT: <0.45
        OCT texture_r:   1.40-1.57 | Smooth: >2.0
    
    Returns:
        (int, list): count of passed indicators and list of indicator names
    """
    passed = []
    
    # 1. OCT-characteristic speckle noise (interferometric speckle)
    si = structural_details.get('speckle_index', 0)
    if 0.10 <= si <= 0.45:
        passed.append('speckle_noise')
    
    # 2. Horizontal edge dominance (retinal layers are horizontal)
    he = structural_details.get('h_edge_dominance', 0)
    if he > 0.58:
        passed.append('horizontal_edges')
    
    # 3. Dark-dominant intensity (vitreous + choroid are dark)
    df = structural_details.get('dark_fraction', 0)
    if df > 0.40:
        passed.append('dark_dominant')
    
    # 4. Moderate row correlation (layers create correlated rows)
    # OCT: 0.49-0.86, NOT 0.0 (random/lines) or 0.99 (synthetic smooth)
    rc = structural_details.get('row_correlation', 0)
    if 0.25 <= rc <= 0.95:
        passed.append('row_correlation')
    
    # 5. Concentrated edge energy (edges in retinal layer band)
    ec = structural_details.get('edge_concentration', 0)
    if ec > 0.48:
        passed.append('edge_concentration')
    
    # 6. OCT-like texture complexity (not too smooth, not pure noise)
    tr = structural_details.get('texture_ratio', 0)
    if 1.2 <= tr <= 2.0:
        passed.append('texture_complexity')
    
    # 7. Vertical intensity variation (dark-bright-dark profile)
    vpstd = structural_details.get('vertical_profile_std', 0)
    if vpstd > 0.12:
        passed.append('vertical_variation')
    
    return len(passed), passed


# ================================================================
# COMBINED OOD DETECTION
# ================================================================
class OCTDomainValidator:
    """
    Combined OOD detector using:
      1. Feature-space cosine similarity (ConvNeXt penultimate layer)
      2. OCT structural analysis (texture/pattern checks)
      3. Multi-signal gating (must satisfy multiple OCT indicators)

    The feature extractor compares the input against reference OCT images.
    The structural analysis checks for OCT-specific image patterns.
    The multi-signal gating ensures non-OCT images can't slip through
    by coincidentally matching one or two OCT characteristics.

    IMPORTANT: Because the classifier maps different OCT classes to
    opposite corners of feature space (negative inter-class cosine
    similarity), we use MAX similarity to any individual reference
    rather than centroid similarity.
    """

    # Thresholds calibrated from comprehensive testing:
    # Real OCT structural scores: 0.61-0.73
    # Non-OCT (car photo): ~0.44 combined
    # Synthetic adversarial: 0.55 combined (horizontal lines)
    STRUCTURAL_SCORE_THRESHOLD = 0.45
    COMBINED_THRESHOLD = 0.50
    MIN_OCT_INDICATORS = 5  # Must satisfy at least 5 of 7 OCT indicators

    def __init__(self, model, preprocess_fn, device):
        self.device = device
        self.preprocess_fn = preprocess_fn
        self.feature_extractor = FeatureExtractor(model)
        self.reference_centroid = None
        self._initialized = False

    def initialize_reference(self, demo_image_dir):
        """
        Store individual reference feature vectors from demo OCT images.
        Called once at startup.
        """
        demo_dir = Path(demo_image_dir)
        if not demo_dir.exists():
            print("[OOD] Warning: Demo image directory not found. "
                  "Feature-space validation will be limited.")
            return

        feature_vectors = []
        self._reference_names = []
        for img_path in sorted(demo_dir.iterdir()):
            if img_path.suffix.lower() in ('.jpeg', '.jpg', '.png'):
                try:
                    pil_img = Image.open(str(img_path)).convert("L")
                    input_tensor = self.preprocess_fn(pil_img).unsqueeze(0).to(self.device)
                    features = self.feature_extractor.extract(input_tensor)
                    feature_vectors.append(features)
                    self._reference_names.append(img_path.name)
                    print(f"[OOD] Registered reference: {img_path.name}")
                except Exception as e:
                    print(f"[OOD] Warning: Could not process {img_path.name}: {e}")

        if feature_vectors:
            stacked = torch.cat(feature_vectors, dim=0)
            self.reference_vectors = stacked
            self.reference_centroid = stacked.mean(dim=0, keepdim=True)
            self._initialized = True
            print(f"[OOD] Reference centroid computed from {len(feature_vectors)} images "
                  f"(feature dim: {stacked.shape[1]})")
        else:
            print("[OOD] Warning: No valid reference images found.")

    def compute_feature_similarity(self, input_tensor):
        """
        Compute cosine similarity between input and OCT reference images.
        Uses MAX similarity to any individual reference vector.

        Returns:
            float: max cosine similarity to any reference (0 to 1)
        """
        if not self._initialized:
            return 1.0

        features = self.feature_extractor.extract(input_tensor)

        individual_sims = F.cosine_similarity(
            features, self.reference_vectors
        )

        max_sim = individual_sims.max().item()
        return float(max_sim)

    def validate(self, image_array, input_tensor):
        """
        Full OOD validation pipeline with multi-signal gating.

        Args:
            image_array: numpy array (H, W, C) in RGB format
            input_tensor: preprocessed PyTorch tensor for the model

        Returns:
            dict with validation results
        """
        # 1. Feature-space similarity (max to any reference)
        feature_sim = self.compute_feature_similarity(input_tensor)

        # 2. Structural analysis
        structural_details = analyze_oct_structure(image_array)
        structural_score = compute_structural_oct_score(structural_details)

        # 3. Multi-signal gating: count OCT indicators
        indicator_count, passed_indicators = count_oct_indicators(structural_details)

        # 4. Combined score
        # Structural analysis is the PRIMARY signal for OOD detection.
        if self._initialized:
            combined = 0.35 * feature_sim + 0.65 * structural_score
        else:
            combined = structural_score

        # 5. Decision logic — MULTI-LAYER defense
        is_valid = True
        rejection_reason = None

        # Layer A: Structural score threshold
        if structural_score < self.STRUCTURAL_SCORE_THRESHOLD:
            is_valid = False
            rejection_reason = (
                "The uploaded image does not appear to be a retinal OCT B-scan. "
                "It lacks the expected layered structure and texture patterns."
            )

        # Layer B: Combined threshold
        if combined < self.COMBINED_THRESHOLD:
            is_valid = False
            if rejection_reason is None:
                rejection_reason = (
                    "The uploaded image does not appear to be a valid retinal "
                    "OCT B-scan based on domain analysis."
                )

        # Layer C: Multi-signal gating — MOST IMPORTANT
        # The image must satisfy at least MIN_OCT_INDICATORS of 7
        # OCT-characteristic tests. This catches non-OCT images that
        # happen to score moderately well on individual metrics.
        if indicator_count < self.MIN_OCT_INDICATORS:
            is_valid = False
            if rejection_reason is None:
                rejection_reason = (
                    "The uploaded image does not satisfy enough OCT B-scan "
                    "characteristics. A valid retinal OCT image must show "
                    "characteristic speckle noise, horizontal layered structure, "
                    "and appropriate intensity distribution."
                )

        # Layer D: Hard structural rejects

        # OCT images MUST have horizontal edge dominance > 0.45
        if structural_details.get('h_edge_dominance', 0) < 0.45:
            is_valid = False
            rejection_reason = (
                "The uploaded image lacks the horizontal layered structure "
                "expected in a retinal OCT B-scan."
            )

        # OCT images MUST have meaningful vertical intensity variation
        if structural_details.get('vertical_profile_std', 0) < 0.08:
            is_valid = False
            rejection_reason = (
                "The uploaded image lacks the characteristic vertical "
                "intensity variation found in OCT B-scans."
            )

        # Edge concentration hard reject (all valid OCT > 0.50)
        edge_conc = structural_details.get('edge_concentration', 0)
        if edge_conc < 0.45:
            is_valid = False
            rejection_reason = (
                "The uploaded image does not show the concentrated retinal "
                "layer pattern expected in OCT B-scans."
            )

        # Compound: negative layer contrast + low edge concentration
        layer_c = structural_details.get('layer_contrast', 0)
        if edge_conc < 0.50 and layer_c < -0.15:
            is_valid = False
            rejection_reason = (
                "The uploaded image does not exhibit the retinal layer "
                "structure expected in OCT B-scans."
            )

        return {
            'is_valid_oct': is_valid,
            'feature_similarity': round(feature_sim, 4),
            'structural_score': round(structural_score, 4),
            'combined_score': round(combined, 4),
            'rejection_reason': rejection_reason,
            'structural_details': structural_details,
            'oct_indicator_count': indicator_count,
            'oct_indicators_passed': passed_indicators,
        }
