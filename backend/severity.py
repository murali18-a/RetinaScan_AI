# ================================================================
# RETINASCAN AI — Feature-Based Disease Severity Assessment
# ================================================================
# Provides disease-specific OCT feature checklists and prototype
# severity categories based on clinician-reviewed feature counts.
#
# IMPORTANT:
#   - This module does NOT automatically detect OCT biomarkers.
#   - Features are intended to be reviewed/checked by a clinician
#     or trained reviewer examining the OCT image.
#   - Severity categories are research prototypes, NOT clinically
#     validated grades.
#   - This module does NOT override or modify model predictions.
# ================================================================

# ================================================================
# DISEASE-SPECIFIC FEATURE DEFINITIONS
# ================================================================

DISEASE_FEATURES = {
    "DME": {
        "label": "Diabetic Macular Edema — OCT Features",
        "description": (
            "The following features are commonly observed in OCT B-scans "
            "of eyes with diabetic macular edema. Review the scan and "
            "indicate which features are visible."
        ),
        "features": [
            {
                "id": "intraretinal_cysts",
                "name": "Intraretinal cysts",
                "description": (
                    "Round or oval hyporeflective spaces within the "
                    "neurosensory retina, indicating fluid accumulation."
                ),
            },
            {
                "id": "retinal_thickening",
                "name": "Retinal thickening",
                "description": (
                    "Increased retinal thickness compared to normal "
                    "macular profile, often visible as a loss of the "
                    "foveal depression."
                ),
            },
            {
                "id": "subretinal_fluid",
                "name": "Subretinal fluid",
                "description": (
                    "Hyporeflective space between the neurosensory "
                    "retina and the retinal pigment epithelium (RPE)."
                ),
            },
            {
                "id": "dril",
                "name": "Disorganisation of retinal inner layers (DRIL)",
                "description": (
                    "Loss of distinct boundaries between the ganglion "
                    "cell layer, inner plexiform layer, and inner "
                    "nuclear layer."
                ),
            },
            {
                "id": "ez_disruption",
                "name": "Ellipsoid zone disruption",
                "description": (
                    "Discontinuity or absence of the ellipsoid zone "
                    "(previously called the IS/OS junction), suggesting "
                    "photoreceptor damage."
                ),
            },
        ],
    },
    "CNV": {
        "label": "Choroidal Neovascularization — OCT Features",
        "description": (
            "The following features are commonly observed in OCT B-scans "
            "of eyes with choroidal neovascularization (wet AMD). Review "
            "the scan and indicate which features are visible."
        ),
        "features": [
            {
                "id": "ped",
                "name": "Pigment epithelial detachment (PED)",
                "description": (
                    "Elevation or separation of the RPE from Bruch's "
                    "membrane, appearing as a dome-shaped hyperreflective "
                    "elevation."
                ),
            },
            {
                "id": "intraretinal_fluid",
                "name": "Intraretinal fluid",
                "description": (
                    "Hyporeflective cystic spaces within the retinal "
                    "layers, indicating fluid leakage from abnormal "
                    "vessels."
                ),
            },
            {
                "id": "subretinal_fluid_cnv",
                "name": "Subretinal fluid",
                "description": (
                    "Hyporeflective space between the neurosensory "
                    "retina and the RPE, often associated with active "
                    "neovascularization."
                ),
            },
            {
                "id": "shrm",
                "name": "Subretinal hyperreflective material (SHRM)",
                "description": (
                    "Hyperreflective material between the retina and "
                    "RPE, which may represent fibrin, blood, or "
                    "neovascular tissue."
                ),
            },
            {
                "id": "outer_retinal_changes",
                "name": "Outer retinal changes",
                "description": (
                    "Disruption, thinning, or loss of the outer nuclear "
                    "layer, external limiting membrane, or ellipsoid "
                    "zone."
                ),
            },
        ],
    },
    "DRUSEN": {
        "label": "Drusen Deposits — OCT Features",
        "description": (
            "The following features are commonly observed in OCT B-scans "
            "of eyes with drusen (dry AMD). Review the scan and indicate "
            "which features are visible."
        ),
        "features": [
            {
                "id": "rpe_elevations",
                "name": "RPE elevations",
                "description": (
                    "Focal elevations of the retinal pigment epithelium "
                    "caused by sub-RPE deposits."
                ),
            },
            {
                "id": "drusen_size",
                "name": "Drusen of notable size",
                "description": (
                    "Individual drusen deposits that appear medium or "
                    "large (occupying a visible area beneath the RPE)."
                ),
            },
            {
                "id": "drusen_distribution",
                "name": "Wide drusen distribution",
                "description": (
                    "Drusen visible across a broad area of the macula, "
                    "not limited to a single focal point."
                ),
            },
            {
                "id": "drusen_confluence",
                "name": "Drusen confluence",
                "description": (
                    "Multiple drusen merging into larger, continuous "
                    "deposits rather than remaining as separate, "
                    "distinct elevations."
                ),
            },
            {
                "id": "reticular_pseudodrusen",
                "name": "Reticular pseudodrusen",
                "description": (
                    "Subretinal drusenoid deposits located above the "
                    "RPE, appearing as triangular or conical "
                    "hyperreflective lesions."
                ),
            },
        ],
    },
    "NORMAL": {
        "label": "Normal Retina — Structural Integrity",
        "description": (
            "For a scan classified as Normal, the following structural "
            "integrity indicators describe the expected appearance of a "
            "healthy retina."
        ),
        "features": [
            {
                "id": "preserved_layers",
                "name": "Preserved retinal layer organisation",
                "description": (
                    "Distinct, well-defined retinal layers visible "
                    "throughout the scan."
                ),
            },
            {
                "id": "intact_foveal_contour",
                "name": "Intact foveal contour",
                "description": (
                    "Normal foveal depression visible in the central "
                    "macular region."
                ),
            },
            {
                "id": "no_disease_features",
                "name": "No obvious disease-associated features",
                "description": (
                    "No visible fluid, cysts, drusen deposits, PED, "
                    "or other pathological features."
                ),
            },
        ],
    },
}


# ================================================================
# SEVERITY CATEGORIES
# ================================================================
# These are prototype research categories based on the number of
# disease-associated features observed. They are NOT clinically
# validated severity grades.

SEVERITY_LEVELS = {
    "unable_to_assess": {
        "label": "Unable to assess",
        "description": (
            "No features were reviewed, or the image quality does not "
            "permit reliable feature identification."
        ),
        "css_class": "badge-warning",
    },
    "low": {
        "label": "Low feature burden",
        "description": (
            "A small number of disease-associated features were "
            "identified. This is a prototype research categorisation."
        ),
        "css_class": "badge-success",
    },
    "moderate": {
        "label": "Moderate feature burden",
        "description": (
            "Several disease-associated features were identified. "
            "This is a prototype research categorisation."
        ),
        "css_class": "badge-warning",
    },
    "high": {
        "label": "High feature burden",
        "description": (
            "Most or all disease-associated features were identified. "
            "This is a prototype research categorisation."
        ),
        "css_class": "badge-error",
    },
}


def compute_severity(prediction, checked_feature_ids):
    """
    Compute a prototype severity category from checked features.

    Args:
        prediction: str — predicted class (CNV, DME, DRUSEN, NORMAL)
        checked_feature_ids: list[str] — IDs of features the reviewer
            marked as present

    Returns:
        dict with severity_level, severity_label, severity_description,
        checked_count, total_features, and the checked feature details.
    """
    if prediction not in DISEASE_FEATURES:
        return {
            "severity_level": "unable_to_assess",
            "severity_label": "Unable to assess",
            "severity_description": "Prediction class not recognised.",
            "checked_count": 0,
            "total_features": 0,
            "checked_features": [],
        }

    disease_def = DISEASE_FEATURES[prediction]
    all_features = disease_def["features"]
    total = len(all_features)

    # For NORMAL: severity doesn't apply in the same way
    if prediction == "NORMAL":
        checked = [f for f in all_features if f["id"] in checked_feature_ids]
        return {
            "severity_level": "normal",
            "severity_label": "Normal — no severity assessment applicable",
            "severity_description": (
                "Severity assessment applies to disease classes. For a "
                "Normal scan, the structural integrity indicators above "
                "describe the expected healthy retinal appearance."
            ),
            "checked_count": len(checked),
            "total_features": total,
            "checked_features": checked,
        }

    # For disease classes: count checked features
    checked = [f for f in all_features if f["id"] in checked_feature_ids]
    count = len(checked)

    if count == 0:
        level = "unable_to_assess"
    elif count <= 2:
        level = "low"
    elif count <= 3:
        level = "moderate"
    else:
        level = "high"

    severity = SEVERITY_LEVELS[level]

    return {
        "severity_level": level,
        "severity_label": severity["label"],
        "severity_description": severity["description"],
        "severity_css": severity["css_class"],
        "checked_count": count,
        "total_features": total,
        "checked_features": checked,
    }


def get_features_for_class(prediction):
    """Return feature definitions for a given prediction class."""
    if prediction in DISEASE_FEATURES:
        return DISEASE_FEATURES[prediction]
    return None

