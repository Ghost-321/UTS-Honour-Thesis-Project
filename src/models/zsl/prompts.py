def get_prompts(kind="no_vs_mild", ensemble=True):
    if kind == "no_vs_mild":
        return {
            "no_dr": [
                "A retinal fundus image with no signs of diabetic retinopathy.",
                "Fundus photo without lesions, hemorrhages, or exudates.",
                "Healthy retina without microaneurysms or neovascularisation.",
                "Normal colour fundus photograph with clear vessels and optic disc.",
                "No evidence of diabetic retinopathy in this retinal image."
            ],
            "mild": [
                "Fundus image with mild non-proliferative diabetic retinopathy showing few microaneurysms.",
                "Retinal photograph with sparse microaneurysms and no hard exudates.",
                "Mild diabetic retinopathy with small dot hemorrhages visible.",
                "Early stage DR with initial microaneurysms and minimal changes.",
                "Mild non-proliferative DR, no neovascularisation."
            ],
        }

    if kind == "5-class":
        return {
            "no_dr": ["Fundus image with no diabetic retinopathy."],
            "mild": ["Mild non-proliferative diabetic retinopathy with few microaneurysms."],
            "moderate": ["Moderate diabetic retinopathy with microaneurysms and dot-blot hemorrhages."],
            "severe": ["Severe non-proliferative diabetic retinopathy with numerous hemorrhages and venous beading."],
            "proliferative": ["Proliferative diabetic retinopathy with neovascularization."]
        }

    if kind == "binary_idrid":
        # Simplified binary classification for DR presence
        return {
            "no_dr": [
                "A healthy retinal fundus image with no signs of diabetic retinopathy.",
                "Normal retina without microaneurysms, hemorrhages, or exudates.",
                "Fundus photograph showing no abnormalities.",
                "Clear retinal image with intact blood vessels and optic disc.",
                "No evidence of diabetic retinopathy."
            ],
            "dr": [
                "A retinal image showing signs of diabetic retinopathy.",
                "Fundus photograph with lesions, hemorrhages, or exudates.",
                "Diabetic retinopathy with visible microaneurysms and abnormalities.",
                "Retina showing pathologic features due to diabetes.",
                "Presence of diabetic retinopathy changes in the fundus."
            ],
        }
    
    if kind == "binary_aptos":
        # Binary DR prompts for APTOS dataset (cross-domain ZSL)
        return {
            "no_dr": [
                "A healthy retinal fundus image with no signs of diabetic retinopathy.",
                "Normal retina without microaneurysms, hemorrhages, or exudates.",
                "Fundus photograph showing no abnormalities.",
                "Clear retinal image with intact vessels and optic disc.",
                "No evidence of diabetic retinopathy in this retinal image."
            ],
            "dr": [
                "A retinal fundus image showing signs of diabetic retinopathy.",
                "Fundus photograph with visible microaneurysms, hemorrhages, or exudates.",
                "An image of the retina with diabetic lesions or vascular damage.",
                "Retina showing pathological changes caused by diabetes.",
                "Presence of diabetic retinopathy features such as hemorrhages or exudates."
            ],
        }


    raise ValueError(f"Unknown prompt set: {kind}")
