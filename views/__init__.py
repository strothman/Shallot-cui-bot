"""
Views Package for Shallot-CUI Bot.

Central re-exporter preserving 100% backward compatibility with the legacy views.py module.
"""

# Modals and Modal Dropdowns
from views.modals import (
    CustomSrefModal,
    RemixModal,
    EditBlendPromptModal,
    EditBlendKreaModal,
    StudyImagineModal,
    EditStyleModal,
    EditPromptModal,
    EditAdoptPromptModal,
    SavedSrefSelectView,
)

# Core Generation & Grid Lifecycle Views
from views.grid_views import (
    CancelGenerationView,
    IsolatedImageButtons,
    UpscaleButtons,
    GridButtons,
    StasisControlsView,
    StasisPausedView,
    BertflowButtons,
)

# SDXL Blend Studio
from views.blend_sdxl import (
    build_blend_embed,
    build_blend_complete_embed,
    build_blended_image_embed,
    BlendButtons,
)

# Krea 2 (Bertflow) Blend Studio
from views.blend_krea import (
    build_blend_krea_embed,
    BlendKreaButtons,
)

# Pagination, Browsing, and Utility Views
from views.pagination import (
    DescribeButtons,
    StudyButtons,
    StylePaginationView,
    PromptPaginationView,
    AdoptButtons,
)

# Native Persistent Dynamic Items
from views.dynamic_items import (
    CancelGenDynamicButton,
    IsolateDynamicButton,
    VariationDynamicButton,
    RerollDynamicButton,
    RemixDynamicButton,
    OutpaintDynamicButton,
)

__all__ = [
    # Modals
    "CustomSrefModal",
    "RemixModal",
    "EditBlendPromptModal",
    "EditBlendKreaModal",
    "StudyImagineModal",
    "EditStyleModal",
    "EditPromptModal",
    "EditAdoptPromptModal",
    "SavedSrefSelectView",
    
    # Grid & Generation Views
    "CancelGenerationView",
    "IsolatedImageButtons",
    "UpscaleButtons",
    "GridButtons",
    "StasisControlsView",
    "StasisPausedView",
    "BertflowButtons",
    
    # Blend SDXL
    "build_blend_embed",
    "build_blend_complete_embed",
    "build_blended_image_embed",
    "BlendButtons",
    
    # Blend Krea
    "build_blend_krea_embed",
    "BlendKreaButtons",
    
    # Pagination & Utility Views
    "DescribeButtons",
    "StudyButtons",
    "StylePaginationView",
    "PromptPaginationView",
    "AdoptButtons",
    
    # Native Persistent Dynamic Items
    "CancelGenDynamicButton",
    "IsolateDynamicButton",
    "VariationDynamicButton",
    "RerollDynamicButton",
    "RemixDynamicButton",
    "OutpaintDynamicButton",
]
