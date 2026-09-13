"""
Semantic Workflow Adapter & Node Decoupling Service.
Provides robust, role-based node discovery, inspection, and manipulation for ComfyUI workflows.
Permanently insulates Shallot-CUI Bot from arbitrary ComfyUI node renumbering.
"""

import copy
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

logger = logging.getLogger("DiscordBot.WorkflowAdapter")


def find_nodes(
    workflow: Dict[str, Any],
    class_type: Optional[Union[str, List[str]]] = None,
    title_contains: Optional[str] = None,
    has_input: Optional[Union[str, List[str]]] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Finds all nodes in a ComfyUI workflow matching the given semantic criteria.
    
    Returns:
        List of (node_id, node_dict) tuples.
    """
    if not isinstance(workflow, dict):
        return []

    target_classes = [class_type.lower()] if isinstance(class_type, str) else ([c.lower() for c in class_type] if class_type else None)
    target_inputs = [has_input.lower()] if isinstance(has_input, str) else ([i.lower() for i in has_input] if has_input else None)
    title_search = title_contains.lower() if title_contains else None

    matches = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue

        # 1. Class type match
        if target_classes:
            node_class = str(node.get("class_type", "")).lower()
            if not any(tc == node_class or tc in node_class for tc in target_classes):
                continue

        # 2. Title substring match
        if title_search:
            title = str(node.get("_meta", {}).get("title", "")).lower()
            if title_search not in title:
                continue

        # 3. Input keys match
        if target_inputs:
            inputs = node.get("inputs", {})
            input_keys = [str(k).lower() for k in inputs.keys()]
            if not all(ti in input_keys for ti in target_inputs):
                continue

        matches.append((str(node_id), node))

    return matches


def find_node(
    workflow: Dict[str, Any],
    class_type: Optional[Union[str, List[str]]] = None,
    title_contains: Optional[str] = None,
    has_input: Optional[Union[str, List[str]]] = None,
    fallback_id: Optional[str] = None
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Finds the first node matching the criteria, or falls back to fallback_id if available.
    """
    matches = find_nodes(workflow, class_type=class_type, title_contains=title_contains, has_input=has_input)
    if matches:
        return matches[0]

    if fallback_id and str(fallback_id) in workflow:
        return (str(fallback_id), workflow[str(fallback_id)])

    return None


def get_positive_negative_nodes(workflow: Dict[str, Any]) -> Tuple[Optional[Tuple[str, Dict[str, Any]]], Optional[Tuple[str, Dict[str, Any]]]]:
    """
    Semantically identifies the positive and negative CLIP text encode nodes.
    First tries tracing sampler 'positive' / 'negative' links.
    If not found, falls back to title inspection or class types.
    """
    pos_node = None
    neg_node = None

    # 1. Try tracing from KSampler
    samplers = find_nodes(workflow, class_type=["ksampler", "ksampleradvanced"])
    if samplers:
        sampler = samplers[0][1]
        inputs = sampler.get("inputs", {})
        pos_link = inputs.get("positive")
        neg_link = inputs.get("negative")

        if isinstance(pos_link, list) and len(pos_link) > 0:
            target_id = str(pos_link[0])
            # Trace through FluxGuidance or ConditioningSetTimestepRange if present
            if target_id in workflow:
                target_node = workflow[target_id]
                if target_node.get("class_type") == "FluxGuidance":
                    inner_link = target_node.get("inputs", {}).get("conditioning")
                    if isinstance(inner_link, list) and str(inner_link[0]) in workflow:
                        target_id = str(inner_link[0])
                        target_node = workflow[target_id]
                pos_node = (target_id, target_node)

        if isinstance(neg_link, list) and len(neg_link) > 0:
            target_id = str(neg_link[0])
            if target_id in workflow:
                neg_node = (target_id, workflow[target_id])

    # 2. If either is missing, inspect titles of CLIPTextEncode nodes
    if not pos_node or not neg_node:
        text_nodes = find_nodes(workflow, class_type=["cliptextencode", "cliptextencodesdxl"])
        for nid, node in text_nodes:
            title = str(node.get("_meta", {}).get("title", "")).lower()
            if "neg" in title:
                if not neg_node:
                    neg_node = (nid, node)
            else:
                if not pos_node:
                    pos_node = (nid, node)

    # 3. Fallback defaults
    if not pos_node and "6" in workflow:
        pos_node = ("6", workflow["6"])
    if not neg_node and "7" in workflow:
        neg_node = ("7", workflow["7"])

    return pos_node, neg_node


def set_workflow_prompt(
    workflow: Dict[str, Any],
    positive: Optional[str] = None,
    negative: Optional[str] = None
) -> bool:
    """
    Semantically sets the positive and/or negative prompt text in the workflow.
    """
    pos_match, neg_match = get_positive_negative_nodes(workflow)
    success = False

    if positive is not None and pos_match:
        pos_match[1].setdefault("inputs", {})["text"] = positive
        success = True

    if negative is not None and neg_match:
        neg_match[1].setdefault("inputs", {})["text"] = negative
        success = True

    return success


def set_workflow_seed(workflow: Dict[str, Any], seed: int) -> int:
    """
    Updates the seed or noise_seed across all sampler and dedicated seed nodes in the workflow.
    Returns the count of nodes updated.
    """
    updated_count = 0

    # 1. Update any KSampler / KSamplerAdvanced nodes
    samplers = find_nodes(workflow, class_type=["ksampler", "ksampleradvanced"])
    for _, node in samplers:
        inputs = node.setdefault("inputs", {})
        if "seed" in inputs or "noise_seed" not in inputs:
            inputs["seed"] = seed
            updated_count += 1
        elif "noise_seed" in inputs and not isinstance(inputs["noise_seed"], list):
            inputs["noise_seed"] = seed
            updated_count += 1

    # 2. Update dedicated Seed nodes (e.g. Seed (rgthree))
    seed_nodes = find_nodes(workflow, has_input="seed")
    for nid, node in seed_nodes:
        inputs = node.setdefault("inputs", {})
        if "seed" in inputs and not isinstance(inputs["seed"], list):
            inputs["seed"] = seed
            updated_count += 1

    # 3. Fallbacks if no nodes matched
    if updated_count == 0:
        for fallback_id in ["3", "11", "15", "85", "649"]:
            if fallback_id in workflow and "inputs" in workflow[fallback_id]:
                workflow[fallback_id]["inputs"]["seed"] = seed
                updated_count += 1

    return updated_count


def set_workflow_dimensions(
    workflow: Dict[str, Any],
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch_size: Optional[int] = None
) -> bool:
    """
    Updates width, height, and batch size on latent generator nodes.
    """
    latent_nodes = find_nodes(workflow, class_type=["emptylatentimage", "emptysd3latentimage", "wanlatent"])
    if not latent_nodes:
        # Fallback to checking has_input width and height
        latent_nodes = find_nodes(workflow, has_input=["width", "height"])

    success = False
    for _, node in latent_nodes:
        inputs = node.setdefault("inputs", {})
        if width is not None and "width" in inputs:
            inputs["width"] = width
            success = True
        if height is not None and "height" in inputs:
            inputs["height"] = height
            success = True
        if batch_size is not None and "batch_size" in inputs:
            inputs["batch_size"] = batch_size
            success = True

    # Fallback default node 5
    if not success and "5" in workflow and "inputs" in workflow["5"]:
        if width is not None:
            workflow["5"]["inputs"]["width"] = width
        if height is not None:
            workflow["5"]["inputs"]["height"] = height
        if batch_size is not None:
            workflow["5"]["inputs"]["batch_size"] = batch_size
        return True

    return success


def set_workflow_checkpoint(
    workflow: Dict[str, Any],
    ckpt_name: Optional[str] = None,
    unet_name: Optional[str] = None
) -> bool:
    """
    Sets the checkpoint or UNet model name on loader nodes.
    """
    success = False

    if ckpt_name:
        loaders = find_nodes(workflow, class_type="checkpointloadersimple")
        if not loaders:
            loaders = find_nodes(workflow, has_input="ckpt_name")
        for _, node in loaders:
            node.setdefault("inputs", {})["ckpt_name"] = ckpt_name
            success = True

    if unet_name:
        loaders = find_nodes(workflow, class_type=["unetloadergguf", "diffusionmodelloader"])
        if not loaders:
            loaders = find_nodes(workflow, has_input="unet_name")
        for _, node in loaders:
            node.setdefault("inputs", {})["unet_name"] = unet_name
            success = True

    return success


def set_workflow_sampler_params(
    workflow: Dict[str, Any],
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    denoise: Optional[float] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None
) -> bool:
    """
    Updates sampler parameters across all KSampler nodes.
    """
    samplers = find_nodes(workflow, class_type=["ksampler", "ksampleradvanced"])
    success = False

    for _, node in samplers:
        inputs = node.setdefault("inputs", {})
        if steps is not None and "steps" in inputs:
            inputs["steps"] = steps
            success = True
        if cfg is not None and "cfg" in inputs:
            inputs["cfg"] = cfg
            success = True
        if denoise is not None and "denoise" in inputs:
            inputs["denoise"] = denoise
            success = True
        if sampler_name is not None and "sampler_name" in inputs:
            inputs["sampler_name"] = sampler_name
            success = True
        if scheduler is not None and "scheduler" in inputs:
            inputs["scheduler"] = scheduler
            success = True

    return success


def set_workflow_input_image(
    workflow: Dict[str, Any],
    image_filename: str,
    title_hint: Optional[str] = None
) -> bool:
    """
    Sets the uploaded image filename on LoadImage nodes.
    """
    match = find_node(workflow, class_type="loadimage", title_contains=title_hint, fallback_id="1")
    if match:
        match[1].setdefault("inputs", {})["image"] = image_filename
        return True
    return False


def set_workflow_filename_prefix(
    workflow: Dict[str, Any],
    prefix: str
) -> bool:
    """
    Updates the output filename prefix on save or export nodes.
    """
    save_nodes = find_nodes(workflow, has_input="filename_prefix")
    success = False
    for _, node in save_nodes:
        node.setdefault("inputs", {})["filename_prefix"] = prefix
        success = True
    return success


def set_workflow_lora(
    workflow: Dict[str, Any],
    lora_name: str,
    strength_model: float = 1.0,
    strength_clip: Optional[float] = None,
    title_hint: Optional[str] = None
) -> bool:
    """
    Locates or configures a LoRA loader node in the workflow.
    Matches by title_hint or by existing lora_name input.
    """
    if strength_clip is None:
        strength_clip = strength_model

    lora_nodes = find_nodes(workflow, class_type=["loraloader", "loraloadermodelonly"])
    for _, node in lora_nodes:
        inputs = node.setdefault("inputs", {})
        title = str(node.get("_meta", {}).get("title", "")).lower()

        # Check title hint or matching lora_name substring
        if (title_hint and title_hint.lower() in title) or (lora_name and lora_name.lower() in str(inputs.get("lora_name", "")).lower()):
            inputs["lora_name"] = lora_name
            inputs["strength_model"] = strength_model
            if "strength_clip" in inputs:
                inputs["strength_clip"] = strength_clip
            return True

    return False
