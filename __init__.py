try:
    # ComfyUI loads custom_nodes/<folder> as a sub-package → relative import works
    from .yaml_prompt.node import YAMLPromptLoader
    from .yaml_prompt.node_minimax_h3 import YAMLPromptLoaderMiniMaxH3
    from .yaml_prompt.apply_lora import ApplyLoraStack
except ImportError:
    # Standalone / pytest context → no parent package, use absolute import
    from yaml_prompt.node import YAMLPromptLoader
    from yaml_prompt.node_minimax_h3 import YAMLPromptLoaderMiniMaxH3
    from yaml_prompt.apply_lora import ApplyLoraStack

NODE_CLASS_MAPPINGS = {
    "YAMLPromptParser": YAMLPromptLoader,
    "YAMLPromptMiniMaxH3": YAMLPromptLoaderMiniMaxH3,
    "YAMLApplyLoraStack": ApplyLoraStack,
}
